"""
AI Service v3
- Gemini 2.5 Flash via API key (google-genai SDK)
- Imagen 3 inpainting per le foto staged (via vertexai SDK)
  NOTE: Imagen 3 deprecato il 24/06/2026 — migrare a Gemini Image quando disponibile
- validate_and_fix_costs(): validazione matematica budget post-Gemini
- compress_image(): compressione foto per PDF < 8MB
"""
import asyncio
import base64
import hashlib
import io
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

from google import genai
from google.genai import types

import vertexai
from vertexai.preview.vision_models import ImageGenerationModel, Image as VisionImage

try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("[WARNING] Pillow non installato — compressione immagini disabilitata")

PROJECT_ID       = os.environ["GCP_PROJECT_ID"]
LOCATION         = os.environ.get("GCP_LOCATION", "us-central1")
GEMINI_API_KEY   = os.environ["GEMINI_API_KEY"]
REPLICATE_TOKEN  = os.environ.get("REPLICATE_API_TOKEN", "")

# Gemini: usa API key diretta (non Vertex AI)
_genai_client = genai.Client(api_key=GEMINI_API_KEY)

# Imagen 3: usa ancora vertexai SDK
vertexai.init(project=PROJECT_ID, location=LOCATION)

_gemini_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="gemini")
_imagen_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="imagen")

_analysis_cache: dict[str, dict] = {}


# ── Utilità: compressione foto ────────────────────────────────────────────────

def compress_image(img_bytes: bytes, max_width: int = 1400, quality: int = 82) -> bytes:
    if not HAS_PIL:
        return img_bytes
    try:
        img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
        if img.width > max_width:
            ratio = max_width / img.width
            img   = img.resize((max_width, int(img.height * ratio)), PILImage.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()
    except Exception as e:
        print(f"[compress_image] fallback a originale: {e}")
        return img_bytes


def _cache_key(photos: list, prefs: dict) -> str:
    h = hashlib.md5()
    for p in photos:
        h.update(p["content"])
    h.update(prefs.get("style", "").encode())
    h.update(str(prefs.get("budget", 0)).encode())
    h.update(prefs.get("location", "").encode())
    return h.hexdigest()


# ── Validazione costi post-Gemini ─────────────────────────────────────────────

def validate_and_fix_costs(analysis: dict, budget: int) -> dict:
    stanze = analysis.get("stanze", [])
    rc     = analysis.setdefault("riepilogo_costi", {})

    # Invariante 1: costo_min <= costo_max
    for room in stanze:
        for iv in room.get("interventi", []):
            cmin = iv.get("costo_min", 0)
            cmax = iv.get("costo_max", 0)
            if cmin > cmax:
                iv["costo_min"], iv["costo_max"] = cmax, cmin

    # Invariante 2+3: somma stanze == totale dichiarato <= budget
    totale_stanze     = sum(r.get("costo_totale_stanza", 0) for r in stanze)
    totale_dichiarato = rc.get("totale", 0)
    totale_reale      = max(totale_stanze, totale_dichiarato)

    if totale_reale > budget or abs(totale_stanze - totale_dichiarato) > 50:
        target = int(budget * 0.95)
        factor = target / max(totale_reale, 1)

        for room in stanze:
            room["costo_totale_stanza"] = int(room["costo_totale_stanza"] * factor)
            for iv in room.get("interventi", []):
                iv["costo_min"] = int(iv["costo_min"] * factor)
                iv["costo_max"] = int(iv["costo_max"] * factor)

        for k in ["manodopera_tinteggiatura", "materiali_pittura",
                  "arredi_complementi", "montaggio_varie"]:
            rc[k] = int(rc.get(k, 0) * factor)

        rc["totale"]         = sum(r["costo_totale_stanza"] for r in stanze)
        rc["budget_residuo"] = budget - rc["totale"]
        nota = rc.get("nota_budget", "")
        rc["nota_budget"] = (nota + ". " if nota else "") + \
            f"Costi ricalibrati automaticamente per rispettare il budget di €{budget}."

    return analysis


# ── Gemini 2.5 Flash ──────────────────────────────────────────────────────────

async def analyze_with_gemini(photos: list, prefs: dict) -> dict:
    key = _cache_key(photos, prefs)
    if key in _analysis_cache:
        print("[Gemini] cache hit")
        return _analysis_cache[key]

    loop   = asyncio.get_running_loop()
    result = await loop.run_in_executor(_gemini_executor, _gemini_sync, photos, prefs)
    result = validate_and_fix_costs(result, prefs["budget"])
    _analysis_cache[key] = result
    return result


def _gemini_sync(photos: list, prefs: dict) -> dict:
    budget      = prefs["budget"]
    style       = prefs["style"]
    location    = prefs["location"]
    destination = prefs["destination"]
    dest_label  = "Airbnb / affitto breve (STR)" if destination == "STR" else "Casa vacanza"

    alloc = {
        "arredi":        int(budget * 0.40),
        "tinteggiatura": int(budget * 0.30),
        "materiali":     int(budget * 0.20),
        "montaggio":     int(budget * 0.10),
    }

    system_instruction = (
        "Sei un esperto consulente di home staging e interior design italiano, "
        "specializzato in affitti brevi e case vacanza. "
        "Conosci in dettaglio i prezzi di mercato delle principali città italiane "
        "per manodopera edile, materiali (Leroy Merlin, Brico) e arredi "
        "(IKEA, H&M Home, Zara Home, Maisons du Monde, mercatini, Amazon, Etsy). "
        "I prezzi devono riflettere il mercato reale della città indicata. "
        "Rispondi ESCLUSIVAMENTE con un oggetto JSON valido. "
        "Nessun testo prima o dopo. Nessun markdown. Nessun backtick."
    )

    image_parts = [
        types.Part.from_bytes(data=p["content"], mime_type=p["content_type"])
        for p in photos
    ]

    prompt = f"""Analizza queste {len(photos)} foto di un appartamento e produci
una scheda professionale di home staging per {dest_label}.

PARAMETRI:
- Budget totale:   €{budget}
- Stile richiesto: {style}
- Città:           {location}
- Destinazione:    {dest_label}

REGOLE PREZZI:
- Usa prezzi reali di {location}.
- Distribuzione consigliata:
    Arredi e complementi:       €{alloc['arredi']} (~40%)
    Tinteggiatura manodopera:   €{alloc['tinteggiatura']} (~30%)
    Materiali pittura/accessori:€{alloc['materiali']} (~20%)
    Montaggio e imprevisti:     €{alloc['montaggio']} (~10%)

REGOLA MATEMATICA CRITICA:
SOMMA(costo_totale_stanza) deve essere <= {budget}.
riepilogo_costi.totale = quella somma.
riepilogo_costi.budget_residuo = {budget} - totale.

REGOLA STILE "{style}":
Tutti gli arredi, colori, tessili devono essere coerenti con lo stile "{style}".
prompt_imagen: descrivi in inglese la stanza DOPO restyling con elementi
specifici e riconoscibili dello stile "{style}", mantenendo geometria originale.

Restituisci SOLO questo JSON (costi come interi):

{{
  "valutazione_generale": "analisi visiva dettagliata delle criticità attuali",
  "punti_di_forza": ["punto 1", "punto 2", "punto 3"],
  "criticita": ["criticità 1", "criticità 2"],
  "potenziale_str": "analisi potenziale per {dest_label} a {location}",
  "tariffe": {{
    "attuale_notte": "€XX-YY",
    "post_restyling_notte": "€XX-YY",
    "incremento_percentuale": "XX%"
  }},
  "stanze": [
    {{
      "nome": "Soggiorno",
      "indice_foto": 0,
      "stato_attuale": "descrizione stato attuale specifica",
      "interventi": [
        {{
          "titolo": "nome intervento breve",
          "dettaglio": "prodotti specifici, brand, prezzo unitario, tariffa {location}",
          "costo_min": 50,
          "costo_max": 120,
          "priorita": "alta",
          "dove_comprare": "negozio coerente con stile {style}"
        }}
      ],
      "costo_totale_stanza": 350,
      "prompt_imagen": "Photorealistic interior photo after home staging, {style} interior design style, [room name], PRESERVED: same walls, same windows, same ceiling, same floor material, CHANGED: [specific new furniture with exact colors/materials matching {style}], [specific textiles: rug pattern/color, curtain material, cushion colors all matching {style}], [specific decorative objects typical of {style}], warm natural light from existing windows, professional architectural photography, 35mm wide angle lens, shot from corner, magazine quality 4k"
    }}
  ],
  "riepilogo_costi": {{
    "manodopera_tinteggiatura": 0,
    "materiali_pittura": 0,
    "arredi_complementi": 0,
    "montaggio_varie": 0,
    "totale": 0,
    "budget_residuo": 0,
    "nota_budget": "commento su priorità e rispetto budget €{budget}"
  }},
  "piano_acquisti": [
    {{
      "categoria": "Tessili",
      "items": ["item specifico coerente con {style}"],
      "budget_stimato": 0,
      "negozi_consigliati": "negozi coerenti con {style}"
    }}
  ],
  "titolo_annuncio_suggerito": "Titolo Airbnb max 50 caratteri per {location}",
  "highlights_str": ["highlight 1", "highlight 2", "highlight 3"],
  "roi_restyling": "ROI: investimento €X, +€Y/notte, break-even Z notti (~N mesi al 60% occupazione)"
}}"""

    response = _genai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=image_parts + [types.Part.from_text(text=prompt)],
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
            max_output_tokens=4096,
        ),
    )

    text = response.text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    return json.loads(text.strip())


# ── Imagen 3 — staged photos (parallelo) ─────────────────────────────────────
# NOTA: Imagen 3 sarà deprecato il 24/06/2026.
# Migrare a gemini-2.5-flash-image quando stabile.

async def generate_staged_photos(photos: list, analysis: dict) -> list:
    stanze         = analysis.get("stanze", [])
    loop           = asyncio.get_running_loop()
    use_controlnet = bool(REPLICATE_TOKEN)
    tasks          = []

    for room in stanze:
        idx    = room.get("indice_foto", 0)
        prompt = room.get("prompt_imagen", "")

        if idx < len(photos) and prompt:
            photo_bytes = photos[idx]["content"]
            fn = _controlnet_depth_sync if use_controlnet else _imagen_edit_sync
            tasks.append(loop.run_in_executor(_imagen_executor, fn, photo_bytes, prompt))
        elif prompt:
            tasks.append(loop.run_in_executor(_imagen_executor, _imagen_generate_sync, prompt))
        else:
            async def _none():
                return None
            tasks.append(_none())

    results = await asyncio.gather(*tasks, return_exceptions=True)
    output  = []
    for r in results:
        if isinstance(r, Exception):
            print(f"[Imagen] task fallito: {r}")
            output.append(None)
        else:
            output.append(r)
    return output


# ── Path A — Imagen 3 inpainting ─────────────────────────────────────────────

def _imagen_edit_sync(photo_bytes: bytes, prompt: str) -> str | None:
    try:
        model        = ImageGenerationModel.from_pretrained("imagegeneration@006")
        source_image = VisionImage(image_bytes=photo_bytes)
        response = model.edit_image(
            base_image=source_image,
            prompt=prompt,
            edit_mode="inpainting-insert",
            mask_prompt=(
                "all movable furniture and soft furnishings: sofa, armchair, chairs, "
                "dining table, coffee table, side tables, curtains, blinds, rugs, carpets, "
                "floor lamps, table lamps, ceiling pendants, cushions, throws, pillows, "
                "wall art, paintings, posters, mirrors, decorative objects, vases, plants, "
                "books, shelves contents, bedding, duvet, bed linen, towels, bathroom accessories"
            ),
            negative_prompt=(
                "do not modify: walls, ceiling, floor tiles, parquet flooring, "
                "windows, window frames, doors, door frames, radiators, "
                "built-in wardrobes structure, kitchen cabinets structure, "
                "kitchen appliances, bathroom fixtures, bathtub, shower, toilet, sink, "
                "structural columns, room geometry, room proportions"
            ),
            number_of_images=1,
            guidance_scale=60,
            seed=42,
        )
        return base64.b64encode(response.images[0]._image_bytes).decode()
    except Exception as edit_err:
        print(f"[Imagen edit] fallback a generazione pura: {edit_err}")
        return _imagen_generate_sync(prompt)


# ── Path B — ControlNet Depth via Replicate (premium) ────────────────────────

def _controlnet_depth_sync(photo_bytes: bytes, prompt: str) -> str | None:
    try:
        import replicate
        import httpx
        img_b64 = base64.b64encode(photo_bytes).decode()
        output  = replicate.run(
            "lucataco/sdxl-controlnet-depth:latest",
            input={
                "image":                         f"data:image/jpeg;base64,{img_b64}",
                "prompt":                        prompt,
                "negative_prompt":               "deformed walls, distorted geometry, wrong perspective, blurry, low quality",
                "controlnet_conditioning_scale": 0.85,
                "num_inference_steps":           30,
                "guidance_scale":                7.5,
            },
        )
        url       = output[0] if isinstance(output, list) else output
        img_bytes = httpx.get(str(url), timeout=60).content
        return base64.b64encode(img_bytes).decode()
    except Exception as ctrl_err:
        print(f"[ControlNet] fallback a Imagen edit: {ctrl_err}")
        return _imagen_edit_sync(photo_bytes, prompt)


# ── Path C — Imagen 3 text-to-image (fallback) ───────────────────────────────

def _imagen_generate_sync(prompt: str) -> str | None:
    try:
        model    = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")
        response = model.generate_images(
            prompt=prompt,
            number_of_images=1,
            aspect_ratio="4:3",
        )
        return base64.b64encode(response.images[0]._image_bytes).decode()
    except Exception as gen_err:
        print(f"[Imagen generate] fallito: {gen_err}")
        return None
