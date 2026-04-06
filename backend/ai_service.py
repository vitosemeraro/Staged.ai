"""
AI Service v6 — prompt_imagen generato in Python, non da Gemini
- Gemini fornisce solo "oggetti_da_sostituire" (3-5 parole)
- Il prompt Imagen è un template fisso nel codice — corto e ancorato alla foto
- guidance_scale=35 per massima fedeltà all'originale
- negative_prompt dinamico per stanza
"""
import asyncio
import base64
import hashlib
import io
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import httpx
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel, Image as VisionImage

try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("[WARNING] Pillow non installato")

PROJECT_ID      = os.environ["GCP_PROJECT_ID"]
LOCATION        = os.environ.get("GCP_LOCATION", "us-central1")
GEMINI_API_KEY  = os.environ["GEMINI_API_KEY"]
REPLICATE_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "")

GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
)

vertexai.init(project=PROJECT_ID, location=LOCATION)

_gemini_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="gemini")
_imagen_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="imagen")

_analysis_cache: dict[str, dict] = {}

# Mask prompt fisso — raccomandato da Gemini per massima fedeltà spaziale
MASK_PROMPT = (
    "furniture, chairs, table, sofa, bed, curtains, cabinets, "
    "refrigerator, oven, sink, lamps, rugs, decorative objects, "
    "wall art, kitchen appliances, towels, mirror, radiator"
)

# Template prompt Imagen — CORTO e fisso, solo gli oggetti variano
# NON menziona walls/windows/floor (li protegge la maschera)
# guidance_scale=35 garantisce fedeltà alla foto originale
def _build_imagen_prompt(style: str, oggetti: str) -> str:
    return f"Professional home staging, {style} style, {oggetti}. High resolution, realistic photography."


async def _noop():
    return None


# ── Compressione foto ─────────────────────────────────────────────────────────

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
        print(f"[compress_image] fallback: {e}")
        return img_bytes


def _cache_key(photos: list, prefs: dict) -> str:
    h = hashlib.md5()
    for p in photos:
        h.update(p["content"])
    h.update(prefs.get("style", "").encode())
    h.update(str(prefs.get("budget", 0)).encode())
    h.update(prefs.get("location", "").encode())
    return h.hexdigest()


def _extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```\s*$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start == -1:
        raise ValueError(f"Nessun JSON nella risposta: {text[:300]}")
    depth = 0
    end = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start=start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        candidate = text[start:]
        ob  = candidate.count("{") - candidate.count("}")
        ob2 = candidate.count("[") - candidate.count("]")
        candidate += ("]" * ob2) + ("}" * ob)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON troncato non riparabile: {e}")
    return json.loads(text[start:end])


# ── Validazione costi ─────────────────────────────────────────────────────────

def validate_and_fix_costs(analysis: dict, budget: int) -> dict:
    stanze = analysis.get("stanze", [])
    rc     = analysis.setdefault("riepilogo_costi", {})
    for room in stanze:
        for iv in room.get("interventi", []):
            if iv.get("costo_min", 0) > iv.get("costo_max", 0):
                iv["costo_min"], iv["costo_max"] = iv["costo_max"], iv["costo_min"]
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
            f"Costi ricalibrati per budget \u20ac{budget}."
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
        "Conosci i prezzi di mercato delle principali citt\u00e0 italiane per manodopera, "
        "materiali (Leroy Merlin, Brico) e arredi (IKEA, H&M Home, Zara Home, "
        "Maisons du Monde, mercatini, Amazon, Etsy). "
        "I prezzi devono riflettere il mercato reale della citt\u00e0 indicata. "
        "Rispondi ESCLUSIVAMENTE con un oggetto JSON valido. "
        "Nessun testo prima o dopo. Nessun markdown. Nessun backtick."
    )

    prompt = f"""Analizza queste {len(photos)} foto di un appartamento e produci
una scheda professionale di home staging per {dest_label}.

PARAMETRI:
- Budget totale:   \u20ac{budget}
- Stile richiesto: {style}
- Citt\u00e0:           {location}
- Destinazione:    {dest_label}

REGOLE PREZZI:
- Usa prezzi reali di {location}.
- Distribuzione consigliata:
    Arredi e complementi:        \u20ac{alloc['arredi']} (~40%)
    Tinteggiatura manodopera:    \u20ac{alloc['tinteggiatura']} (~30%)
    Materiali pittura/accessori: \u20ac{alloc['materiali']} (~20%)
    Montaggio e imprevisti:      \u20ac{alloc['montaggio']} (~10%)

REGOLA MATEMATICA CRITICA:
SOMMA(costo_totale_stanza) deve essere <= {budget}.
riepilogo_costi.totale = quella somma.
riepilogo_costi.budget_residuo = {budget} - totale.

REGOLA oggetti_da_sostituire (IMPORTANTE):
Elenca in inglese SOLO i 3-5 nuovi oggetti specifici da inserire nella stanza
in stile {style}, con materiale e colore. Max 20 parole. Solo oggetti, no descrizioni.
Esempio: "light oak dining table, linen white curtains, wool grey rug, pendant lamp"

REGOLA negative_prompt_imagen:
Elenca in inglese gli elementi fissi/strutturali NON coperti dagli interventi.
Includi SEMPRE: blurry, low quality, watermark, text, distorted perspective.
Aggiungi elementi non toccati dal budget, es:
- cucina senza interventi mobili: "kitchen cabinets, kitchen units, worktop"
- bagno senza interventi sanitari: "bathtub, toilet, sink, bathroom tiles"

Restituisci SOLO questo JSON (costi come interi):

{{
  "valutazione_generale": "analisi visiva dettagliata",
  "punti_di_forza": ["punto 1", "punto 2", "punto 3"],
  "criticita": ["critica 1", "critica 2"],
  "potenziale_str": "potenziale per {dest_label} a {location}",
  "tariffe": {{
    "attuale_notte": "\u20acXX-YY",
    "post_restyling_notte": "\u20acXX-YY",
    "incremento_percentuale": "XX%"
  }},
  "stanze": [
    {{
      "nome": "Soggiorno",
      "indice_foto": 0,
      "stato_attuale": "descrizione stato attuale",
      "interventi": [
        {{
          "titolo": "nome breve",
          "dettaglio": "prodotti, brand, prezzo, tariffa {location}",
          "costo_min": 50,
          "costo_max": 120,
          "priorita": "alta",
          "dove_comprare": "negozio per {style}"
        }}
      ],
      "costo_totale_stanza": 350,
      "oggetti_da_sostituire": "light oak sofa, linen curtains white, wool rug grey, pendant lamp white",
      "negative_prompt_imagen": "blurry, low quality, watermark, text, distorted perspective, kitchen cabinets, kitchen units"
    }}
  ],
  "riepilogo_costi": {{
    "manodopera_tinteggiatura": 0,
    "materiali_pittura": 0,
    "arredi_complementi": 0,
    "montaggio_varie": 0,
    "totale": 0,
    "budget_residuo": 0,
    "nota_budget": "commento budget \u20ac{budget}"
  }},
  "piano_acquisti": [
    {{
      "categoria": "Tessili",
      "articoli": ["item per {style}"],
      "budget_stimato": 0,
      "negozi_consigliati": "negozi per {style}"
    }}
  ],
  "titolo_annuncio_suggerito": "Titolo Airbnb max 50 caratteri",
  "highlights_str": ["highlight 1", "highlight 2", "highlight 3"],
  "roi_restyling": "ROI: \u20acX investimento, +\u20acY/notte, break-even Z notti"
}}"""

    parts = []
    for p in photos:
        parts.append({
            "inline_data": {
                "mime_type": p["content_type"],
                "data": base64.b64encode(p["content"]).decode()
            }
        })
    parts.append({"text": prompt})

    payload = {
        "system_instruction": {"parts": [{"text": system_instruction}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 16384,
            "responseMimeType": "application/json"
        }
    }

    response = httpx.post(
        GEMINI_URL,
        json=payload,
        timeout=120.0,
        headers={"Content-Type": "application/json"}
    )
    response.raise_for_status()

    data = response.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    result = json.loads(text) if text.startswith("{") else _extract_json(text)

    # Costruisci prompt_imagen da template fisso + oggetti_da_sostituire di Gemini
    style = prefs["style"]
    for room in result.get("stanze", []):
        oggetti = room.get("oggetti_da_sostituire", "minimalist furniture and decor")
        room["prompt_imagen"] = _build_imagen_prompt(style, oggetti)

    return result


# ── Imagen 3 — staged photos (parallelo) ──────────────────────────────────────

async def generate_staged_photos(photos: list, analysis: dict) -> list:
    stanze         = analysis.get("stanze", [])
    loop           = asyncio.get_running_loop()
    use_controlnet = bool(REPLICATE_TOKEN)
    tasks          = []

    BASE_NEGATIVE = (
        "structural changes, moving walls, changing windows, changing floor tiles, "
        "modifying ceiling, different room layout, distorted architecture, blurry, low quality"
    )

    for room in stanze:
        idx    = room.get("indice_foto", 0)
        prompt = room.get("prompt_imagen", "")
        room_negative = room.get("negative_prompt_imagen", "")
        full_negative = f"{BASE_NEGATIVE}, {room_negative}" if room_negative else BASE_NEGATIVE

        if idx < len(photos) and prompt:
            photo_bytes = photos[idx]["content"]
            fn = _controlnet_depth_sync if use_controlnet else _imagen_edit_sync
            tasks.append(loop.run_in_executor(
                _imagen_executor, fn, photo_bytes, prompt, full_negative
            ))
        elif prompt:
            tasks.append(loop.run_in_executor(
                _imagen_executor, _imagen_generate_sync, prompt, full_negative
            ))
        else:
            tasks.append(_noop())

    results = await asyncio.gather(*tasks, return_exceptions=True)
    output  = []
    for r in results:
        if isinstance(r, Exception):
            print(f"[Imagen] task fallito: {r}")
            output.append(None)
        else:
            output.append(r)
    return output


def _imagen_edit_sync(photo_bytes: bytes, prompt: str, negative_prompt: str) -> str | None:
    try:
        model        = ImageGenerationModel.from_pretrained("imagegeneration@006")
        source_image = VisionImage(image_bytes=photo_bytes)
        response = model.edit_image(
            base_image=source_image,
            prompt=prompt,
            edit_mode="inpainting-insert",
            mask_prompt="furniture, decor, rugs, clutter",  # chirurgico — non tocca struttura
            negative_prompt=negative_prompt,
            number_of_images=1,
            guidance_scale=12,  # 10-12 raccomandato da Gemini — fedeltà senza allucinazioni
            seed=42,
        )
        return base64.b64encode(response.images[0]._image_bytes).decode()
    except Exception as e:
        # NO fallback a generate_images — meglio None che una stanza inventata
        print(f"[Imagen edit] errore: {e}")
        return None


def _controlnet_depth_sync(photo_bytes: bytes, prompt: str, negative_prompt: str) -> str | None:
    try:
        import replicate
        img_b64 = base64.b64encode(photo_bytes).decode()
        output  = replicate.run(
            "lucataco/sdxl-controlnet-depth:latest",
            input={
                "image":                         f"data:image/jpeg;base64,{img_b64}",
                "prompt":                        prompt,
                "negative_prompt":               negative_prompt,
                "controlnet_conditioning_scale": 0.85,
                "num_inference_steps":           30,
                "guidance_scale":                7.5,
            },
        )
        url       = output[0] if isinstance(output, list) else output
        img_bytes = httpx.get(str(url), timeout=60).content
        return base64.b64encode(img_bytes).decode()
    except Exception as e:
        print(f"[ControlNet] fallback: {e}")
        return _imagen_edit_sync(photo_bytes, prompt, negative_prompt)


def _imagen_generate_sync(prompt: str, negative_prompt: str) -> str | None:
    try:
        model    = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")
        response = model.generate_images(prompt=prompt, number_of_images=1, aspect_ratio="4:3")
        return base64.b64encode(response.images[0]._image_bytes).decode()
    except Exception as e:
        print(f"[Imagen generate] fallito: {e}")
        return None
