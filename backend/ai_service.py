"""
AI Service
- analyze_with_gemini()      Gemini 1.5 Pro multimodal → JSON scheda completa
- validate_and_fix_costs()   Verifica e corregge i costi post-Gemini
- generate_staged_photos()   Imagen 3 inpainting (+ ControlNet depth fallback) in parallelo
- compress_image()           Riduce il peso delle foto prima di embedderle nel PDF
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
from vertexai.preview.vision_models import ImageGenerationModel, Image as VisionImage



try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("[WARNING] Pillow non installato — compressione immagini disabilitata")

PROJECT_ID      = os.environ["GCP_PROJECT_ID"]
LOCATION        = os.environ.get("GCP_LOCATION", "us-central1")
REPLICATE_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "")

_genai_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# Executor separati: Gemini è I/O bound con poca concorrenza per job,
# Imagen gira in parallelo per ogni stanza e beneficia di più thread.
_gemini_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="gemini")
_imagen_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="imagen")

# Cache in-memoria per evitare chiamate Gemini duplicate su foto identiche.
_analysis_cache: dict[str, dict] = {}


# ── Utilità: compressione foto ────────────────────────────────────────────────

def compress_image(img_bytes: bytes, max_width: int = 1400, quality: int = 82) -> bytes:
    """
    Ridimensiona e comprime un'immagine JPEG/PNG.
    Tipicamente porta una foto da 3-5 MB a 200-400 KB mantenendo qualità
    visiva sufficiente per la stampa PDF A4.
    """
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
    """
    Verifica tre invarianti matematiche e corregge eventuali errori di Gemini:

    1. costo_min <= costo_max per ogni intervento
    2. SOMMA(costo_totale_stanza) == riepilogo_costi.totale
    3. riepilogo_costi.totale <= budget

    Se una o più invarianti sono violate, scala proporzionalmente tutti i costi
    al 95% del budget e ricalcola i campi derivati.
    """
    stanze = analysis.get("stanze", [])
    rc     = analysis.setdefault("riepilogo_costi", {})

    # Invariante 1: min/max coerenti
    for room in stanze:
        for iv in room.get("interventi", []):
            cmin = iv.get("costo_min", 0)
            cmax = iv.get("costo_max", 0)
            if cmin > cmax:
                iv["costo_min"], iv["costo_max"] = cmax, cmin

    # Invariante 2 + 3
    totale_stanze     = sum(r.get("costo_totale_stanza", 0) for r in stanze)
    totale_dichiarato = rc.get("totale", 0)
    totale_reale      = max(totale_stanze, totale_dichiarato)

    needs_fix = (
        totale_reale > budget
        or abs(totale_stanze - totale_dichiarato) > 50   # discrepanza > €50
    )

    if needs_fix:
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
        nota_orig            = rc.get("nota_budget", "")
        rc["nota_budget"]    = (
            nota_orig +
            (". " if nota_orig else "") +
            f"Costi ricalibrati automaticamente per rispettare il budget di €{budget}."
        )

    return analysis


# ── Gemini 1.5 Pro ─────────────────────────────────────────────────────────────

async def analyze_with_gemini(photos: list, prefs: dict) -> dict:
    key = _cache_key(photos, prefs)
    if key in _analysis_cache:
        print("[Gemini] cache hit — analisi già eseguita per queste foto")
        return _analysis_cache[key]

    loop   = asyncio.get_running_loop()
    result = await loop.run_in_executor(_gemini_executor, _gemini_sync, photos, prefs)
    result = validate_and_fix_costs(result, prefs["budget"])
    _analysis_cache[key] = result
    return result


def _gemini_sync(photos: list, prefs: dict) -> dict:
    budget      = prefs["budget"]
    style       = prefs["style"]        # testo libero — qualsiasi stile digitato dall'utente
    location    = prefs["location"]
    destination = prefs["destination"]
    dest_label  = (
        "Airbnb / affitto breve (STR)"
        if destination == "STR"
        else "Casa vacanza"
    )

    # Percentuali di allocazione (no importi fissi hardcoded)
    alloc = {
        "arredi":        int(budget * 0.40),
        "tinteggiatura": int(budget * 0.30),
        "materiali":     int(budget * 0.20),
        "montaggio":     int(budget * 0.10),
    }

    model = GenerativeModel(
         "gemini-2.0-flash-001",
        system_instruction=(
            "Sei un esperto consulente di home staging e interior design italiano, "
            "specializzato in affitti brevi e case vacanza. "
            "Conosci in dettaglio i prezzi di mercato delle principali città italiane "
            "per: manodopera edile (tinteggiatori, imbianchini, tuttofare), "
            "materiali da costruzione (Leroy Merlin, Brico), "
            "arredi di fascia bassa/media/alta (IKEA, H&M Home, Zara Home, Maisons du Monde, "
            "mercatini dell'usato, Amazon, Etsy). "
            "I prezzi che fornisci devono riflettere il mercato reale della città indicata — "
            "Milano e Venezia hanno tariffe di manodopera sensibilmente più alte di Napoli o Palermo. "
            "Rispondi ESCLUSIVAMENTE con un oggetto JSON valido. "
            "Nessun testo prima o dopo il JSON. Nessun markdown. Nessun backtick."
        ),
    )

    parts = [
        Part.from_data(data=p["content"], mime_type=p["content_type"])
        for p in photos
    ]

    prompt = f"""Analizza queste {len(photos)} foto di un appartamento e produci
una scheda professionale di home staging per {dest_label}.

═══════════════════════════════
PARAMETRI PROGETTO
═══════════════════════════════
Budget totale:   €{budget}
Stile richiesto: {style}
Città:           {location}
Destinazione:    {dest_label}

═══════════════════════════════
REGOLE SUI PREZZI
═══════════════════════════════
• I prezzi DEVONO essere quelli reali di mercato a {location}.
  Esempi orientativi (adatta alla realtà locale):
  – Tinteggiatura manodopera: €6-12/mq secondo la città
  – Tuttofare: €18-35/ora secondo la città
  – Trasporto/montaggio IKEA: €30-80 forfait
• Per ogni intervento includi nel campo "dettaglio" la motivazione del prezzo
  (es. "tinteggiatura €9/mq — tariffa corrente artigiani {location}").
• Distribuzione budget CONSIGLIATA (modificala in base alle priorità visive):
    Arredi e complementi:          €{alloc['arredi']}  (~40%)
    Tinteggiatura (manodopera):    €{alloc['tinteggiatura']}  (~30%)
    Materiali pittura/accessori:   €{alloc['materiali']}  (~20%)
    Montaggio, trasporto, imprevisti: €{alloc['montaggio']}  (~10%)

═══════════════════════════════
REGOLA MATEMATICA — CRITICA
═══════════════════════════════
PRIMA di scrivere il JSON, calcola:
  TOTALE = SOMMA(costo_totale_stanza di ogni stanza nel campo "stanze")
  TOTALE deve essere ≤ {budget}
Se TOTALE > {budget}: riduci o elimina interventi a bassa priorità.
Nel JSON finale:
  riepilogo_costi.totale = TOTALE
  riepilogo_costi.budget_residuo = {budget} - TOTALE

═══════════════════════════════
REGOLA SULLO STILE "{style}"
═══════════════════════════════
• Ogni arredo, colore, tessile e oggetto decorativo consigliato deve essere
  COERENTE con lo stile "{style}" e con il budget disponibile.
• I negozi/brand suggeriti devono essere appropriati allo stile
  (es. boho → Etsy/mercatini/H&M Home; scandinavo → IKEA/Hay;
   lusso → Zara Home/Maisons du Monde; industrial → Amazon/mercatini).
• Il campo prompt_imagen deve descrivere in inglese la stanza DOPO il restyling,
  con elementi SPECIFICI e VISIVAMENTE RICONOSCIBILI dello stile "{style}":
  colori esatti, materiali, pattern, oggetti decorativi tipici.

═══════════════════════════════
JSON DA RESTITUIRE
═══════════════════════════════
{{
  "valutazione_generale": "analisi visiva dettagliata: stato pareti, pavimenti, illuminazione, arredi esistenti, ingombri, criticità estetiche specifiche",
  "punti_di_forza": ["punto 1 specifico", "punto 2", "punto 3"],
  "criticita": ["criticità visiva specifica 1", "criticità 2"],
  "potenziale_str": "analisi concreta: target ospiti ideali per {dest_label} a {location}, fascia di prezzo competitor, stagionalità, cosa manca attualmente",
  "tariffe": {{
    "attuale_notte": "€XX-YY",
    "post_restyling_notte": "€XX-YY",
    "incremento_percentuale": "XX%"
  }},
  "stanze": [
    {{
      "nome": "Soggiorno",
      "indice_foto": 0,
      "stato_attuale": "descrizione specifica: colori pareti, arredi presenti, problemi visibili",
      "interventi": [
        {{
          "titolo": "nome intervento breve",
          "dettaglio": "cosa fare esattamente: prodotto specifico, brand, dimensione/colore, prezzo unitario e tariffa manodopera {location}",
          "costo_min": 50,
          "costo_max": 120,
          "priorita": "alta",
          "dove_comprare": "negozio o piattaforma specifici, coerenti con stile {style}"
        }}
      ],
      "costo_totale_stanza": 350,
      "prompt_imagen": "Photorealistic interior design photo, {style} style, [room name], same room geometry preserved: same walls same windows same ceiling same floor material, [SPECIFIC furniture items with exact colors and materials matching {style} aesthetic], [SPECIFIC textiles: describe exact rug pattern/color, curtain material/color, cushion colors matching {style}], [SPECIFIC decorative objects typical of {style}], warm natural light from existing windows, professional architectural photography, 35mm wide angle lens, shot from corner, magazine quality 4k"
    }}
  ],
  "riepilogo_costi": {{
    "manodopera_tinteggiatura": 0,
    "materiali_pittura": 0,
    "arredi_complementi": 0,
    "montaggio_varie": 0,
    "totale": 0,
    "budget_residuo": 0,
    "nota_budget": "commento: cosa è stato prioritizzato e perché, come si è rispettato il budget di €{budget}"
  }},
  "piano_acquisti": [
    {{
      "categoria": "Tessili",
      "items": ["item specifico coerente con stile {style}", "altro item specifico"],
      "budget_stimato": 0,
      "negozi_consigliati": "negozi coerenti con stile {style} e prezzi {location}"
    }}
  ],
  "titolo_annuncio_suggerito": "Titolo Airbnb max 50 caratteri, evocativo e specifico per {location}",
  "highlights_str": ["highlight specifico 1", "highlight 2", "highlight 3"],
  "roi_restyling": "ROI concreto: investimento €X, tariffa da €A a €B/notte (+€C), break-even in Z notti (~N mesi con occupazione 60%)"
}}"""

    response = model.generate_content(
        [*parts, prompt],
        generation_config={"temperature": 0.2, "max_output_tokens": 4096},
    )

    text = response.text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$",       "", text)
    return json.loads(text.strip())


# ── Imagen 3 — staged photos in parallelo ─────────────────────────────────────

async def generate_staged_photos(photos: list, analysis: dict) -> list:
    """
    Per ogni stanza genera una foto staged.
    Path A (default):  Imagen 3 inpainting con mask_prompt + negative_prompt
    Path B (premium):  ControlNet Depth via Replicate (se REPLICATE_API_TOKEN impostato)
    Path C (fallback): Imagen 3 text-to-image puro
    """
    stanze         = analysis.get("stanze", [])
    loop           = asyncio.get_running_loop()
    use_controlnet = bool(REPLICATE_TOKEN)
    tasks          = []

    for room in stanze:
        idx    = room.get("indice_foto", 0)
        prompt = room.get("prompt_imagen", "")

        if idx < len(photos) and prompt:
            photo_bytes = photos[idx]["content"]
            if use_controlnet:
                tasks.append(
                    loop.run_in_executor(
                        _imagen_executor,
                        _controlnet_depth_sync,
                        photo_bytes, prompt,
                    )
                )
            else:
                tasks.append(
                    loop.run_in_executor(
                        _imagen_executor,
                        _imagen_edit_sync,
                        photo_bytes, prompt,
                    )
                )
        elif prompt:
            tasks.append(
                loop.run_in_executor(_imagen_executor, _imagen_generate_sync, prompt)
            )
        else:
            # Stanza senza foto né prompt: placeholder None
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
    """
    Imagen 3 inpainting: sostituisce mobili e complementi sulla foto originale
    preservando la geometria della stanza (pareti, pavimenti, finestre, volumi).
    mask_prompt    → cosa sostituire
    negative_prompt → cosa NON modificare mai
    """
    try:
        model        = ImageGenerationModel.from_pretrained("imagegeneration@006")
        source_image = VisionImage(image_bytes=photo_bytes)

        response = model.edit_image(
            base_image=source_image,
            prompt=prompt,
            edit_mode="inpainting-insert",
            mask_prompt=(
                "all movable furniture and soft furnishings: "
                "sofa, armchair, chairs, dining table, coffee table, side tables, "
                "curtains, blinds, rugs, carpets, floor lamps, table lamps, "
                "ceiling pendants, cushions, throws, pillows, "
                "wall art, paintings, posters, framed photos, mirrors, "
                "decorative objects, vases, plants, books, shelves contents, "
                "bedding, duvet, bed linen, towels, bathroom accessories"
            ),
            negative_prompt=(
                "do not modify: walls, ceiling, floor tiles, parquet flooring, "
                "windows, window frames, window glass, doors, door frames, "
                "radiators, built-in wardrobes structure, kitchen cabinets structure, "
                "kitchen appliances, bathroom fixtures, bathtub, shower, toilet, sink, "
                "structural columns, room geometry, room proportions"
            ),
            number_of_images=1,
            guidance_scale=60,   # 55-65: valori bassi = più fedele alla foto originale
            seed=42,
        )
        return base64.b64encode(response.images[0]._image_bytes).decode()

    except Exception as edit_err:
        print(f"[Imagen edit] fallback a generazione pura: {edit_err}")
        return _imagen_generate_sync(prompt)


# ── Path B — ControlNet Depth via Replicate (premium) ────────────────────────

def _controlnet_depth_sync(photo_bytes: bytes, prompt: str) -> str | None:
    """
    ControlNet con Depth map via Replicate.
    La depth map estrae la struttura 3D della stanza (profondità muri, volumi,
    prospettiva) e la impone alla generazione: è impossibile distorcere
    la planimetria o spostare finestre/pareti.
    Attivato automaticamente se REPLICATE_API_TOKEN è impostato.
    """
    try:
        import replicate
        import httpx

        img_b64 = base64.b64encode(photo_bytes).decode()
        output  = replicate.run(
            "lucataco/sdxl-controlnet-depth:latest",
            input={
                "image":                         f"data:image/jpeg;base64,{img_b64}",
                "prompt":                        prompt,
                "negative_prompt": (
                    "deformed walls, distorted geometry, wrong perspective, "
                    "blurry, low quality, watermark, text"
                ),
                "controlnet_conditioning_scale": 0.85,   # 0.8-0.9 = equilibrio fedeltà/creatività
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


# ── Path C — Imagen 3 text-to-image (ultimo fallback) ────────────────────────

def _imagen_generate_sync(prompt: str) -> str | None:
    """Genera una nuova foto da prompt puro, senza foto di riferimento."""
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
