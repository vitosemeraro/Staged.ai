"""
AI Service v11 — Multi-approach staging
Genera 4 varianti per ogni stanza per confronto:
  A: generate_images prompt base (attuale, funziona)
  B: generate_images prompt geometrico pesante
  C: generate_images con reference_image (foto originale come ancora)
  D: edit_image con imagen-3.0-capability-001 (inpainting nativo)
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
from google import genai
from google.genai import types as genai_types

try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("[WARNING] Pillow non installato")

PROJECT_ID     = os.environ.get("GCP_PROJECT_ID", "")
LOCATION       = os.environ.get("GCP_LOCATION", "us-central1")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

print(f"[startup] PROJECT_ID={PROJECT_ID!r}")
print(f"[startup] LOCATION={LOCATION!r}")
print(f"[startup] GEMINI_API_KEY={'SET' if GEMINI_API_KEY else 'MISSING'}")

GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
)

_vertex_client = None

def _get_vertex_client():
    global _vertex_client
    if _vertex_client is None:
        print(f"[Imagen] init Vertex AI client project={PROJECT_ID} location={LOCATION}")
        _vertex_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    return _vertex_client

_gemini_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="gemini")
_imagen_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="imagen")

_analysis_cache: dict[str, dict] = {}


async def _noop():
    return None


def compress_image(img_bytes: bytes, max_width: int = 1200, quality: int = 85) -> bytes:
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
        raise ValueError(f"Nessun JSON: {text[:300]}")
    depth = 0; end = -1; in_string = False; escape = False
    for i, ch in enumerate(text[start:], start=start):
        if escape: escape = False; continue
        if ch == "\\" and in_string: escape = True; continue
        if ch == '"' and not escape: in_string = not in_string; continue
        if in_string: continue
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0: end = i + 1; break
    if end == -1:
        candidate = text[start:]
        ob = candidate.count("{") - candidate.count("}")
        ob2 = candidate.count("[") - candidate.count("]")
        candidate += ("]" * ob2) + ("}" * ob)
        try: return json.loads(candidate)
        except json.JSONDecodeError as e: raise ValueError(f"JSON troncato: {e}")
    return json.loads(text[start:end])


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

    prompt = (
        f"Analizza queste {len(photos)} foto di un appartamento e produci\n"
        f"una scheda professionale di home staging per {dest_label}.\n\n"
        f"PARAMETRI:\n"
        f"- Budget totale:   \u20ac{budget}\n"
        f"- Stile richiesto: {style}\n"
        f"- Citt\u00e0:           {location}\n"
        f"- Destinazione:    {dest_label}\n\n"
        f"REGOLE PREZZI:\n"
        f"- Usa prezzi reali di {location}.\n"
        f"- Distribuzione consigliata:\n"
        f"    Arredi e complementi:        \u20ac{alloc['arredi']} (~40%)\n"
        f"    Tinteggiatura manodopera:    \u20ac{alloc['tinteggiatura']} (~30%)\n"
        f"    Materiali pittura/accessori: \u20ac{alloc['materiali']} (~20%)\n"
        f"    Montaggio e imprevisti:      \u20ac{alloc['montaggio']} (~10%)\n\n"
        f"REGOLA MATEMATICA CRITICA:\n"
        f"SOMMA(costo_totale_stanza) deve essere <= {budget}.\n"
        f"riepilogo_costi.totale = quella somma.\n"
        f"riepilogo_costi.budget_residuo = {budget} - totale.\n\n"
        f"REGOLA prompt_imagen:\n"
        f"Scrivi un prompt fotografico professionale in inglese per Imagen 3.\n"
        f"Descrivi la stanza DOPO il restyling in stile {style}.\n"
        f"Includi: tipo stanza, materiali e colori specifici degli arredi nello stile {style},\n"
        f"illuminazione naturale, atmosfera accogliente. Max 50 parole.\n"
        f"Esempio per stile Scandinavo soggiorno:\n"
        f"\"Photorealistic interior, Scandinavian living room, light oak coffee table,\n"
        f"grey linen sofa, wool rug, pendant lamp, potted plants, warm natural light, 4k\"\n\n"
        f"Restituisci SOLO questo JSON (costi come interi):\n\n"
        "{{\n"
        "  \"valutazione_generale\": \"analisi visiva\",\n"
        "  \"punti_di_forza\": [\"p1\", \"p2\", \"p3\"],\n"
        "  \"criticita\": [\"c1\", \"c2\"],\n"
        f"  \"potenziale_str\": \"potenziale per {dest_label} a {location}\",\n"
        "  \"tariffe\": {{\n"
        "    \"attuale_notte\": \"\u20acXX-YY\",\n"
        "    \"post_restyling_notte\": \"\u20acXX-YY\",\n"
        "    \"incremento_percentuale\": \"XX%\"\n"
        "  }},\n"
        "  \"stanze\": [\n"
        "    {{\n"
        "      \"nome\": \"Soggiorno\",\n"
        "      \"indice_foto\": 0,\n"
        "      \"stato_attuale\": \"descrizione\",\n"
        "      \"interventi\": [\n"
        "        {{\n"
        "          \"titolo\": \"nome breve\",\n"
        f"          \"dettaglio\": \"prodotti, brand, prezzo {location}\",\n"
        "          \"costo_min\": 50,\n"
        "          \"costo_max\": 120,\n"
        "          \"priorita\": \"alta\",\n"
        f"          \"dove_comprare\": \"negozio {style}\"\n"
        "        }}\n"
        "      ],\n"
        "      \"costo_totale_stanza\": 350,\n"
        f"      \"prompt_imagen\": \"Photorealistic interior, {style} style room, [specific furniture and colors], warm natural light, 4k\"\n"
        "    }}\n"
        "  ],\n"
        "  \"riepilogo_costi\": {{\n"
        "    \"manodopera_tinteggiatura\": 0,\n"
        "    \"materiali_pittura\": 0,\n"
        "    \"arredi_complementi\": 0,\n"
        "    \"montaggio_varie\": 0,\n"
        "    \"totale\": 0,\n"
        "    \"budget_residuo\": 0,\n"
        f"    \"nota_budget\": \"commento budget \u20ac{budget}\"\n"
        "  }},\n"
        "  \"piano_acquisti\": [\n"
        "    {{\n"
        "      \"categoria\": \"Tessili\",\n"
        f"      \"articoli\": [\"item {style}\"],\n"
        "      \"budget_stimato\": 0,\n"
        f"      \"negozi_consigliati\": \"negozi {style}\"\n"
        "    }}\n"
        "  ],\n"
        "  \"titolo_annuncio_suggerito\": \"Titolo Airbnb max 50 car\",\n"
        "  \"highlights_str\": [\"h1\", \"h2\", \"h3\"],\n"
        "  \"roi_restyling\": \"ROI: \u20acX, +\u20acY/notte, break-even Z notti\"\n"
        "}}"
    )

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

    response = httpx.post(GEMINI_URL, json=payload, timeout=120.0,
                          headers={"Content-Type": "application/json"})
    response.raise_for_status()

    data = response.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    return json.loads(text) if text.startswith("{") else _extract_json(text)


# ── MULTI-APPROACH STAGED PHOTOS ──────────────────────────────────────────────
#
# generate_staged_photos ora restituisce una lista di dizionari:
# [
#   {
#     "A_base":      "<base64>",   # generate_images prompt semplice
#     "B_geometric": "<base64>",   # generate_images prompt geometrico
#     "C_reference": "<base64>",   # generate_images con foto come reference
#     "D_edit":      "<base64>",   # edit_image inpainting nativo
#   },
#   ...  (un dict per ogni stanza)
# ]
# Il PDF mostrerà tutte e 4 le varianti per ogni stanza.

APPROACH_LABELS = {
    "A_base":      "A — Generate base",
    "B_geometric": "B — Generate geometric",
    "C_reference": "C — Generate + reference",
    "D_edit":      "D — Edit inpainting",
}


async def generate_staged_photos(photos: list, analysis: dict) -> list:
    stanze = analysis.get("stanze", [])
    loop   = asyncio.get_running_loop()
    # Ogni elemento della lista risultante è un dict con le 4 varianti
    room_tasks = []

    for room in stanze:
        idx    = room.get("indice_foto", 0)
        prompt = room.get("prompt_imagen", "")
        photo_bytes = photos[idx]["content"] if idx < len(photos) else None

        if not prompt:
            room_tasks.append(None)
            continue

        room_tasks.append((photo_bytes, prompt))

    # Lancia tutti gli approcci per tutte le stanze in parallelo
    all_futures = []
    room_indices = []

    for i, task in enumerate(room_tasks):
        if task is None:
            continue
        photo_bytes, prompt = task
        room_indices.append(i)

        # A: generate_images con prompt base (funziona, è la baseline)
        all_futures.append(("A_base", i,
            loop.run_in_executor(_imagen_executor, _approach_A_base, prompt)
        ))
        # B: generate_images con prompt geometrico pesante
        all_futures.append(("B_geometric", i,
            loop.run_in_executor(_imagen_executor, _approach_B_geometric, prompt)
        ))
        # C: generate_images con reference_image (foto originale come ancora visiva)
        all_futures.append(("C_reference", i,
            loop.run_in_executor(_imagen_executor, _approach_C_reference,
                                 photo_bytes, prompt)
        ))
        # D: edit_image inpainting nativo con imagen-3.0-capability-001
        all_futures.append(("D_edit", i,
            loop.run_in_executor(_imagen_executor, _approach_D_edit,
                                 photo_bytes, prompt)
        ))

    # Raccoglie risultati
    results = [{} for _ in stanze]

    gathered = await asyncio.gather(
        *[f for _, _, f in all_futures],
        return_exceptions=True
    )

    for (approach, room_idx, _), result in zip(all_futures, gathered):
        if isinstance(result, Exception):
            print(f"[Approccio {approach} stanza {room_idx}] ERRORE: {result}")
            results[room_idx][approach] = None
        else:
            results[room_idx][approach] = result

    return results


# ── Approccio A: generate_images prompt base ─────────────────────────────────

def _approach_A_base(prompt: str) -> str | None:
    try:
        print(f"[A_base] prompt: {prompt[:60]}")
        client = _get_vertex_client()
        response = client.models.generate_images(
            model="imagen-3.0-generate-002",
            prompt=prompt,
            config=genai_types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="4:3",
                safety_filter_level="block_only_high",
            ),
        )
        if response.generated_images:
            print("[A_base] SUCCESS")
            return base64.b64encode(
                response.generated_images[0].image.image_bytes
            ).decode()
        return None
    except Exception as e:
        print(f"[A_base] ERRORE: {type(e).__name__}: {e}")
        return None


# ── Approccio B: generate_images prompt geometrico pesante ───────────────────

def _approach_B_geometric(prompt: str) -> str | None:
    try:
        geo_prompt = (
            "Maintain exactly the same room layout, window positions, wall colors "
            "and floor material as the reference. Only replace movable furniture and decor. "
            + prompt
            + " Negative: moving walls, changing window shapes, different floor, "
            "distorted architecture, different room proportions."
        )
        print(f"[B_geometric] prompt: {geo_prompt[:60]}")
        client = _get_vertex_client()
        response = client.models.generate_images(
            model="imagen-3.0-generate-002",
            prompt=geo_prompt,
            config=genai_types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="4:3",
                safety_filter_level="block_only_high",
            ),
        )
        if response.generated_images:
            print("[B_geometric] SUCCESS")
            return base64.b64encode(
                response.generated_images[0].image.image_bytes
            ).decode()
        return None
    except Exception as e:
        print(f"[B_geometric] ERRORE: {type(e).__name__}: {e}")
        return None


# ── Approccio C: generate_images con reference_image ────────────────────────

def _approach_C_reference(photo_bytes: bytes | None, prompt: str) -> str | None:
    if not photo_bytes:
        print("[C_reference] nessuna foto originale, skip")
        return None
    try:
        compressed = compress_image(photo_bytes, max_width=1024, quality=80)
        print(f"[C_reference] foto: {len(compressed)//1024}KB, prompt: {prompt[:60]}")
        client = _get_vertex_client()

        # RawReferenceImage ancora l'output visivamente alla foto originale
        raw_ref = genai_types.RawReferenceImage(
            reference_id=1,
            reference_image=genai_types.Image(image_bytes=compressed),
        )
        response = client.models.generate_images(
            model="imagen-3.0-generate-002",
            prompt=prompt,
            config=genai_types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="4:3",
                safety_filter_level="block_only_high",
                reference_images=[raw_ref],
            ),
        )
        if response.generated_images:
            print("[C_reference] SUCCESS")
            return base64.b64encode(
                response.generated_images[0].image.image_bytes
            ).decode()
        return None
    except Exception as e:
        print(f"[C_reference] ERRORE: {type(e).__name__}: {e}")
        return None


# ── Approccio D: edit_image inpainting nativo ────────────────────────────────

def _approach_D_edit(photo_bytes: bytes | None, prompt: str) -> str | None:
    if not photo_bytes:
        print("[D_edit] nessuna foto originale, skip")
        return None
    try:
        compressed = compress_image(photo_bytes, max_width=1024, quality=80)
        print(f"[D_edit] foto: {len(compressed)//1024}KB, prompt: {prompt[:60]}")
        client = _get_vertex_client()

        edit_prompt = (
            "Home staging refurbishment of this existing room. "
            + prompt
            + " Keep the exact same walls, floor, windows and ceiling. "
            "Replace only furniture, curtains, rugs and decorative objects."
        )

        response = client.models.edit_image(
            model="imagen-3.0-capability-001",
            prompt=edit_prompt,
            reference_images=[
                genai_types.RawReferenceImage(
                    reference_id=1,
                    reference_image=genai_types.Image(image_bytes=compressed),
                )
            ],
            config=genai_types.EditImageConfig(
                edit_mode=genai_types.EditMode.INPAINTING_INSERT,
                number_of_images=1,
                safety_filter_level="block_only_high",
            ),
        )
        if response.generated_images:
            print("[D_edit] SUCCESS")
            return base64.b64encode(
                response.generated_images[0].image.image_bytes
            ).decode()
        print("[D_edit] nessuna immagine generata")
        return None
    except Exception as e:
        print(f"[D_edit] ERRORE: {type(e).__name__}: {e}")
        return None
