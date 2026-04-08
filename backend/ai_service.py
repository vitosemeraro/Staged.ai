"""
AI Service v9
- Gemini 2.5 Flash via REST API per analisi JSON
- imagen-3.0-generate-002 via google-genai SDK per staged photos
  (generate_images con prompt dettagliato dalla descrizione Gemini)
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
print(f"[startup] GEMINI_API_KEY={'SET' if GEMINI_API_KEY else 'MISSING'}")

GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
)

_genai_client = None

def _get_genai_client():
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client(api_key=GEMINI_API_KEY)
    return _genai_client

_gemini_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="gemini")
_imagen_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="imagen")

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
        f"Deve descrivere la stanza DOPO il restyling in stile {style}.\n"
        f"Includi: tipo stanza, stile {style}, colori, materiali specifici degli arredi,\n"
        f"illuminazione, atmosfera. Max 60 parole. NON menzionare pareti o finestre.\n"
        f"Esempio: \"Photorealistic interior photo, Scandinavian living room after staging,\n"
        f"light oak furniture, white linen sofa, grey wool rug, pendant lamp,\n"
        f"green plants, warm natural light, minimalist decor, 4k quality\"\n\n"
        f"Restituisci SOLO questo JSON (costi come interi):\n\n"
        "{{\n"
        "  \"valutazione_generale\": \"analisi visiva dettagliata\",\n"
        "  \"punti_di_forza\": [\"punto 1\", \"punto 2\", \"punto 3\"],\n"
        "  \"criticita\": [\"critica 1\", \"critica 2\"],\n"
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
        "      \"stato_attuale\": \"descrizione stato attuale\",\n"
        "      \"interventi\": [\n"
        "        {{\n"
        "          \"titolo\": \"nome breve\",\n"
        f"          \"dettaglio\": \"prodotti, brand, prezzo, tariffa {location}\",\n"
        "          \"costo_min\": 50,\n"
        "          \"costo_max\": 120,\n"
        "          \"priorita\": \"alta\",\n"
        f"          \"dove_comprare\": \"negozio per {style}\"\n"
        "        }}\n"
        "      ],\n"
        "      \"costo_totale_stanza\": 350,\n"
        f"      \"prompt_imagen\": \"Photorealistic interior photo, {style} style room after home staging, [specific furniture and decor], warm natural light, 4k quality\"\n"
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
        f"      \"articoli\": [\"item per {style}\"],\n"
        "      \"budget_stimato\": 0,\n"
        f"      \"negozi_consigliati\": \"negozi per {style}\"\n"
        "    }}\n"
        "  ],\n"
        "  \"titolo_annuncio_suggerito\": \"Titolo Airbnb max 50 caratteri\",\n"
        "  \"highlights_str\": [\"highlight 1\", \"highlight 2\", \"highlight 3\"],\n"
        "  \"roi_restyling\": \"ROI: \u20acX investimento, +\u20acY/notte, break-even Z notti\"\n"
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


async def generate_staged_photos(photos: list, analysis: dict) -> list:
    stanze = analysis.get("stanze", [])
    loop   = asyncio.get_running_loop()
    tasks  = []

    for room in stanze:
        prompt = room.get("prompt_imagen", "")
        if prompt:
            tasks.append(loop.run_in_executor(
                _imagen_executor, _imagen_generate_sync, prompt
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


def _imagen_generate_sync(prompt: str) -> str | None:
    """
    Genera foto staged con imagen-3.0-generate-002 via google-genai SDK.
    Usa generate_images() — metodo nativo del nuovo SDK, diverso dal vecchio
    vertexai SDK che usava imagegeneration@006 (deprecato).
    """
    try:
        print(f"[Imagen] generazione con prompt: {prompt[:80]}")
        client = _get_genai_client()

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
            img = response.generated_images[0]
            print(f"[Imagen] SUCCESS")
            return base64.b64encode(img.image.image_bytes).decode()

        print("[Imagen] nessuna immagine generata")
        return None

    except Exception as e:
        import traceback
        print(f"[Imagen] ERRORE: {type(e).__name__}: {e}")
        print(traceback.format_exc())
        return None
