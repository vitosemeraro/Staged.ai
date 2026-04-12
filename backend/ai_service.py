"""
AI Service v13 — C4 + C5_SMART_FULL
Layout PDF per ogni stanza:
  - Foto originale
  - C4_FULL (guidance=25, trasformazione totale con fedeltà geometrica)
  - C5_SMART_FULL (guidance=28, decluttering + design attivo + lighting)
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
    from PIL import Image as PILImage, ImageOps
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
_imagen_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="imagen")

_analysis_cache: dict[str, dict] = {}


async def _noop():
    return None


def compress_image(img_bytes: bytes, max_width: int = 1200, quality: int = 85) -> bytes:
    if not HAS_PIL:
        return img_bytes
    try:
        img = PILImage.open(io.BytesIO(img_bytes))
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
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


# ── GEMINI ANALYSIS ───────────────────────────────────────────────────────────

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
        "Sei un Interior Designer senior italiano specializzato in home staging per affitti brevi. "
        "Pensi come un architetto: elimini il superfluo, sostituisci con arredi iconici e accessibili "
        "(IKEA, H&M Home, Zara Home), ottimizzi luce e fotogenia per Airbnb. "
        "Conosci i prezzi reali di mercato delle principali citta' italiane. "
        "Rispondi ESCLUSIVAMENTE con un oggetto JSON valido. "
        "Nessun testo prima o dopo. Nessun markdown. Nessun backtick."
    )

    prompt = (
        f"Analizza queste {len(photos)} foto e produci una scheda di home staging per {dest_label}.\n\n"
        f"PARAMETRI:\n"
        f"- Budget: \u20ac{budget} | Stile: {style} | Citta': {location} | Dest: {dest_label}\n\n"
        f"DISTRIBUZIONE BUDGET:\n"
        f"  Arredi \u20ac{alloc['arredi']} | Tinteggiatura \u20ac{alloc['tinteggiatura']} "
        f"| Materiali \u20ac{alloc['materiali']} | Montaggio \u20ac{alloc['montaggio']}\n\n"
        f"REGOLA MATEMATICA: SOMMA(costo_totale_stanza) <= {budget}.\n\n"
        f"REGOLA INTERVENTI — agisci da designer, non da consulente:\n"
        f"- Rimuovi ESPLICITAMENTE mobili brutti, ingombranti o inutili per affitti brevi.\n"
        f"- Sostituisci arredi datati con pezzi IKEA moderni se il budget lo permette.\n"
        f"- Specifica sempre: texture parete (es. 'matte plaster white'), "
        f"modello arredi (es. 'IKEA LISABO table'), tessili (colore+materiale).\n"
        f"- Rimuovi elementi disturbanti: sacchi spazzatura, cavi, oggetti buttati.\n\n"
        f"GENERA 2 varianti staged per ogni stanza:\n\n"
        f"VARIANTE C4_FULL (guidance=25) — trasformazione totale coordinata:\n"
        f"Prompt denso in inglese (max 70 parole). Include: new wall color+texture, "
        f"all new furniture, textiles, lighting, decor. Coerente con stile {style}.\n\n"
        f"VARIANTE D_FULL_SMART (guidance=35) — 'Vestire' e 'Illuminare' la stanza:\n"
        f"Agisci da fotografo di interni e stylist senior. Il prompt DEVE:\n"
        f"1. Citare esplicitamente colore e materiale delle pareti esistenti.\n"
        f"2. Sostituire mobili brutti/ingombranti con modelli IKEA stilosi e congrui.\n"
        f"3. Aggiungere LAYERING TESSILE ricco: tappeti, cuscini multipli, tende, plaid.\n"
        f"4. Aggiungere decor: quadri, piante, vasi, lampade, libri, oggetti decorativi.\n"
        f"5. Creare contrasti materici: legno caldo, tessuto naturale, metallo, ceramica.\n"
        f"6. Terminare SEMPRE con: 'professional real estate photography, 24mm wide angle lens, "
        f"cinematic warm lighting, balanced exposure, perfectly staged, highly detailed.'\n\n"
        f"Restituisci SOLO questo JSON:\n\n"
        "{{\n"
        "  \"valutazione_generale\": \"analisi visiva dettagliata\",\n"
        "  \"punti_di_forza\": [\"p1\", \"p2\"],\n"
        "  \"criticita\": [\"c1 — elemento specifico da rimuovere/cambiare\"],\n"
        f"  \"potenziale_str\": \"potenziale {dest_label} a {location}\",\n"
        "  \"tariffe\": {{\n"
        "    \"attuale_notte\": \"\u20acXX-YY\",\n"
        "    \"post_restyling_notte\": \"\u20acXX-YY\",\n"
        "    \"incremento_percentuale\": \"XX%\"\n"
        "  }},\n"
        "  \"stanze\": [\n"
        "    {{\n"
        "      \"nome\": \"Soggiorno\",\n"
        "      \"indice_foto\": 0,\n"
        "      \"stato_attuale\": \"descrizione dettagliata inclusi elementi da rimuovere\",\n"
        "      \"interventi\": [\n"
        "        {{\n"
        "          \"titolo\": \"nome intervento\",\n"
        f"          \"dettaglio\": \"prodotto specifico, brand, prezzo {location}\",\n"
        "          \"costo_min\": 50, \"costo_max\": 120,\n"
        "          \"priorita\": \"alta\",\n"
        f"          \"dove_comprare\": \"negozio coerente con {style}\"\n"
        "        }}\n"
        "      ],\n"
        "      \"costo_totale_stanza\": 350,\n"
        "      \"esperimenti_staged\": [\n"
        "        {{\n"
        "          \"logic_id\": \"C4_FULL\",\n"
        "          \"guidance_scale\": 25,\n"
        f"          \"prompt_en\": \"Full home staging transformation. [New wall color+texture]. [New furniture {style} style with exact names]. [Textiles: rug, curtains, cushions]. [Lighting]. Bright natural light. Professional interior photography. 4k.\",\n"
        "          \"interventi_lista\": [\n"
        "            {{\"voce\": \"Pittura pareti colore specifico\", \"costo\": 300}},\n"
        "            {{\"voce\": \"Arredo principale (es. divano IKEA KIVIK)\", \"costo\": 400}},\n"
        "            {{\"voce\": \"Tessili e decor coordinati\", \"costo\": 200}}\n"
        "          ],\n"
        "          \"costo_simulato\": 900\n"
        "        }},\n"
        "        {{\n"
        "          \"logic_id\": \"D_FULL_SMART\",\n"
        "          \"guidance_scale\": 35,\n"
        "          \"prompt_en\": \"The room features [existing wall color] walls and [floor material]. Remove old bulky furniture. Replace with [specific IKEA model] in [color]. Add rich textile layering: [rug material+color], [curtains], [cushions x3], [plaid]. Decor: [plants+vases], [art prints], [pendant lamp], [books]. Contrasting materials: warm oak wood, natural linen, matte ceramic. Professional real estate photography, 24mm wide angle lens, cinematic warm lighting, balanced exposure, perfectly staged, highly detailed.\",\n"
        "          \"interventi_lista\": [\n"
        "            {{\"voce\": \"Rimozione mobili ingombranti\", \"costo\": 80}},\n"
        "            {{\"voce\": \"Arredo sostitutivo IKEA-style\", \"costo\": 350}},\n"
        "            {{\"voce\": \"Layering tessile ricco\", \"costo\": 180}},\n"
        "            {{\"voce\": \"Decor, piante, lampade\", \"costo\": 120}}\n"
        "          ],\n"
        "          \"costo_simulato\": 730\n"
        "        }}\n"
        "      ]\n"
        "    }}\n"
        "  ],\n"
        "  \"riepilogo_costi\": {{\n"
        "    \"manodopera_tinteggiatura\": 0, \"materiali_pittura\": 0,\n"
        "    \"arredi_complementi\": 0, \"montaggio_varie\": 0,\n"
        "    \"totale\": 0, \"budget_residuo\": 0,\n"
        f"    \"nota_budget\": \"commento budget \u20ac{budget}\"\n"
        "  }},\n"
        "  \"piano_acquisti\": [\n"
        "    {{\n"
        "      \"categoria\": \"Arredi\",\n"
        f"      \"articoli\": [\"item specifico {style}\"],\n"
        "      \"budget_stimato\": 0,\n"
        f"      \"negozi_consigliati\": \"IKEA, H&M Home, {location}\"\n"
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
            "maxOutputTokens": 65536,
            "responseMimeType": "application/json"
        }
    }

    response = httpx.post(GEMINI_URL, json=payload, timeout=120.0,
                          headers={"Content-Type": "application/json"})
    response.raise_for_status()

    data = response.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    return _extract_json(text)


# ── STAGED PHOTOS — solo C4 e D_FULL_SMART ───────────────────────────────────

async def generate_staged_photos(photos: list, analysis: dict) -> list:
    stanze = analysis.get("stanze", [])
    loop   = asyncio.get_running_loop()
    all_futures = []

    for i, room in enumerate(stanze):
        idx         = room.get("indice_foto", 0)
        esperimenti = room.get("esperimenti_staged", [])
        photo_bytes = photos[idx]["content"] if idx < len(photos) else None

        for esp in esperimenti:
            logic_id = esp.get("logic_id", "")
            if logic_id not in ("C4_FULL", "D_FULL_SMART"):
                continue
            prompt   = esp.get("prompt_en", "")
            guidance = esp.get("guidance_scale", 25)
            if not prompt or not photo_bytes:
                continue

            all_futures.append((logic_id, i,
                loop.run_in_executor(
                    _imagen_executor, _approach_C_edit,
                    photo_bytes, prompt, guidance, logic_id
                )
            ))

    results = [{} for _ in stanze]
    gathered = await asyncio.gather(
        *[f for _, _, f in all_futures],
        return_exceptions=True
    )
    for (key, room_idx, _), result in zip(all_futures, gathered):
        if isinstance(result, Exception):
            print(f"[{key} stanza {room_idx}] ERRORE: {result}")
            results[room_idx][key] = None
        else:
            results[room_idx][key] = result

    return results


# ── Core edit function ────────────────────────────────────────────────────────

def _approach_C_edit(photo_bytes: bytes, prompt: str,
                     guidance_scale: int, label: str) -> str | None:
    try:
        compressed = compress_image(photo_bytes, max_width=1024, quality=80)
        print(f"[{label}] foto: {len(compressed)//1024}KB guidance={guidance_scale}")

        client = _get_vertex_client()

        negative_base = (
            "distorted architecture, blurry textures, changing window frame positions, "
            "moving doors, wrong room proportions, deformed walls, different ceiling height, "
            "watermark, low quality, unrealistic"
        )

        if label == "D_FULL_SMART":
            negative_prompt = (
                negative_base + ", "
                "clutter, messy cables, trash bags, dark shadows, bad exposure, "
                "overexposed, underexposed, harsh lighting, noise, grainy, empty walls, "
                "unwelcoming atmosphere, cheap furniture, bad framing, sparse decor"
            )
        else:
            # C4_FULL
            negative_prompt = negative_base

        response = client.models.edit_image(
            model="imagen-3.0-capability-001",
            prompt=prompt,
            reference_images=[
                genai_types.RawReferenceImage(
                    reference_id=1,
                    reference_image=genai_types.Image(image_bytes=compressed),
                )
            ],
            config=genai_types.EditImageConfig(
                edit_mode="EDIT_MODE_DEFAULT",
                number_of_images=1,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                safety_filter_level="block_only_high",
            ),
        )

        if response.generated_images:
            print(f"[{label}] SUCCESS")
            return base64.b64encode(
                response.generated_images[0].image.image_bytes
            ).decode()

        print(f"[{label}] nessuna immagine generata")
        return None

    except Exception as e:
        print(f"[{label}] ERRORE: {type(e).__name__}: {e}")
        return None