"""
AI Service v14 — C4 + C5_SMART_FULL
Fix:
  - Multi-foto: genera ESATTAMENTE N stanze, una per foto (con etichette sulle immagini)
  - Inventario visivo obbligatorio (detected_elements) — no elementi allucinati
  - Replacement logic per Imagen: "In place of [old], a [new IKEA] stands in the same spot"
  - C5 guidance hardcodata >= 32 — pareti come elemento dominante
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


def _validate_stanze_count(analysis: dict, n_photos: int) -> dict:
    """
    Garantisce che ci siano esattamente n_photos stanze e che indice_foto
    sia corretto (0..n_photos-1). Se Gemini ne ha restituite di meno, logga
    un warning ma non blocca — il PDF mostrerà ciò che c'è.
    """
    stanze = analysis.get("stanze", [])
    actual = len(stanze)
    if actual != n_photos:
        print(f"[WARNING] Gemini ha restituito {actual} stanze su {n_photos} foto attese.")
    # Correzione di sicurezza: se indice_foto manca o è fuori range, usa l'indice posizionale
    for i, room in enumerate(stanze):
        idx = room.get("indice_foto")
        if idx is None or not isinstance(idx, int) or idx >= n_photos:
            print(f"[WARNING] stanza {i} ha indice_foto={idx!r}, corretto a {i}")
            room["indice_foto"] = i
    return analysis


# ── GEMINI ANALYSIS ───────────────────────────────────────────────────────────

async def analyze_with_gemini(photos: list, prefs: dict) -> dict:
    key = _cache_key(photos, prefs)
    if key in _analysis_cache:
        print("[Gemini] cache hit")
        return _analysis_cache[key]
    loop   = asyncio.get_running_loop()
    result = await loop.run_in_executor(_gemini_executor, _gemini_sync, photos, prefs)
    result = _validate_stanze_count(result, len(photos))
    result = validate_and_fix_costs(result, prefs["budget"])
    _analysis_cache[key] = result
    return result


def _gemini_sync(photos: list, prefs: dict) -> dict:
    budget      = prefs["budget"]
    style       = prefs["style"]
    location    = prefs["location"]
    destination = prefs["destination"]
    dest_label  = "Airbnb / affitto breve (STR)" if destination == "STR" else "Casa vacanza"
    n           = len(photos)

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
        "REGOLA ASSOLUTA — INVENTARIO VISIVO: Prima di scrivere qualsiasi prompt_en, "
        "devi fare un inventario letterale di cio' che vedi nella foto. "
        "Non inventare elementi non visibili: se non vedi finestre, NON mettere tende. "
        "Se non vedi un tavolo, NON rimuovere un tavolo. "
        "Il campo detected_elements deve elencare SOLO cio' che e' fisicamente visibile nella foto. "
        "I prompt_en devono essere costruiti ESCLUSIVAMENTE a partire da detected_elements. "
        "Rispondi ESCLUSIVAMENTE con un oggetto JSON valido. "
        "Nessun testo prima o dopo. Nessun markdown. Nessun backtick."
    )

    # Costruzione della lista foto con indici espliciti nel testo
    # così Gemini può abbinare ogni stanza alla foto giusta
    foto_index_list = "\n".join(
        f"  - FOTO {i}: indice_foto={i} — analizza questa foto e crea la stanza {i+1}"
        for i in range(n)
    )

    prompt = (
        f"Analizza queste {n} foto e produci una scheda di home staging per {dest_label}.\n\n"
        f"PARAMETRI:\n"
        f"- Budget: \u20ac{budget} | Stile: {style} | Citta': {location} | Dest: {dest_label}\n\n"
        f"DISTRIBUZIONE BUDGET:\n"
        f"  Arredi \u20ac{alloc['arredi']} | Tinteggiatura \u20ac{alloc['tinteggiatura']} "
        f"| Materiali \u20ac{alloc['materiali']} | Montaggio \u20ac{alloc['montaggio']}\n\n"
        f"REGOLA MATEMATICA: SOMMA(costo_totale_stanza) <= {budget}.\n\n"
        f"═══════════════════════════════════════════════════\n"
        f"REGOLA MULTI-FOTO — CRITICA:\n"
        f"Hai ricevuto {n} foto, etichettate nell'ordine:\n"
        f"{foto_index_list}\n"
        f"Devi generare ESATTAMENTE {n} oggetti nell'array 'stanze', uno per foto.\n"
        f"indice_foto DEVE essere: 0 per la prima foto, 1 per la seconda, ... {n-1} per l'ultima.\n"
        f"Non raggruppare foto diverse in una sola stanza. Non saltare foto.\n"
        f"═══════════════════════════════════════════════════\n\n"
        f"REGOLA INTERVENTI — agisci da designer, non da consulente:\n"
        f"- Rimuovi ESPLICITAMENTE mobili brutti, ingombranti o inutili per affitti brevi.\n"
        f"- Sostituisci arredi datati con pezzi IKEA moderni se il budget lo permette.\n"
        f"- Specifica sempre: texture parete, modello arredi (es. 'IKEA LISABO'), tessili.\n"
        f"- Rimuovi elementi disturbanti: sacchi spazzatura, cavi, oggetti abbandonati.\n\n"
        f"STEP 1 — INVENTARIO VISIVO (per ogni foto, obbligatorio):\n"
        f"Compila detected_elements elencando SOLO cio' che vedi fisicamente in quella foto:\n"
        f"es. ['red sofa', 'wooden floor', 'yellow walls', 'no windows visible', 'overhead lamp']\n"
        f"CRITICO: se non vedi finestre scrivi 'no windows visible'. "
        f"Se non vedi tende, NON inserire 'curtains' o 'drapes' nei prompt.\n\n"
        f"STEP 2 — GENERA 2 varianti staged per ogni stanza BASATE SU detected_elements:\n\n"
        f"VARIANTE C4_FULL (guidance=25) — trasformazione totale:\n"
        f"Prompt in inglese (max 70 parole). "
        f"Inizia con: 'The room features [New Color] walls covering all surfaces from floor to ceiling.' "
        f"Poi usa replacement logic: 'In place of the [old item], a [new IKEA model] stands in the same position.' "
        f"Include nuovi arredi, tessili, luci coerenti con stile {style}. "
        f"NON inventare elementi non in detected_elements.\n\n"
        f"VARIANTE C5_SMART_FULL (guidance=32) — decluttering + sostituzione precisa:\n"
        f"REGOLE FERREE:\n"
        f"1. INIZIA SEMPRE con: 'The room features freshly painted matte white walls "
        f"covering all surfaces from floor to ceiling, replacing all previous colors and textures.'\n"
        f"2. Per OGNI mobile in detected_elements usa: "
        f"'In place of the [colore+tipo], a [modello IKEA esatto] stands in the same spot.'\n"
        f"3. VIETATO 'curtains', 'drapes', 'blinds' se detected_elements ha 'no windows visible'.\n"
        f"4. Includi SOLO sostituzioni dirette di elementi realmente visibili.\n"
        f"5. Chiudi con: 'Professional real estate photography, soft natural light, "
        f"bright and airy atmosphere, wide angle from corner, 8k resolution.'\n\n"
        f"Restituisci SOLO questo JSON (con {n} oggetti nell'array 'stanze'):\n\n"
        "{{\n"
        "  \"valutazione_generale\": \"analisi visiva complessiva di tutte le stanze\",\n"
        "  \"punti_di_forza\": [\"p1\", \"p2\"],\n"
        "  \"criticita\": [\"c1 — elemento specifico da rimuovere/cambiare\"],\n"
        f"  \"potenziale_str\": \"potenziale {dest_label} a {location}\",\n"
        "  \"tariffe\": {{\n"
        "    \"attuale_notte\": \"\u20acXX-YY\",\n"
        "    \"post_restyling_notte\": \"\u20acXX-YY\",\n"
        "    \"incremento_percentuale\": \"XX%\"\n"
        "  }},\n"
        f"  \"stanze\": [\n"
        f"    /* RIPETI QUESTO BLOCCO {n} VOLTE, una per ogni foto (indice_foto: 0..{n-1}) */\n"
        "    {{\n"
        "      \"nome\": \"Nome stanza identificata nella foto (es. Soggiorno, Camera, Cucina)\",\n"
        "      \"indice_foto\": 0,\n"
        "      \"detected_elements\": [\"elemento1 visibile\", \"elemento2 visibile\", \"no windows visible se assenti\"],\n"
        "      \"stato_attuale\": \"descrizione dettagliata inclusi elementi da rimuovere\",\n"
        "      \"interventi\": [\n"
        "        {{\n"
        "          \"titolo\": \"nome intervento\",\n"
        f"          \"dettaglio\": \"prodotto specifico, brand, prezzo a {location}\",\n"
        "          \"costo_min\": 50, \"costo_max\": 120,\n"
        "          \"priorita\": \"alta\",\n"
        f"          \"dove_comprare\": \"negozio coerente con stile {style}\"\n"
        "        }}\n"
        "      ],\n"
        "      \"costo_totale_stanza\": 350,\n"
        "      \"esperimenti_staged\": [\n"
        "        {{\n"
        "          \"logic_id\": \"C4_FULL\",\n"
        "          \"guidance_scale\": 25,\n"
        f"          \"prompt_en\": \"The room features [New Color] walls covering all surfaces from floor to ceiling. In place of the [old item from detected_elements], a [IKEA model] stands in the same position. [Other replacements]. {style} style. Professional interior photography. 4k.\",\n"
        "          \"interventi_lista\": [\n"
        "            {{\"voce\": \"Pittura pareti — [colore specifico]\", \"costo\": 300}},\n"
        "            {{\"voce\": \"Sostituzione [mobile] — IKEA [modello]\", \"costo\": 400}},\n"
        "            {{\"voce\": \"Tessili e decor coordinati\", \"costo\": 200}}\n"
        "          ],\n"
        "          \"costo_simulato\": 900\n"
        "        }},\n"
        "        {{\n"
        "          \"logic_id\": \"C5_SMART_FULL\",\n"
        "          \"guidance_scale\": 32,\n"
        "          \"prompt_en\": \"The room features freshly painted matte white walls covering all surfaces from floor to ceiling, replacing all previous colors and textures. In place of the [colore+tipo da detected_elements], an IKEA [modello esatto] stands in the same spot. [Altri replacement da detected_elements]. Professional real estate photography, soft natural light, bright and airy atmosphere, wide angle from corner, 8k resolution.\",\n"
        "          \"interventi_lista\": [\n"
        "            {{\"voce\": \"Pittura pareti bianco opaco (elemento dominante)\", \"costo\": 350}},\n"
        "            {{\"voce\": \"Sostituzione [mobile 1] — IKEA [modello]\", \"costo\": 450}},\n"
        "            {{\"voce\": \"Sostituzione [mobile 2] — IKEA [modello]\", \"costo\": 80}},\n"
        "            {{\"voce\": \"Tappeto e cuscini coordinati\", \"costo\": 100}}\n"
        "          ],\n"
        "          \"costo_simulato\": 980\n"
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
        f"      \"articoli\": [\"item specifico stile {style}\"],\n"
        "      \"budget_stimato\": 0,\n"
        f"      \"negozi_consigliati\": \"IKEA, H&M Home, Zara Home — {location}\"\n"
        "    }}\n"
        "  ],\n"
        "  \"titolo_annuncio_suggerito\": \"Titolo Airbnb max 50 caratteri\",\n"
        "  \"highlights_str\": [\"h1\", \"h2\", \"h3\"],\n"
        "  \"roi_restyling\": \"ROI: \u20acX, +\u20acY/notte, break-even Z notti\"\n"
        "}}"
    )

    # ── Costruzione parts: ogni foto è preceduta da un'etichetta testuale ──
    # Questo permette a Gemini di abbinare esattamente ogni immagine al suo indice_foto
    parts = []
    for i, p in enumerate(photos):
        parts.append({
            "text": f"[FOTO {i} — indice_foto: {i} — analizza questa immagine per la stanza {i+1}]"
        })
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

    response = httpx.post(GEMINI_URL, json=payload, timeout=180.0,
                          headers={"Content-Type": "application/json"})
    response.raise_for_status()

    data = response.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    result = _extract_json(text)
    print(f"[Gemini] stanze restituite: {len(result.get('stanze', []))} / {n} foto")
    return result


# ── STAGED PHOTOS — C4 e C5 ──────────────────────────────────────────────────
# Output per ogni stanza: {"C4_FULL": "<b64>", "C5_SMART_FULL": "<b64>"}

async def generate_staged_photos(photos: list, analysis: dict) -> list:
    stanze = analysis.get("stanze", [])
    loop   = asyncio.get_running_loop()
    all_futures = []

    for i, room in enumerate(stanze):
        idx         = room.get("indice_foto", i)
        esperimenti = room.get("esperimenti_staged", [])
        photo_bytes = photos[idx]["content"] if idx < len(photos) else None

        for esp in esperimenti:
            logic_id = esp.get("logic_id", "")
            if logic_id not in ("C4_FULL", "C5_SMART_FULL"):
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
    """
    edit_image EDIT_MODE_DEFAULT + RawReferenceImage.
    C4: guidance=25, replacement logic, pareti come elemento dominante
    C5: guidance >= 32 (hardcodato), matte white walls + replacement logic precisa per mobile
    """
    try:
        # C5: forza guidance minimo 32 — il testo del prompt deve dominare sull'immagine
        if label == "C5_SMART_FULL":
            guidance_scale = max(guidance_scale, 32)

        compressed = compress_image(photo_bytes, max_width=1024, quality=80)
        print(f"[{label}] foto: {len(compressed)//1024}KB guidance={guidance_scale}")

        client = _get_vertex_client()

        # Negative prompt base — protegge geometria per entrambe le varianti
        negative_base = (
            "distorted architecture, blurry textures, changing window frame positions, "
            "moving doors, wrong room proportions, deformed walls, different ceiling height, "
            "watermark, low quality, unrealistic"
        )

        # C5: negative prompt potenziato per forzare pulizia e cambio colore pareti
        if label == "C5_SMART_FULL":
            negative_prompt = (
                negative_base + ", "
                "clutter, trash bags, messy cables, old bulky furniture, "
                "dark shadows, blurry background, yellowish tint, original wall color retained, "
                "dirty surfaces, crowded space, outdated appliances visible, "
                "mismatched colors, cheap materials, same walls as original"
            )
        else:
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