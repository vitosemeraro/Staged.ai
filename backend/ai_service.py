"""
AI Service v12 — C-variants experimental staging
Genera:
  A: generate_images base (baseline)
  B: generate_images + vincoli geometrici
  C: edit_image DEFAULT con RawReferenceImage (la migliore per fedeltà)
  C1-C4: 4 sotto-varianti della C con guidance_scale crescente e prompt specifici:
    C1 SOFT      guidance=10  — luce, tessili, pulizia visiva
    C2 CHROMATIC guidance=15  — colore pareti + decor
    C3 BOLD      guidance=20  — sostituzione arredi principali
    C4 FULL      guidance=25  — trasformazione totale coordinata
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
_imagen_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="imagen")

_analysis_cache: dict[str, dict] = {}


async def _noop():
    return None


def compress_image(img_bytes: bytes, max_width: int = 1200, quality: int = 85) -> bytes:
    if not HAS_PIL:
        return img_bytes
    try:
        img = PILImage.open(io.BytesIO(img_bytes))
        img = ImageOps.exif_transpose(img)  # corregge rotazione foto da mobile
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
        f"REGOLA prompt_imagen (base, per approcci A e B):\n"
        f"Scrivi un prompt fotografico professionale in inglese per Imagen 3.\n"
        f"Descrivi la stanza DOPO il restyling in stile {style}. Max 50 parole.\n\n"
        f"REGOLA esperimenti_staged (4 varianti sperimentali per approccio C):\n"
        f"Per ogni stanza genera 4 varianti con aggressivita' crescente.\n"
        f"Ogni variante deve:\n"
        f"1. Avere un prompt IN INGLESE denso e specifico (max 60 parole) che descriva\n"
        f"   UN INSIEME COERENTE di interventi visibili nella foto: texture pareti,\n"
        f"   colori, arredi specifici, tessili, piante, oggetti decorativi.\n"
        f"   Il prompt deve specificare SEMPRE: wall texture/color + main furniture + textiles.\n"
        f"2. Avere una lista minima di interventi in italiano (3-6 voci) con costo stimato.\n"
        f"3. Rientrare nel budget totale di \u20ac{budget} (il costo_simulato di TUTTE e 4\n"
        f"   le varianti deve essere <= {budget}, ognuna e' indipendente).\n\n"
        f"Le 4 varianti devono seguire questi gradienti:\n"
        f"- C1 SOFT (guidance=10): Solo luce e tessili. Tende, cuscini, piante, tappeto.\n"
        f"  Nessun cambio strutturale. Mantieni colori pareti originali.\n"
        f"- C2 CHROMATIC (guidance=15): Cambio colore pareti + complementi medi\n"
        f"  (quadri, specchi, lampade). Verifica se Imagen 'dipinge' le pareti.\n"
        f"- C3 BOLD (guidance=20): Sostituzione arredi principali (tavolo, sedie, divano)\n"
        f"  con modelli di design coerenti con {style}. Mantieni pareti.\n"
        f"- C4 FULL (guidance=25): Trasformazione totale: pareti + arredi + decor.\n"
        f"  Test di massima aggressivita' per vedere fino a dove Imagen mantiene\n"
        f"  la geometria originale della stanza.\n\n"
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
        f"      \"prompt_imagen\": \"Photorealistic interior, {style} style room, [specific furniture and colors], warm natural light, 4k\",\n"
        "      \"esperimenti_staged\": [\n"
        "        {{\n"
        "          \"logic_id\": \"C1_SOFT\",\n"
        "          \"guidance_scale\": 10,\n"
        "          \"prompt_en\": \"[60-word English prompt: keep original wall color, add linen curtains, wool rug, plants, cushions. Specify exact colors and materials.]\",\n"
        "          \"interventi_lista\": [\n"
        "            {{\"voce\": \"Tende lino bianco\", \"costo\": 50}},\n"
        "            {{\"voce\": \"Tappeto lana grigio 160x230\", \"costo\": 80}},\n"
        "            {{\"voce\": \"Piante e vasi ceramica\", \"costo\": 40}}\n"
        "          ],\n"
        "          \"costo_simulato\": 170\n"
        "        }},\n"
        "        {{\n"
        "          \"logic_id\": \"C2_CHROMATIC\",\n"
        "          \"guidance_scale\": 15,\n"
        "          \"prompt_en\": \"[60-word English prompt: change wall color to specific shade, add medium decor like art prints, mirror, pendant lamp. Specify wall texture: matte plaster.]\",\n"
        "          \"interventi_lista\": [\n"
        "            {{\"voce\": \"Pareti grigio tortora opaco\", \"costo\": 120}},\n"
        "            {{\"voce\": \"Stampe e cornici\", \"costo\": 60}},\n"
        "            {{\"voce\": \"Lampada a sospensione\", \"costo\": 80}}\n"
        "          ],\n"
        "          \"costo_simulato\": 260\n"
        "        }},\n"
        "        {{\n"
        "          \"logic_id\": \"C3_BOLD\",\n"
        "          \"guidance_scale\": 20,\n"
        "          \"prompt_en\": \"[60-word English prompt: replace main furniture (table, chairs, sofa) with specific design pieces matching style. Keep original walls. Specify materials and colors.]\",\n"
        "          \"interventi_lista\": [\n"
        "            {{\"voce\": \"Tavolo rovere naturale\", \"costo\": 150}},\n"
        "            {{\"voce\": \"Sedie design (x4)\", \"costo\": 160}},\n"
        "            {{\"voce\": \"Tappeto juta naturale\", \"costo\": 70}}\n"
        "          ],\n"
        "          \"costo_simulato\": 380\n"
        "        }},\n"
        "        {{\n"
        "          \"logic_id\": \"C4_FULL\",\n"
        "          \"guidance_scale\": 25,\n"
        "          \"prompt_en\": \"[60-word English prompt: full transformation — specify new wall color+texture, all new furniture, textiles, lighting and decor. Coherent with style. Test geometry preservation.]\",\n"
        "          \"interventi_lista\": [\n"
        "            {{\"voce\": \"Pareti bianco puro opaco\", \"costo\": 120}},\n"
        "            {{\"voce\": \"Tavolo + sedie completo\", \"costo\": 280}},\n"
        "            {{\"voce\": \"Tessili, decor e illuminazione\", \"costo\": 200}}\n"
        "          ],\n"
        "          \"costo_simulato\": 600\n"
        "        }}\n"
        "      ]\n"
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


# ── STAGED PHOTOS ─────────────────────────────────────────────────────────────
# Output per ogni stanza:
# {
#   "A_base":      "<b64>",
#   "B_geometric": "<b64>",
#   "C_base":      "<b64>",
#   "C1_SOFT":     "<b64>",
#   "C2_CHROMATIC": "<b64>",
#   "C3_BOLD":     "<b64>",
#   "C4_FULL":     "<b64>",
# }

async def generate_staged_photos(photos: list, analysis: dict) -> list:
    stanze = analysis.get("stanze", [])
    loop   = asyncio.get_running_loop()
    all_futures = []

    for i, room in enumerate(stanze):
        idx         = room.get("indice_foto", 0)
        prompt_base = room.get("prompt_imagen", "")
        esperimenti = room.get("esperimenti_staged", [])
        photo_bytes = photos[idx]["content"] if idx < len(photos) else None

        if not prompt_base:
            continue

        # A: generate_images base
        all_futures.append(("A_base", i,
            loop.run_in_executor(_imagen_executor, _approach_A_base, prompt_base)
        ))
        # B: generate_images + vincoli geometrici
        all_futures.append(("B_geometric", i,
            loop.run_in_executor(_imagen_executor, _approach_B_geometric, prompt_base)
        ))
        # C base: edit_image DEFAULT
        all_futures.append(("C_base", i,
            loop.run_in_executor(_imagen_executor, _approach_C_edit,
                                 photo_bytes, prompt_base, 12)
        ))
        # C1-C4: varianti sperimentali con guidance_scale e prompt specifici
        for esp in esperimenti:
            logic_id   = esp.get("logic_id", "Cx")
            prompt_esp = esp.get("prompt_en", prompt_base)
            guidance   = esp.get("guidance_scale", 15)
            all_futures.append((logic_id, i,
                loop.run_in_executor(_imagen_executor, _approach_C_edit,
                                     photo_bytes, prompt_esp, guidance)
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


# ── Approccio A ───────────────────────────────────────────────────────────────

def _approach_A_base(prompt: str) -> str | None:
    try:
        print(f"[A_base] START")
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
            return base64.b64encode(response.generated_images[0].image.image_bytes).decode()
        return None
    except Exception as e:
        print(f"[A_base] ERRORE: {type(e).__name__}: {e}")
        return None


# ── Approccio B ───────────────────────────────────────────────────────────────

def _approach_B_geometric(prompt: str) -> str | None:
    try:
        print(f"[B_geometric] START")
        geo_prompt = (
            "Maintain exactly the same room layout, window positions, wall colors "
            "and floor material as the reference. Only replace movable furniture and decor. "
            + prompt
            + " Negative: moving walls, changing window shapes, different floor, "
            "distorted architecture, different room proportions."
        )
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
            return base64.b64encode(response.generated_images[0].image.image_bytes).decode()
        return None
    except Exception as e:
        print(f"[B_geometric] ERRORE: {type(e).__name__}: {e}")
        return None


# ── Approccio C (base e varianti) ────────────────────────────────────────────

def _approach_C_edit(photo_bytes: bytes | None, prompt: str, guidance_scale: int) -> str | None:
    """
    edit_image con EDIT_MODE_DEFAULT + RawReferenceImage.
    guidance_scale variabile: basso (10) = più fedele, alto (25) = più creativo.
    Negative prompt dinamico per proteggere la geometria.
    """
    if not photo_bytes:
        print(f"[C guidance={guidance_scale}] nessuna foto, skip")
        return None
    try:
        compressed = compress_image(photo_bytes, max_width=1024, quality=80)
        print(f"[C guidance={guidance_scale}] foto: {len(compressed)//1024}KB")

        client = _get_vertex_client()

        negative_prompt = (
            "distorted architecture, blurry textures, changing window frame positions, "
            "moving doors, wrong room proportions, deformed walls, different ceiling height, "
            "watermark, low quality, unrealistic"
        )

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
            print(f"[C guidance={guidance_scale}] SUCCESS")
            return base64.b64encode(response.generated_images[0].image.image_bytes).decode()
        print(f"[C guidance={guidance_scale}] nessuna immagine generata")
        return None
    except Exception as e:
        print(f"[C guidance={guidance_scale}] ERRORE: {type(e).__name__}: {e}")
        return None
