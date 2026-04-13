"""
AI Service v16 — C4 + C5_SMART_FULL + D_FULL_SMART (two-stage)

Implementa le raccomandazioni Gemini:
  1. Guidance scale abbassato: D usa 10 (range 8-12), non 28-35
  2. Chain-of-Thought visual analysis: Gemini mappa sorgente luce + materiali
     prima di scrivere i prompt, inietta consistent shadows / global illumination
  3. Keyword fotorealismo nei prompt: depth of field f/8, ray tracing, HDR,
     soft shadows matching window light, realistic fabric folds, ambient occlusion
  4. Negative prompt potenziato: flat lighting, floating objects, sticker effect
  5. D two-stage workflow: Stage 1 pulisce la stanza, Stage 2 la arreda
     (risolve il collo di bottiglia "rimuovi E ricostruisci" in un colpo solo)

Fix ereditati v14-v15:
  - Multi-foto: N stanze con etichette sulle immagini
  - Inventario visivo obbligatorio (detected_elements)
  - Replacement logic per Imagen
  - Fallback indice_foto posizionale
"""
import asyncio
import base64
import hashlib
import io
import json
import os
import re
import traceback
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

# ── Guidance scale per variante ───────────────────────────────────────────────
# Fonte: raccomandazione Gemini — guidance alta crea "effetto collage"
# Range sicuro EDIT_MODE_DEFAULT: 1–30 (>30 = 0 immagini silenzioso)
# D two-stage: Stage1 (clean) usa 8, Stage2 (stage) usa 10
GUIDANCE = {
    "C4_FULL":          25,   # trasformazione totale fedele
    "C5_SMART_FULL":    22,   # pareti bianche, meno rigido
    "D_STAGE1_CLEAN":    8,   # pulizia stanza — massima libertà creativa
    "D_STAGE2_STAGE":   10,   # arredo su stanza pulita
}

_vertex_client = None

def _get_vertex_client():
    global _vertex_client
    if _vertex_client is None:
        print(f"[Imagen] init client project={PROJECT_ID} location={LOCATION}")
        _vertex_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    return _vertex_client

_gemini_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="gemini")
_imagen_executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="imagen")

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
        ob  = candidate.count("{") - candidate.count("}")
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
    stanze = analysis.get("stanze", [])
    actual = len(stanze)
    if actual != n_photos:
        print(f"[WARNING] Gemini ha restituito {actual} stanze su {n_photos} attese.")
    for i, room in enumerate(stanze):
        idx = room.get("indice_foto")
        if idx is None or not isinstance(idx, int) or idx >= n_photos:
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
        "tinteggiatura": int(budget * 0.25),
        "materiali":     int(budget * 0.20),
        "montaggio":     int(budget * 0.15),
    }

    system_instruction = (
        "Sei un Interior Designer senior e fotografo di interni specializzato in home staging per affitti brevi. "
        "Prima di generare qualsiasi prompt Imagen, esegui una MAPPATURA SPAZIALE della foto: "
        "1) Identifica la posizione della sorgente luminosa principale (es. 'finestra a destra a 45 gradi'). "
        "2) Identifica i materiali esistenti (pavimento, pareti, soffitto). "
        "3) Stima la direzione delle ombre portate. "
        "Questa analisi va nel campo 'light_analysis' e DEVE essere usata per costruire i prompt_en: "
        "ogni prompt deve includere 'consistent shadows', 'global illumination', 'ambient occlusion' "
        "per evitare l'effetto sticker/collage sugli oggetti aggiunti. "
        "REGOLA INVENTARIO VISIVO: il campo detected_elements elenca SOLO cio' che e' fisicamente visibile. "
        "Se non vedi finestre, scrivi 'no windows visible' e NON inserire tende nei prompt. "
        "Rispondi ESCLUSIVAMENTE con un oggetto JSON valido, senza markdown ne' backtick."
    )

    foto_index_list = "\n".join(
        f"  - FOTO {i}: indice_foto={i} — stanza {i + 1}"
        for i in range(n)
    )

    prompt = (
        f"Analizza queste {n} foto e produci una scheda di home staging per {dest_label}.\n\n"
        f"PARAMETRI: Budget \u20ac{budget} | Stile: {style} | Citta': {location}\n\n"
        f"DISTRIBUZIONE BUDGET:\n"
        f"  Arredi \u20ac{alloc['arredi']} | Tinteggiatura \u20ac{alloc['tinteggiatura']} "
        f"| Materiali \u20ac{alloc['materiali']} | Montaggio \u20ac{alloc['montaggio']}\n\n"
        f"REGOLA MATEMATICA: SOMMA(costo_totale_stanza) <= {budget}.\n\n"
        f"REGOLA MULTI-FOTO:\n"
        f"{foto_index_list}\n"
        f"Genera ESATTAMENTE {n} oggetti in 'stanze', uno per foto.\n"
        f"Non raggruppare foto diverse. Non saltare foto.\n\n"

        f"═══ STEP 1 — ANALISI VISIVA E LUCE (per ogni foto) ═══\n"
        f"Compila per ogni stanza:\n"
        f"  detected_elements: lista oggetti fisicamente visibili\n"
        f"  light_analysis: posizione sorgente luce (es. 'finestra sinistra, luce naturale diffusa'),\n"
        f"    materiali esistenti (es. 'pavimento parquet chiaro, pareti gialle opache'),\n"
        f"    direzione ombre portate (es. 'ombre verso destra e sul pavimento')\n\n"

        f"═══ STEP 2 — 3 VARIANTI STAGED ═══\n\n"

        f"C4_FULL (guidance=25) — Trasformazione totale:\n"
        f"  Inizia: 'The room features [New Color] walls covering all surfaces from floor to ceiling.'\n"
        f"  Usa replacement logic: 'In place of the [old item], a [new IKEA model] stands in the same position.'\n"
        f"  Inietta dalla light_analysis: 'soft shadows falling [direzione] consistent with [sorgente]'\n"
        f"  Chiudi con: 'global illumination, ambient occlusion, depth of field f/8, HDR, 8k.'\n\n"

        f"C5_SMART_FULL (guidance=22) — Pareti bianche + sostituzione precisa:\n"
        f"  Inizia: 'The room features freshly painted matte white walls covering all surfaces.'\n"
        f"  Replacement logic per ogni mobile in detected_elements.\n"
        f"  Inietta light coerenza: 'soft shadows matching [sorgente da light_analysis]'\n"
        f"  Chiudi con: 'consistent shadows, global illumination, ambient occlusion, ray tracing, 8k.'\n\n"

        f"D_FULL_SMART — TWO-STAGE (guidance Stage1=8, Stage2=10):\n"
        f"FILOSOFIA: workflow a due stadi per evitare il collo di bottiglia\n"
        f"  'rimuovi E ricostruisci contemporaneamente'.\n"
        f"  Stage 1 (pulisci): rimuovi tutto il superfluo, lascia struttura pulita.\n"
        f"  Stage 2 (arreda): aggiungi layering ricco sulla stanza pulita.\n\n"

        f"  PROMPT_STAGE1 (pulizia):\n"
        f"  Inizia: 'Empty, clean room. Remove all existing furniture, objects, clutter.'\n"
        f"  Mantieni: pareti (con nuova finitura materica {style}), pavimento originale, soffitto.\n"
        f"  Specifica finitura pareti coerente con {style}.\n"
        f"  Chiudi con: 'photorealistic, consistent lighting, 8k.'\n\n"

        f"  PROMPT_STAGE2 (arredo layering):\n"
        f"  Inizia descrivendo la stanza vuota con la nuova finitura dal Stage1.\n"
        f"  Aggiungi elementi in ordine spaziale (dal basso): pavimento → sedute → illuminazione → deco.\n"
        f"  Usa light_analysis per ombreggiature: 'shadows fall [direzione], soft shadows from [sorgente]'\n"
        f"  LAYERING OBBLIGATORIO: tappeto, 3-5 cuscini texture misti, pianta grande in vaso terracotta,\n"
        f"    mensola metallo nero con luci Edison, 2 stampe in cornici nere, lampada a stelo.\n"
        f"  Keyword fotorealismo OBBLIGATORIE: 'realistic fabric folds', 'reflections on polished surfaces',\n"
        f"    'soft shadows matching window light', 'consistent shadows', 'global illumination',\n"
        f"    'ambient occlusion', 'depth of field f/8', 'ray tracing', 'HDR'\n"
        f"  Chiudi con: 'professional real estate photography, 24mm wide angle lens, cinematic warm lighting,\n"
        f"    balanced exposure, perfectly staged interior, highly detailed, 8k.'\n\n"

        f"Restituisci SOLO questo JSON ({n} oggetti in 'stanze'):\n\n"
        "{{\n"
        "  \"valutazione_generale\": \"analisi visiva complessiva\",\n"
        "  \"punti_di_forza\": [\"p1\", \"p2\"],\n"
        "  \"criticita\": [\"c1\"],\n"
        f"  \"potenziale_str\": \"potenziale {dest_label} a {location}\",\n"
        "  \"tariffe\": {{\n"
        "    \"attuale_notte\": \"\u20acXX-YY\",\n"
        "    \"post_restyling_notte\": \"\u20acXX-YY\",\n"
        "    \"incremento_percentuale\": \"XX%\"\n"
        "  }},\n"
        f"  \"stanze\": [\n"
        f"    /* RIPETI {n} VOLTE — indice_foto 0..{n - 1} */\n"
        "    {{\n"
        "      \"nome\": \"Nome stanza\",\n"
        "      \"indice_foto\": 0,\n"
        "      \"detected_elements\": [\"elemento visibile 1\", \"no windows visible se assenti\"],\n"
        "      \"light_analysis\": {{\n"
        "        \"light_source\": \"es. finestra sinistra, luce naturale diffusa\",\n"
        "        \"existing_materials\": \"es. parquet chiaro, pareti gialle opache, soffitto bianco\",\n"
        "        \"shadow_direction\": \"es. ombre verso destra sul pavimento\"\n"
        "      }},\n"
        "      \"stato_attuale\": \"descrizione dettagliata\",\n"
        "      \"interventi\": [\n"
        "        {{\n"
        "          \"titolo\": \"nome intervento\",\n"
        f"          \"dettaglio\": \"prodotto specifico, brand, prezzo a {location}\",\n"
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
        f"          \"prompt_en\": \"The room features [color] walls. In place of [old item], a [IKEA model] stands in same position. Shadows fall [direzione da light_analysis] consistent with [sorgente]. Global illumination, ambient occlusion, depth of field f/8, HDR, {style} style, 8k.\",\n"
        "          \"interventi_lista\": [\n"
        "            {{\"voce\": \"Pittura pareti [colore]\", \"costo\": 300}},\n"
        "            {{\"voce\": \"Sostituzione [mobile] — IKEA [modello]\", \"costo\": 400}},\n"
        "            {{\"voce\": \"Tessili e decor\", \"costo\": 200}}\n"
        "          ],\n"
        "          \"costo_simulato\": 900\n"
        "        }},\n"
        "        {{\n"
        "          \"logic_id\": \"C5_SMART_FULL\",\n"
        "          \"guidance_scale\": 22,\n"
        "          \"prompt_en\": \"The room features freshly painted matte white walls covering all surfaces from floor to ceiling, replacing all previous colors and textures. In place of [old item], an IKEA [exact model] stands in same spot. Soft shadows matching [sorgente da light_analysis]. Consistent shadows, global illumination, ambient occlusion, ray tracing, 8k.\",\n"
        "          \"interventi_lista\": [\n"
        "            {{\"voce\": \"Pittura pareti bianco opaco\", \"costo\": 350}},\n"
        "            {{\"voce\": \"Sostituzione [mobile] — IKEA [modello]\", \"costo\": 450}},\n"
        "            {{\"voce\": \"Tappeto e cuscini\", \"costo\": 100}}\n"
        "          ],\n"
        "          \"costo_simulato\": 900\n"
        "        }},\n"
        "        {{\n"
        "          \"logic_id\": \"D_FULL_SMART\",\n"
        "          \"guidance_scale_stage1\": 8,\n"
        "          \"guidance_scale_stage2\": 10,\n"
        "          \"prompt_stage1\": \"Empty, clean room. Remove all existing furniture, objects and clutter. Keep the structure: [finitura pareti materica coerente con style, es. venetian plaster warm grey], [pavimento da detected_elements]. Photorealistic, consistent lighting matching [sorgente da light_analysis], 8k.\",\n"
        f"          \"prompt_stage2\": \"[Descrivi stanza vuota con nuova finitura]. Add: [tappeto design] on [pavimento]. Keep [mobile principale se funzionale] dressed with [3-4 cuscini texture misti]. Add: [mensola metallo nero con 3 luci Edison]. Hang: [2 stampe cornici nere]. Add corner: [Monstera in vaso terracotta]. Add: [lampada a stelo metallo nero]. Realistic fabric folds, reflections on polished surfaces, soft shadows matching [sorgente], consistent shadows, global illumination, ambient occlusion, depth of field f/8, ray tracing, HDR. Professional real estate photography, 24mm wide angle lens, cinematic warm lighting, balanced exposure, highly detailed, 8k.\",\n"
        "          \"interventi_lista\": [\n"
        "            {{\"voce\": \"Finitura pareti materica [tipo]\", \"costo\": 400}},\n"
        "            {{\"voce\": \"Tappeto design — H&M Home / Westwing\", \"costo\": 120}},\n"
        "            {{\"voce\": \"Cuscini 5 pz — Zara Home\", \"costo\": 80}},\n"
        "            {{\"voce\": \"Mensola metallo nero + luci Edison\", \"costo\": 90}},\n"
        "            {{\"voce\": \"2 stampe + cornici nere IKEA FISKBO\", \"costo\": 40}},\n"
        "            {{\"voce\": \"Pianta Monstera + vaso terracotta\", \"costo\": 45}},\n"
        "            {{\"voce\": \"Lampada a stelo metallo nero\", \"costo\": 60}}\n"
        "          ],\n"
        "          \"costo_simulato\": 835\n"
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
        "      \"categoria\": \"Layering e Decor\",\n"
        f"      \"articoli\": [\"item specifico stile {style}\"],\n"
        "      \"budget_stimato\": 0,\n"
        f"      \"negozi_consigliati\": \"IKEA, H&M Home, Zara Home, Westwing — {location}\"\n"
        "    }}\n"
        "  ],\n"
        "  \"titolo_annuncio_suggerito\": \"Titolo Airbnb max 50 caratteri\",\n"
        "  \"highlights_str\": [\"h1\", \"h2\", \"h3\"],\n"
        "  \"roi_restyling\": \"ROI: \u20acX, +\u20acY/notte, break-even Z notti\"\n"
        "}}"
    )

    # Etichetta testuale prima di ogni foto
    parts = []
    for i, p in enumerate(photos):
        parts.append({
            "text": f"[FOTO {i} — indice_foto: {i} — stanza {i + 1}]"
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


# ── STAGED PHOTOS ─────────────────────────────────────────────────────────────

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

            if logic_id == "D_FULL_SMART":
                # Two-stage: passa entrambi i prompt e i guidance separati
                p1 = esp.get("prompt_stage1", "")
                p2 = esp.get("prompt_stage2", "")
                g1 = esp.get("guidance_scale_stage1", GUIDANCE["D_STAGE1_CLEAN"])
                g2 = esp.get("guidance_scale_stage2", GUIDANCE["D_STAGE2_STAGE"])
                if not p1 or not p2 or not photo_bytes:
                    continue
                all_futures.append(("D_FULL_SMART", i,
                    loop.run_in_executor(
                        _imagen_executor, _approach_D_two_stage,
                        photo_bytes, p1, p2, g1, g2
                    )
                ))

            elif logic_id in ("C4_FULL", "C5_SMART_FULL"):
                prompt   = esp.get("prompt_en", "")
                guidance = GUIDANCE.get(logic_id, 25)
                if not prompt or not photo_bytes:
                    continue
                all_futures.append((logic_id, i,
                    loop.run_in_executor(
                        _imagen_executor, _approach_single,
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
            print(f"[{key} stanza {room_idx}] ERRORE: {result}\n{traceback.format_exc()}")
            results[room_idx][key] = None
        else:
            results[room_idx][key] = result

    return results


# ── Core: singola chiamata Imagen (C4, C5) ────────────────────────────────────

def _approach_single(photo_bytes: bytes, prompt: str,
                     guidance: int, label: str) -> str | None:
    try:
        compressed = compress_image(photo_bytes, max_width=1024, quality=80)
        print(f"[{label}] foto: {len(compressed)//1024}KB  guidance={guidance}")

        client = _get_vertex_client()

        # Negative prompt base — protegge geometria
        negative_base = (
            "distorted architecture, blurry textures, changing window positions, "
            "moving doors, wrong room proportions, deformed walls, different ceiling height, "
            "watermark, low quality, unrealistic, flat lighting, cartoon, illustration, "
            "floating objects, sticker effect, inconsistent shadows, bad lighting"
        )

        if label == "C5_SMART_FULL":
            negative = (
                negative_base + ", "
                "clutter, trash bags, old bulky furniture, yellowish tint, "
                "original wall color retained, dirty surfaces, same walls as original"
            )
        else:
            negative = negative_base

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
                guidance_scale=float(guidance),
                negative_prompt=negative,
                safety_filter_level="block_only_high",
            ),
        )

        if response.generated_images:
            print(f"[{label}] SUCCESS")
            return base64.b64encode(
                response.generated_images[0].image.image_bytes
            ).decode()

        print(f"[{label}] 0 immagini (guidance={guidance}). "
              f"generated_images={response.generated_images!r}")
        return None

    except Exception as e:
        print(f"[{label}] ERRORE: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return None


# ── Core: two-stage D ─────────────────────────────────────────────────────────

def _approach_D_two_stage(photo_bytes: bytes,
                           prompt_stage1: str, prompt_stage2: str,
                           guidance1: int, guidance2: int) -> str | None:
    """
    Stage 1: pulisce la stanza (guidance basso = massima libertà di rimozione)
    Stage 2: arreda la stanza pulita con layering ricco e keyword fotorealismo
    """
    try:
        # ── Stage 1: Clean ────────────────────────────────────────────────────
        compressed = compress_image(photo_bytes, max_width=1024, quality=80)
        print(f"[D Stage1 Clean] {len(compressed)//1024}KB  guidance={guidance1}")

        client = _get_vertex_client()

        negative_clean = (
            "furniture, objects, clutter, trash bags, messy cables, "
            "distorted architecture, watermark, low quality, unrealistic"
        )

        r1 = client.models.edit_image(
            model="imagen-3.0-capability-001",
            prompt=prompt_stage1,
            reference_images=[
                genai_types.RawReferenceImage(
                    reference_id=1,
                    reference_image=genai_types.Image(image_bytes=compressed),
                )
            ],
            config=genai_types.EditImageConfig(
                edit_mode="EDIT_MODE_DEFAULT",
                number_of_images=1,
                guidance_scale=float(guidance1),
                negative_prompt=negative_clean,
                safety_filter_level="block_only_high",
            ),
        )

        if not r1.generated_images:
            print(f"[D Stage1] 0 immagini. Fallback su foto originale per Stage2.")
            stage1_bytes = compress_image(photo_bytes, max_width=1024, quality=80)
        else:
            print(f"[D Stage1] SUCCESS → passo a Stage2")
            stage1_bytes = r1.generated_images[0].image.image_bytes

        # ── Stage 2: Stage ────────────────────────────────────────────────────
        print(f"[D Stage2 Stage]  guidance={guidance2}")

        negative_stage = (
            "empty room, bare walls, clutter, trash bags, messy cables, "
            "dark shadows, bad exposure, overexposed, underexposed, harsh lighting, "
            "noise, grainy, unwelcoming atmosphere, bad framing, overcrowded, "
            "cartoon style, flat lighting, floating objects, sticker effect, "
            "inconsistent shadows, distorted architecture, watermark, low quality"
        )

        r2 = client.models.edit_image(
            model="imagen-3.0-capability-001",
            prompt=prompt_stage2,
            reference_images=[
                genai_types.RawReferenceImage(
                    reference_id=1,
                    reference_image=genai_types.Image(image_bytes=stage1_bytes),
                )
            ],
            config=genai_types.EditImageConfig(
                edit_mode="EDIT_MODE_DEFAULT",
                number_of_images=1,
                guidance_scale=float(guidance2),
                negative_prompt=negative_stage,
                safety_filter_level="block_only_high",
            ),
        )

        if r2.generated_images:
            print(f"[D Stage2] SUCCESS")
            return base64.b64encode(
                r2.generated_images[0].image.image_bytes
            ).decode()

        print(f"[D Stage2] 0 immagini (guidance={guidance2}).")
        return None

    except Exception as e:
        print(f"[D two-stage] ERRORE: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return None