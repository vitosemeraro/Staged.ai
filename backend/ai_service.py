"""
AI Service v21 — E_WALL_FORCE single-stage

Architettura:
  - D_FULL_SMART: DISABILITATO (riattivare rimuovendo il 'continue')
  - E_WALL_FORCE: single-stage, 1 chiamata per stanza (4 stanze = 4 call totali)
      guidance=15, solid opaque coating, protezione bagni anti-finestre-fantasma

Rispetto a v17:
  - Two-stage rimosso → da 16 chiamate a 4 per job
  - Semaforo reale: il task parte solo quando il worker è libero
  - is_bathroom: no finestre aggiuntive, piastrelle come elementi strutturali
  - Retry 2× con backoff 15s
  - Guidance Stage1 14→15, Stage2 eliminato (single-stage)
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

# ── Guidance hardcodati ───────────────────────────────────────────────────────
# EDIT_MODE_DEFAULT: valori > 30 = 0 immagini silenzioso.
# D: Stage1=8  (libertà massima), Stage2=10
# E: Stage1=14 (aggressivo sul colore pareti), Stage2=10
GUIDANCE = {
    "C4_FULL":          25,
    "C5_SMART_FULL":    22,
    "D_STAGE1_CLEAN":    8,
    "D_STAGE2_STAGE":   10,
    "E_STAGE1_WALL":    15,   # alzato a 15: forza copertura colore pareti originale
    "E_STAGE2_STAGE":   12,   # alzato a 12: più dettaglio nell'arredo finale
}

_vertex_client = None

def _get_vertex_client():
    global _vertex_client
    if _vertex_client is None:
        print(f"[Imagen] init client project={PROJECT_ID} location={LOCATION}")
        _vertex_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    return _vertex_client

_gemini_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="gemini")
_imagen_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="imagen")

# Semaforo: max 2 chiamate Imagen simultanee
_imagen_semaphore: asyncio.Semaphore | None = None

def _get_semaphore() -> asyncio.Semaphore:
    global _imagen_semaphore
    if _imagen_semaphore is None:
        _imagen_semaphore = asyncio.Semaphore(2)
    return _imagen_semaphore

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


async def validate_input_photos(photos: list) -> dict:
    """
    Stub di compatibilità — la versione completa è nelle v18+.
    Restituisce tutte le foto come valide senza chiamare Gemini.
    """
    n = len(photos)
    return {
        "valid":       [True] * n,
        "issues":      ["ok"] * n,
        "room_types":  ["unknown"] * n,
        "is_outdoor":  [False] * n,
        "suggestions": [""] * n,
        "structural":  [[] for _ in range(n)],
        "warnings":    [],
        "layout_hint": "",
        "all_valid":   True,
    }


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
        "Sei un Interior Designer senior e fotografo di interni specializzato in home staging. "
        "Prima di generare qualsiasi prompt Imagen, esegui una MAPPATURA SPAZIALE: "
        "1) Identifica la posizione della sorgente luminosa principale. "
        "2) Identifica i materiali esistenti (pavimento, pareti, soffitto). "
        "3) Stima la direzione delle ombre portate. "
        "4) Identifica il COLORE ESATTO delle pareti attuali (es. 'giallo paglierino', "
        "'beige opaco', 'bianco sporco') — questa informazione è critica per la variante E. "
        "5) Identifica se la stanza è un BAGNO tramite il campo 'is_bathroom'.\n\n"
        "REGOLE FERREE v21:\n"
        "PROTOCOLLO BAGNO: Se is_bathroom=true, è VIETATO aggiungere finestre non visibili "
        "nella foto originale. Le piastrelle esistenti sono elementi strutturali: non vanno "
        "rimosse ma ricolorate con 'tile paint' o coperte con 'micro-cement finish overlay'. "
        "Non mettere mai finestre fittizie in un bagno interno.\n"
        "PITTURA PARETI: Nei prompt usa sempre 'solid opaque structural coating' per coprire "
        "il colore originale. Mai 'paint over' o 'repaint' da soli — aggiungere sempre "
        "'zero bleed-through, 100% opacity, no original color visible'.\n"
        "COERENZA COSTI: Ogni elemento aggiunto nel rendering (piastrelle, mobili, luci) "
        "DEVE avere una voce corrispondente in 'interventi' con costo_min e costo_max.\n"
        "I prompt_en devono includere 'consistent shadows', 'global illumination', "
        "'ambient occlusion' per evitare l'effetto sticker. "
        "detected_elements elenca SOLO cio' che e' fisicamente visibile. "
        "Rispondi ESCLUSIVAMENTE con un oggetto JSON valido, senza markdown ne' backtick."
    )

    foto_index_list = "\n".join(
        f"  - FOTO {i}: indice_foto={i} — stanza {i + 1}" for i in range(n)
    )

    prompt = (
        f"Analizza queste {n} foto e produci una scheda di home staging per {dest_label}.\n\n"
        f"PARAMETRI: Budget \u20ac{budget} | Stile: {style} | Citta': {location}\n\n"
        f"DISTRIBUZIONE BUDGET: Arredi \u20ac{alloc['arredi']} | "
        f"Tinteggiatura \u20ac{alloc['tinteggiatura']} | "
        f"Materiali \u20ac{alloc['materiali']} | Montaggio \u20ac{alloc['montaggio']}\n\n"
        f"REGOLA MATEMATICA: SOMMA(costo_totale_stanza) <= {budget}.\n\n"
        f"REGOLA MULTI-FOTO:\n{foto_index_list}\n"
        f"Genera ESATTAMENTE {n} oggetti in 'stanze'. Non raggruppare. Non saltare.\n\n"

        f"STEP 1 — ANALISI VISIVA per ogni foto:\n"
        f"  detected_elements: lista oggetti fisicamente visibili\n"
        f"  light_analysis: sorgente luce, materiali esistenti, direzione ombre\n"
        f"  current_wall_color: colore esatto pareti attuali (es. 'giallo paglierino opaco')\n"
        f"  target_wall_color:  nuovo colore pareti coerente con {style}\n"
        f"  target_wall_finish: finitura (es. 'venetian plaster warm grey', 'matte white')\n\n"

        f"STEP 2 — GENERA 2 VARIANTI TWO-STAGE per ogni stanza:\n\n"

        f"═══ D_FULL_SMART (Stage1 guidance=8, Stage2 guidance=10) ═══\n"
        f"Stage1: rimuovi mobili + clutter, applica nuova finitura pareti.\n"
        f"Stage2: aggiungi layering su stanza pulita. Chiudi con keyword fotorealismo.\n\n"

        f"═══ E_WALL_FORCE (Stage1 guidance=14, Stage2 guidance=10) ═══\n"
        f"DIFFERENZA CHIAVE rispetto a D: Stage1 è più aggressivo sul cambio colore pareti.\n"
        f"REGOLE FERREE per E Stage1:\n"
        f"  1. Nomina esplicitamente il colore da eliminare: "
        f"'The original [current_wall_color] wall color MUST be entirely replaced.'\n"
        f"  2. Nomina esplicitamente il nuovo colore: "
        f"'All walls are now [target_wall_finish] [target_wall_color].'\n"
        f"  3. Aggiungi: 'Solid, opaque paint, no transparency, no bleed-through, "
        f"no traces of [current_wall_color] visible anywhere.'\n"
        f"  4. Chiudi con: 'Complete architectural renovation, empty room, "
        f"consistent lighting, photorealistic, 8k.'\n"
        f"REGOLE per E Stage2:\n"
        f"  1. INIZIA SEMPRE con conferma esplicita delle nuove pareti: "
        f"'The room is now perfectly clean with new [target_wall_finish] [target_wall_color] "
        f"walls from Stage 1. These walls show no trace of the previous [current_wall_color].'\n"
        f"  2. Poi aggiungi stesso layering di D: tappeto, cuscini, pianta, mensola, "
        f"stampe, lampada.\n"
        f"  3. Inietta light_analysis per ombreggiature coerenti.\n"
        f"  4. Chiudi con keyword fotorealismo + 'professional real estate photography, "
        f"24mm wide angle lens, cinematic warm lighting, 8k.'\n\n"

        f"Restituisci SOLO questo JSON ({n} oggetti in 'stanze'):\n\n"
        "{{\n"
        "  \"valutazione_generale\": \"analisi visiva complessiva\",\n"
        "  \"punti_di_forza\": [\"p1\"],\n"
        "  \"criticita\": [\"c1\"],\n"
        f"  \"potenziale_str\": \"potenziale {dest_label} a {location}\",\n"
        "  \"tariffe\": {{\n"
        "    \"attuale_notte\": \"\u20acXX-YY\",\n"
        "    \"post_restyling_notte\": \"\u20acXX-YY\",\n"
        "    \"incremento_percentuale\": \"XX%\"\n"
        "  }},\n"
        f"  \"stanze\": [\n"
        f"    /* RIPETI {n} VOLTE */\n"
        "    {{\n"
        "      \"nome\": \"Nome stanza\",\n"
        "      \"indice_foto\": 0,\n"
        "      \"room_type\": \"bagno|cucina|camera|soggiorno|balcone|altro\",\n"
        "      \"is_bathroom\": false,\n"
        "      \"detected_elements\": [\"elemento visibile\", \"no windows visible se assenti\"],\n"
        "      \"current_wall_color\": \"es. giallo paglierino opaco\",\n"
        "      \"target_wall_color\": \"es. warm grey\",\n"
        "      \"target_wall_finish\": \"es. venetian plaster\",\n"
        "      \"light_analysis\": {{\n"
        "        \"light_source\": \"es. finestra sinistra, luce diffusa\",\n"
        "        \"existing_materials\": \"es. parquet chiaro, pareti gialle, soffitto bianco\",\n"
        "        \"shadow_direction\": \"es. ombre verso destra\"\n"
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
        "          \"logic_id\": \"D_FULL_SMART\",\n"
        "          \"guidance_scale_stage1\": 8,\n"
        "          \"guidance_scale_stage2\": 10,\n"
        "          \"prompt_stage1\": \"Empty, clean room. Remove all furniture and clutter. Keep structure: [target_wall_finish] [target_wall_color] walls, [pavimento originale]. Photorealistic, consistent lighting matching [light_source], 8k.\",\n"
        f"          \"prompt_stage2\": \"[Stanza vuota con [target_wall_finish] [target_wall_color] walls]. Add: [tappeto design] on [pavimento]. [Mobile principale se funzionale] with [3-4 cuscini texture misti]. Add: [mensola metallo nero con luci Edison]. Hang: [2 stampe cornici nere]. Corner: [Monstera in vaso terracotta]. Add: [lampada a stelo metallo nero]. Realistic fabric folds, reflections on polished surfaces, soft shadows matching [light_source], consistent shadows, global illumination, ambient occlusion, depth of field f/8, ray tracing, HDR. Professional real estate photography, 24mm wide angle lens, cinematic warm lighting, 8k.\",\n"
        "          \"interventi_lista\": [\n"
        "            {{\"voce\": \"Finitura pareti [target_wall_finish]\", \"costo\": 400}},\n"
        "            {{\"voce\": \"Tappeto design — H&M Home\", \"costo\": 120}},\n"
        "            {{\"voce\": \"Cuscini 5 pz — Zara Home\", \"costo\": 80}},\n"
        "            {{\"voce\": \"Mensola metallo nero + luci Edison\", \"costo\": 90}},\n"
        "            {{\"voce\": \"2 stampe + cornici nere IKEA FISKBO\", \"costo\": 40}},\n"
        "            {{\"voce\": \"Pianta Monstera + vaso terracotta\", \"costo\": 45}},\n"
        "            {{\"voce\": \"Lampada a stelo metallo nero\", \"costo\": 60}}\n"
        "          ],\n"
        "          \"costo_simulato\": 835\n"
        "        }},\n"
        "        {{\n"
        "          \"logic_id\": \"E_WALL_FORCE\",\n"
        "          \"guidance_scale_stage1\": 14,\n"
        "          \"guidance_scale_stage2\": 10,\n"
        "          \"current_wall_color\": \"[colore esatto pareti attuali da current_wall_color]\",\n"
        "          \"target_wall_color\": \"[nuovo colore da target_wall_color]\",\n"
        "          \"target_wall_finish\": \"[finitura da target_wall_finish]\",\n"
        "          \"prompt_stage1\": \"Complete architectural renovation. The original [current_wall_color] wall color MUST be entirely replaced. All walls are now [target_wall_finish] [target_wall_color]. Solid, opaque paint, no transparency, no bleed-through, no traces of [current_wall_color] visible anywhere. Empty room, no furniture. Consistent lighting matching [light_source]. Photorealistic, 8k.\",\n"
        f"          \"prompt_stage2\": \"The room is now perfectly clean with new [target_wall_finish] [target_wall_color] walls from Stage 1. These walls show no trace of the previous [current_wall_color]. On these new walls: [tappeto design] on [pavimento]. [Mobile principale se funzionale] with [3-4 cuscini texture misti]. [mensola metallo nero con luci Edison] against the wall. [2 stampe cornici nere] hung above. Corner: [Monstera in vaso terracotta]. [lampada a stelo metallo nero] for warm atmosphere. Soft shadows matching [light_source], consistent shadows, global illumination, ambient occlusion, depth of field f/8, ray tracing, HDR. Professional real estate photography, 24mm wide angle lens, cinematic warm lighting, balanced exposure, highly detailed, 8k.\",\n"
        "          \"interventi_lista\": [\n"
        "            {{\"voce\": \"Tinteggiatura [target_wall_finish] [target_wall_color]\", \"costo\": 400}},\n"
        "            {{\"voce\": \"Tappeto design — H&M Home\", \"costo\": 120}},\n"
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

    parts = []
    for i, p in enumerate(photos):
        parts.append({"text": f"[FOTO {i} — indice_foto: {i} — stanza {i + 1}]"})
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
        is_bathroom = room.get("is_bathroom", False) or \
                      room.get("room_type", "").lower() in ("bagno", "bathroom")
        cwc = room.get("current_wall_color", "")

        for esp in esperimenti:
            logic_id = esp.get("logic_id", "")

            # ── D: DISABILITATO — riattivare quando E è stabile
            if logic_id == "D_FULL_SMART":
                continue

            # ── E: single-stage (no two-stage) — 1 chiamata per stanza invece di 2
            # Dimezza il carico API: 4 stanze = 4 chiamate totali
            elif logic_id == "E_WALL_FORCE":
                # Usa prompt_stage2 come prompt finale (più ricco di dettaglio arredo)
                prompt = esp.get("prompt_stage2", "") or esp.get("prompt_stage1", "")
                if not prompt or not photo_bytes:
                    continue
                all_futures.append(("E_WALL_FORCE", i,
                    photo_bytes, prompt, cwc, is_bathroom
                ))

    # ── Esegui in sequenza con semaforo reale (non pre-submitting) ───────────
    sem = _get_semaphore()
    results = [{} for _ in stanze]

    async def _run_one(label, room_idx, photo_bytes, prompt, cwc, is_bathroom):
        async with sem:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                _imagen_executor, _approach_E_single,
                photo_bytes, prompt, cwc, is_bathroom
            )
            return label, room_idx, result

    tasks = [
        _run_one(label, room_idx, pb, prompt, cwc, ib)
        for label, room_idx, pb, prompt, cwc, ib in all_futures
    ]
    gathered = await asyncio.gather(*tasks, return_exceptions=True)

    for item in gathered:
        if isinstance(item, Exception):
            print(f"[generate_staged_photos] ERRORE: {item}")
            continue
        label, room_idx, result = item
        results[room_idx][label] = result

    return results


# ── Core: singola chiamata (C4, C5) ──────────────────────────────────────────

def _approach_single(photo_bytes: bytes, prompt: str,
                     guidance: int, label: str) -> str | None:
    import time
    for attempt in range(1, 3):  # max 2 tentativi
        try:
            compressed = compress_image(photo_bytes, max_width=1024, quality=80)
            print(f"[{label}] attempt={attempt} {len(compressed)//1024}KB guidance={guidance}")
            client = _get_vertex_client()

            negative_base = (
                "distorted architecture, blurry textures, changing window positions, "
                "moving doors, wrong room proportions, deformed walls, different ceiling height, "
                "watermark, low quality, unrealistic, flat lighting, cartoon, illustration, "
                "floating objects, sticker effect, inconsistent shadows"
            )
            negative = (
                negative_base + ", clutter, trash bags, old bulky furniture, yellowish tint, "
                "original wall color retained, dirty surfaces, same walls as original"
            ) if label == "C5_SMART_FULL" else negative_base

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
                print(f"[{label}] SUCCESS attempt={attempt}")
                return base64.b64encode(response.generated_images[0].image.image_bytes).decode()
            print(f"[{label}] 0 immagini (guidance={guidance}) — possibile safety filter o quota")
            return None
        except Exception as e:
            err_str = f"{type(e).__name__}: {e}"
            print(f"[{label}] ERRORE attempt={attempt}: {err_str}")
            if attempt < 2:
                wait = 10 * attempt  # 10s poi 20s
                print(f"[{label}] retry tra {wait}s…")
                time.sleep(wait)
            else:
                print(f"[{label}] fallito dopo 2 tentativi")
                return None


# ── Core: D two-stage (INVARIATA) ────────────────────────────────────────────

def _approach_D_two_stage(photo_bytes: bytes,
                           prompt_stage1: str, prompt_stage2: str,
                           guidance1: int, guidance2: int) -> str | None:
    """Stage1 g=8 (libero), Stage2 g=10 (layering). Retry con backoff."""
    import time

    # Semplifica Stage2: riduci keyword tecniche a 3 essenziali
    prompt_stage2_clean = (
        prompt_stage2.split("Realistic fabric folds")[0].strip().rstrip(",")
        + ". Photorealistic, global illumination, consistent shadows, cinematic lighting, high-end interior photography."
    )

    for attempt in range(1, 3):
        try:
            compressed = compress_image(photo_bytes, max_width=1024, quality=80)
            print(f"[D Stage1] attempt={attempt} {len(compressed)//1024}KB guidance={guidance1}")
            client = _get_vertex_client()

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
                    negative_prompt=(
                        "furniture, objects, clutter, trash bags, messy cables, "
                        "distorted architecture, watermark, low quality, unrealistic"
                    ),
                    safety_filter_level="block_only_high",
                ),
            )

            stage1_bytes = (
                r1.generated_images[0].image.image_bytes
                if r1.generated_images
                else compress_image(photo_bytes, max_width=1024, quality=80)
            )
            if not r1.generated_images:
                print("[D Stage1] 0 immagini → fallback su originale")
            else:
                print("[D Stage1] SUCCESS → Stage2")

            print(f"[D Stage2] guidance={guidance2}")
            r2 = client.models.edit_image(
                model="imagen-3.0-capability-001",
                prompt=prompt_stage2_clean,
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
                    negative_prompt=(
                        "empty room, bare walls, clutter, trash bags, dark shadows, "
                        "overexposed, noise, cartoon, floating objects, "
                        "inconsistent shadows, distorted architecture, watermark"
                    ),
                    safety_filter_level="block_only_high",
                ),
            )
            if r2.generated_images:
                print(f"[D Stage2] SUCCESS attempt={attempt}")
                return base64.b64encode(r2.generated_images[0].image.image_bytes).decode()
            print(f"[D Stage2] 0 immagini (guidance={guidance2}) — safety filter o quota")
            return None

        except Exception as e:
            print(f"[D two-stage] ERRORE attempt={attempt}: {type(e).__name__}: {e}")
            if attempt < 2:
                wait = 10 * attempt
                print(f"[D] retry tra {wait}s…")
                time.sleep(wait)
            else:
                print("[D] fallito dopo 2 tentativi")
                return None


# ── Core: E single-stage ─────────────────────────────────────────────────────
# Una sola chiamata per stanza: prompt ricco con wall recolor + arredo.
# Guidance=15 → trasformazione decisa ma ancora ancorata alla foto originale.

def _approach_E_single(photo_bytes: bytes, prompt: str,
                       current_wall_color: str = "",
                       is_bathroom: bool = False) -> str | None:
    import time

    # Suffix: forza cambio colore + protezione bagno
    suffix = (
        f" The original {current_wall_color} color is completely replaced by solid opaque "
        "structural coating. Zero bleed-through. 100% opacity."
        if current_wall_color
        else " All walls: solid opaque structural coating. Zero bleed-through."
    )
    if is_bathroom:
        suffix += (
            " BATHROOM: DO NOT add windows absent in original. "
            "Keep all tile boundaries. Only recolor tiles or apply micro-cement overlay."
        )
    suffix += " Photorealistic, global illumination, consistent shadows, cinematic lighting, high-end interior photography."

    prompt_final = prompt.split("Realistic fabric folds")[0].strip().rstrip(",") + suffix

    neg = (
        "distorted architecture, watermark, low quality, unrealistic, "
        "original wall color retained, bleed-through, transparent paint, "
        "floating objects, sticker effect, inconsistent shadows, cartoon, "
        "overexposed, noise, dark shadows"
    )
    if current_wall_color:
        neg += f", {current_wall_color} on walls"
    if is_bathroom:
        neg += ", added windows, fake windows, exterior view, missing tiles"

    guidance = GUIDANCE["E_STAGE1_WALL"]  # 15

    for attempt in range(1, 3):
        try:
            compressed = compress_image(photo_bytes, max_width=1024, quality=80)
            bath_tag = " [BAGNO]" if is_bathroom else ""
            print(f"[E-single{bath_tag}] attempt={attempt} {len(compressed)//1024}KB g={guidance}")
            client = _get_vertex_client()

            response = client.models.edit_image(
                model="imagen-3.0-capability-001",
                prompt=prompt_final,
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
                    negative_prompt=neg,
                    safety_filter_level="block_only_high",
                ),
            )
            if response.generated_images:
                print(f"[E-single] SUCCESS attempt={attempt}")
                return base64.b64encode(response.generated_images[0].image.image_bytes).decode()
            print(f"[E-single] 0 immagini (g={guidance}) — safety filter o quota")
            return None

        except Exception as e:
            print(f"[E-single] ERRORE attempt={attempt}: {type(e).__name__}: {e}")
            if attempt < 2:
                wait = 15
                print(f"[E-single] retry tra {wait}s…")
                time.sleep(wait)
            else:
                print("[E-single] fallito dopo 2 tentativi")
                return None




def _approach_E_two_stage(photo_bytes: bytes,
                           prompt_stage1: str, prompt_stage2: str,
                           guidance1: int, guidance2: int,
                           current_wall_color: str = "",
                           is_bathroom: bool = False) -> str | None:
    """
    E_WALL_FORCE v21:
    - Stage1 guidance=15: copertura opaca forzata del colore pareti originale
    - Stage2 guidance=12: più dettaglio nell'arredo
    - Protezione bagni: no finestre fantasma, piastrelle come elementi strutturali
    - Retry con backoff esponenziale
    """
    import time

    # Rinforzo prompt Stage1: solid opaque coating
    p1_suffix = (
        f" The original {current_wall_color} color is completely hidden by solid opaque "
        "structural coating. Zero bleed-through. 100% opacity. No original color visible anywhere."
        if current_wall_color
        else " Solid opaque structural coating on all walls. Zero bleed-through. 100% opacity."
    )
    if is_bathroom:
        p1_suffix += (
            " BATHROOM: DO NOT add any windows that were not in the original photo. "
            "Keep all existing tile boundaries exactly as in the original. "
            "Tiles are structural elements — only recolor or apply micro-cement finish overlay."
        )
    prompt_stage1_final = prompt_stage1 + p1_suffix

    # Rinforzo prompt Stage2: keyword ridotte + protezione bagno
    p2_base = (
        prompt_stage2.split("Realistic fabric folds")[0].strip().rstrip(",")
        + ". Photorealistic, global illumination, consistent shadows, "
        "cinematic lighting, high-end interior photography."
    )
    if is_bathroom:
        p2_base += (
            " BATHROOM RULES: No windows added. Tiles unchanged in shape and position. "
            "Preserve original ceiling light or replace only with similar overhead fixture."
        )
    prompt_stage2_final = p2_base

    # Negative prompts
    base_neg_e1 = (
        "furniture, objects, clutter, trash bags, messy cables, "
        "distorted architecture, watermark, low quality, unrealistic, "
        "original wall color, bleed-through, transparent paint, "
        "color bleeding, previous paint color showing through, dirty walls"
    )
    neg_e1 = base_neg_e1
    if current_wall_color:
        neg_e1 += f", {current_wall_color}, {current_wall_color} walls"
    if is_bathroom:
        neg_e1 += ", added windows, fake windows, new windows, exterior view"

    neg_e2 = (
        "empty room, bare walls, clutter, trash bags, dark shadows, "
        "overexposed, noise, cartoon, floating objects, "
        "inconsistent shadows, distorted architecture, watermark, "
        "yellowish tint, original wall color retained, bleed-through"
    )
    if current_wall_color:
        neg_e2 += f", {current_wall_color} on walls"
    if is_bathroom:
        neg_e2 += ", added windows, fake windows, new windows, exterior view, missing tiles"

    for attempt in range(1, 3):
        try:
            compressed = compress_image(photo_bytes, max_width=1024, quality=80)
            bath_tag = " [BAGNO]" if is_bathroom else ""
            print(f"[E Stage1{bath_tag}] attempt={attempt} {len(compressed)//1024}KB g={guidance1}")
            client = _get_vertex_client()

            r1 = client.models.edit_image(
                model="imagen-3.0-capability-001",
                prompt=prompt_stage1_final,
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
                    negative_prompt=neg_e1,
                    safety_filter_level="block_only_high",
                ),
            )

            stage1_bytes = (
                r1.generated_images[0].image.image_bytes
                if r1.generated_images
                else compress_image(photo_bytes, max_width=1024, quality=80)
            )
            print(
                f"[E Stage1] {'SUCCESS' if r1.generated_images else '0 img → fallback originale'}"
            )

            print(f"[E Stage2{bath_tag}] g={guidance2}")
            r2 = client.models.edit_image(
                model="imagen-3.0-capability-001",
                prompt=prompt_stage2_final,
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
                    negative_prompt=neg_e2,
                    safety_filter_level="block_only_high",
                ),
            )
            if r2.generated_images:
                print(f"[E Stage2] SUCCESS attempt={attempt}")
                return base64.b64encode(r2.generated_images[0].image.image_bytes).decode()
            print(f"[E Stage2] 0 immagini (g={guidance2}) — safety filter o quota")
            return None

        except Exception as e:
            print(f"[E two-stage] ERRORE attempt={attempt}: {type(e).__name__}: {e}")
            if attempt < 2:
                wait = 10 * attempt
                print(f"[E] retry tra {wait}s…")
                time.sleep(wait)
            else:
                print("[E] fallito dopo 2 tentativi")
                return None
