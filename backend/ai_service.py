"""
AI Service v18 — Architetto Digitale

Nuovi moduli:
  1. PhotoValidator        — valida le foto prima del processing (luminosità, inquadratura)
  2. GlobalSpatialAnchor  — mappa adiacenze tra stanze e inietta layout_constraints
                            tra stanze contigue per evitare allucinazioni spaziali
  3. UnifiedStyleDNA      — profilo stilistico globale con palette ristretta per stile
                            e material anchoring tra tutte le stanze
  4. KitchenBrutalistProtocol — prompts specifici per cucine in E_WALL_FORCE:
                            Stage1: strip ante beige, ridipingi tutto in Charcoal
                            Stage2: refacing industriale ante + layering cucina

Invariati: D_FULL_SMART (g1=8, g2=10) · guidance E hardcodati · multi-foto · cache
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
GUIDANCE = {
    "C4_FULL":          25,
    "C5_SMART_FULL":    22,
    "D_STAGE1_CLEAN":    8,
    "D_STAGE2_STAGE":   10,
    "E_STAGE1_WALL":    14,
    "E_STAGE2_STAGE":   10,
}

# ── Palette autorizzate per stile ────────────────────────────────────────────
STYLE_PALETTES = {
    "Scandinavian": {
        "wall_colors":        "optical white, warm white, pearl grey, light greige. "
                              "FORBIDDEN: orange, terracotta, red, electric blue, yellow.",
        "wall_finish_choice": "matte warm white",
        "accent":             "light oak wood, matte black metal, natural linen, soft grey textiles",
        "wood_material":      "Light Oak",
        "metal_material":     "Matte Black",
        "kitchen_cabinet_color": "matte white",
        "kitchen_counter":       "light oak butcher block or white quartz",
    },
    "Industrial": {
        "wall_colors":        "concrete grey, dark charcoal, warm brick white, off-white. "
                              "FORBIDDEN: pink, pastel, bright orange, yellow.",
        "wall_finish_choice": "matte concrete grey",
        "accent":             "dark steel, reclaimed wood, exposed brick, Edison bulbs",
        "wood_material":      "Reclaimed Dark Wood",
        "metal_material":     "Dark Steel",
        "kitchen_cabinet_color": "matte charcoal grey",
        "kitchen_counter":       "raw concrete or dark steel industrial countertop",
    },
    "Japandi": {
        "wall_colors":        "wabi-sabi white, warm beige, pale sage, soft clay. "
                              "FORBIDDEN: saturated colors, neon, orange.",
        "wall_finish_choice": "matte warm beige",
        "accent":             "natural bamboo, stone, warm linen, matte ceramics",
        "wood_material":      "Blonde Bamboo",
        "metal_material":     "Brushed Brass",
        "kitchen_cabinet_color": "warm linen white",
        "kitchen_counter":       "natural stone or light bamboo",
    },
}

def _get_palette(style: str) -> dict:
    key = style.strip().title()
    if "Scandi" in key:  key = "Scandinavian"
    elif "Japandi" in key or "Japan" in key: key = "Japandi"
    elif "Industri" in key: key = "Industrial"
    return STYLE_PALETTES.get(key, {
        "wall_colors":        f"neutral tones coherent with {style}. Avoid saturated colors.",
        "wall_finish_choice": "matte neutral",
        "accent":             f"materials coherent with {style}",
        "wood_material":      "Light Natural Wood",
        "metal_material":     "Matte Black",
        "kitchen_cabinet_color": "matte white",
        "kitchen_counter":       "light natural stone",
    })

_vertex_client = None

def _get_vertex_client():
    global _vertex_client
    if _vertex_client is None:
        print(f"[Imagen] init client project={PROJECT_ID} location={LOCATION}")
        _vertex_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    return _vertex_client

_gemini_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="gemini")
_imagen_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="imagen")

_analysis_cache: dict[str, dict] = {}


# ═══════════════════════════════════════════════════════════════════════════════
# MODULO 1 — PhotoValidator
# ═══════════════════════════════════════════════════════════════════════════════

async def validate_input_photos(photos: list) -> dict:
    """
    Valida le foto prima del processing principale.
    Ritorna: {
        "valid": [True/False per ogni foto],
        "issues": ["descrizione problema o 'ok'" per ogni foto],
        "warnings": [lista warning globali],
        "all_valid": bool
    }
    Chiamata in main.py dopo il caricamento e prima di avviare il job.
    """
    if not photos:
        return {"valid": [], "issues": [], "warnings": ["Nessuna foto"], "all_valid": False}

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        _gemini_executor, _validate_photos_sync, photos
    )
    return result


def _validate_photos_sync(photos: list) -> dict:
    """Chiama Gemini per analizzare qualità e adeguatezza di ogni foto."""
    n = len(photos)

    parts = []
    for i, p in enumerate(photos):
        parts.append({"text": f"[FOTO {i}]"})
        parts.append({
            "inline_data": {
                "mime_type": p["content_type"],
                "data": base64.b64encode(p["content"]).decode()
            }
        })

    validator_prompt = (
        f"Analizza queste {n} foto per valutarne l'idoneità a un servizio di AI home staging.\n\n"
        f"Per ogni foto valuta:\n"
        f"1. LUMINOSITÀ: la foto è sufficientemente illuminata? (scarta se < 30% luminosità)\n"
        f"2. INQUADRATURA: si vede abbastanza della stanza? (scarta se si vede solo un mobile)\n"
        f"3. OSTRUZIONE: porte, persone o oggetti coprono più del 50% della visuale?\n"
        f"4. QUALITÀ: è sfocata, rumorosa, o troppo compressa?\n"
        f"5. TIPO STANZA: che stanza è (soggiorno, cucina, camera, bagno, corridoio)?\n\n"
        f"Restituisci SOLO questo JSON:\n"
        "{{\n"
        f"  \"photos\": [\n"
        f"    /* RIPETI {n} VOLTE, una per foto */\n"
        "    {{\n"
        "      \"index\": 0,\n"
        "      \"valid\": true,\n"
        "      \"room_type\": \"soggiorno\",\n"
        "      \"issue\": \"ok oppure descrizione problema specifico\",\n"
        "      \"suggestion\": \"consiglio pratico se non valida (es. Accendi le luci, Usa grandangolo)\"\n"
        "    }}\n"
        "  ],\n"
        "  \"global_warnings\": [\"warning globale se presente\"],\n"
        "  \"apartment_layout_hint\": \"descrizione breve di come le stanze sembrano collegate\"\n"
        "}}"
    )

    parts.append({"text": validator_prompt})

    payload = {
        "system_instruction": {"parts": [{
            "text": "Sei un esperto di fotografia immobiliare. Analizza criticamente le foto. "
                    "Rispondi ESCLUSIVAMENTE con JSON valido, senza markdown."
        }]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 4096,
            "responseMimeType": "application/json"
        }
    }

    try:
        response = httpx.post(GEMINI_URL, json=payload, timeout=60.0,
                              headers={"Content-Type": "application/json"})
        response.raise_for_status()
        data = response.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        parsed = _extract_json(text)

        photos_data = parsed.get("photos", [])
        valid_list  = [p.get("valid", True) for p in photos_data]
        issues_list = [p.get("issue", "ok") for p in photos_data]

        print(f"[PhotoValidator] {sum(valid_list)}/{n} foto valide")
        return {
            "valid":           valid_list,
            "issues":          issues_list,
            "room_types":      [p.get("room_type", "unknown") for p in photos_data],
            "suggestions":     [p.get("suggestion", "") for p in photos_data],
            "warnings":        parsed.get("global_warnings", []),
            "layout_hint":     parsed.get("apartment_layout_hint", ""),
            "all_valid":       all(valid_list),
        }
    except Exception as e:
        print(f"[PhotoValidator] ERRORE: {e} — procedo comunque")
        return {
            "valid":       [True] * n,
            "issues":      ["validation skipped"] * n,
            "room_types":  ["unknown"] * n,
            "suggestions": [""] * n,
            "warnings":    [],
            "layout_hint": "",
            "all_valid":   True,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# MODULO 2+3+4 — Gemini Analysis
# ═══════════════════════════════════════════════════════════════════════════════

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
    palette     = _get_palette(style)

    alloc = {
        "arredi":        int(budget * 0.40),
        "tinteggiatura": int(budget * 0.25),
        "materiali":     int(budget * 0.20),
        "montaggio":     int(budget * 0.15),
    }

    system_instruction = (
        "Sei un Architetto Digitale specializzato in home staging AI. "
        "Devi mantenere coerenza spaziale e stilistica tra tutte le foto di un appartamento. "
        "REGOLA ASSOLUTA — PALETTE: "
        f"I target_wall_color devono essere ESCLUSIVAMENTE: {palette['wall_colors']} "
        "REGOLA MATERIALI: In ogni prompt_stage2 di ogni stanza usa SEMPRE gli stessi materiali: "
        f"legno='{palette['wood_material']}', metallo='{palette['metal_material']}'. "
        "REGOLA CUCINA: Se una stanza è una cucina, usa il KitchenBrutalistProtocol: "
        f"ridipingi ante in '{palette['kitchen_cabinet_color']}', "
        f"il piano diventa '{palette['kitchen_counter']}'. "
        "REGOLA SPAZIALE: Se due stanze confinano, i layout_constraints devono essere "
        "rispettati in entrambe le stanze per evitare allucinazioni spaziali. "
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

        # ── STEP 0: GlobalSpatialAnchor ──────────────────────────────────────
        f"═══ STEP 0 — PIANTINA SPAZIALE (GlobalSpatialAnchor) ═══\n"
        f"Prima di qualsiasi altra analisi, crea una mappa delle adiacenze tra stanze:\n"
        f"Per ogni coppia di foto che mostra stanze contigue (es. soggiorno vede il frigo "
        f"della cucina, o una porta aperta rivela un'altra stanza), definisci:\n"
        f"  adjacent_rooms: lista di coppie [indice_foto_A, indice_foto_B]\n"
        f"  shared_boundary: descrizione del confine (es. 'porta aperta lato est')\n"
        f"  visible_from_A_in_B: elementi della stanza B visibili dalla stanza A\n"
        f"  layout_note: vincolo spaziale critico (es. 'frigorifero angolo nord-est visibile dal soggiorno')\n\n"
        f"Questi dati alimentano i layout_constraints iniettati nei prompt di ogni stanza.\n\n"

        # ── STEP 1: UnifiedStyleDNA ──────────────────────────────────────────
        f"═══ STEP 1 — DNA STILISTICO + ANALISI VISIVA ═══\n"
        f"Genera UN SOLO global_style_profile per tutto l'appartamento:\n"
        f"  wall_color_choice: UN SOLO colore tra: {palette['wall_colors']}\n"
        f"  wall_finish_choice: es. 'matte concrete grey'\n"
        f"  wood_material: '{palette['wood_material']}' — FISSO per tutte le stanze\n"
        f"  metal_material: '{palette['metal_material']}' — FISSO per tutte le stanze\n"
        f"  kitchen_cabinet_color: '{palette['kitchen_cabinet_color']}'\n"
        f"  kitchen_counter: '{palette['kitchen_counter']}'\n\n"
        f"POI per ogni foto:\n"
        f"  room_type: 'cucina' | 'soggiorno' | 'camera' | 'bagno' | 'altro'\n"
        f"  is_kitchen: true se room_type == 'cucina'\n"
        f"  detected_elements: lista oggetti fisicamente visibili\n"
        f"  layout_constraints: vincoli spaziali da rispettare (da STEP 0 adiacenze)\n"
        f"    Es: 'Frigorifero angolo nord-est NON va spostato — visibile dal soggiorno'\n"
        f"    Es: 'Non aggiungere divano lato est — interferirebbe con ingresso cucina'\n"
        f"  light_analysis: sorgente luce, materiali esistenti, direzione ombre\n"
        f"  current_wall_color: colore ESATTO pareti attuali\n"
        f"  target_wall_color: da global_style_profile.wall_color_choice (STESSO per tutte)\n"
        f"  target_wall_finish: da global_style_profile.wall_finish_choice\n\n"

        # ── STEP 2: Varianti ─────────────────────────────────────────────────
        f"═══ STEP 2 — 2 VARIANTI TWO-STAGE PER OGNI STANZA ═══\n\n"

        f"D_FULL_SMART (Stage1 g=8, Stage2 g=10):\n"
        f"  Stage1: rimuovi mobili+clutter, applica target_wall_finish/color, rispetta layout_constraints.\n"
        f"  Stage2: aggiungi layering. MATERIAL ANCHORING: "
        f"legno={palette['wood_material']}, metallo={palette['metal_material']}. "
        f"Rispetta layout_constraints (non mettere mobili dove ci sono vincoli spaziali).\n\n"

        f"E_WALL_FORCE (Stage1 g=14, Stage2 g=10):\n"
        f"CUCINE — KitchenBrutalistProtocol attivo se is_kitchen=true:\n"
        f"  Stage1: 'Complete renovation. Strip all cabinet doors from their existing color. "
        f"Repaint all cabinet fronts in {palette['kitchen_cabinet_color']} matte finish. "
        f"Remove red curtains or any bright-colored textiles. "
        f"Paint all walls {palette['wall_finish_choice']} {palette['wall_colors'].split(',')[0]}. "
        f"Zero bleed-through of original colors. Photorealistic, 8k.'\n"
        f"  Stage2: 'Working on the newly painted {palette['kitchen_cabinet_color']} cabinets: "
        f"add {palette['kitchen_counter']} countertop. "
        f"Add Edison bulb pendant lamp in {palette['metal_material']}. "
        f"Add potted herbs in ceramic pots on windowsill. "
        f"Add dark runner rug on floor. Remove red curtains, replace with sheer linen. "
        f"Realistic surfaces, consistent shadows, global illumination, 8k.'\n\n"
        f"STANZE NON-CUCINA — E_WALL_FORCE standard:\n"
        f"  Stage1: nomina current_wall_color da eliminare, nomina target_wall_finish/color, "
        f"'Solid opaque paint, zero bleed-through. Empty room. Photorealistic, 8k.'\n"
        f"  Stage2: inizia con 'Working on the newly painted [color] walls from Stage 1:' "
        f"poi aggiungi layering. Rispetta layout_constraints.\n\n"

        f"Restituisci SOLO questo JSON ({n} oggetti in 'stanze'):\n\n"
        "{{\n"
        "  \"spatial_map\": {{\n"
        "    \"adjacencies\": [\n"
        "      {{\n"
        "        \"rooms\": [0, 1],\n"
        "        \"shared_boundary\": \"descrizione confine\",\n"
        "        \"visible_from_0_in_1\": \"cosa si vede\",\n"
        "        \"layout_note\": \"vincolo spaziale critico\"\n"
        "      }}\n"
        "    ]\n"
        "  }},\n"
        "  \"global_style_profile\": {{\n"
        f"    \"style\": \"{style}\",\n"
        f"    \"wall_color_choice\": \"UN colore da: {palette['wall_colors'].split('.')[0]}\",\n"
        "    \"wall_finish_choice\": \"es. matte concrete grey\",\n"
        f"    \"wood_material\": \"{palette['wood_material']}\",\n"
        f"    \"metal_material\": \"{palette['metal_material']}\",\n"
        f"    \"kitchen_cabinet_color\": \"{palette['kitchen_cabinet_color']}\",\n"
        f"    \"kitchen_counter\": \"{palette['kitchen_counter']}\"\n"
        "  }},\n"
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
        "      \"room_type\": \"soggiorno\",\n"
        "      \"is_kitchen\": false,\n"
        "      \"detected_elements\": [\"elemento visibile\"],\n"
        "      \"layout_constraints\": [\n"
        "        \"vincolo spaziale da rispettare (es. non spostare frigo — visibile dal soggiorno)\"\n"
        "      ],\n"
        "      \"current_wall_color\": \"colore esatto pareti attuali\",\n"
        "      \"target_wall_color\": \"DA global_style_profile.wall_color_choice — STESSO per tutte\",\n"
        "      \"target_wall_finish\": \"DA global_style_profile.wall_finish_choice\",\n"
        "      \"light_analysis\": {{\n"
        "        \"light_source\": \"es. finestra sinistra\",\n"
        "        \"existing_materials\": \"es. parquet chiaro, pareti gialle\",\n"
        "        \"shadow_direction\": \"es. ombre verso destra\"\n"
        "      }},\n"
        "      \"stato_attuale\": \"descrizione dettagliata\",\n"
        "      \"interventi\": [\n"
        "        {{\n"
        "          \"titolo\": \"nome intervento\",\n"
        f"          \"dettaglio\": \"prodotto specifico a {location}\",\n"
        "          \"costo_min\": 50, \"costo_max\": 120,\n"
        "          \"priorita\": \"alta\",\n"
        f"          \"dove_comprare\": \"negozio {style}\"\n"
        "        }}\n"
        "      ],\n"
        "      \"costo_totale_stanza\": 350,\n"
        "      \"esperimenti_staged\": [\n"
        "        {{\n"
        "          \"logic_id\": \"D_FULL_SMART\",\n"
        "          \"guidance_scale_stage1\": 8,\n"
        "          \"guidance_scale_stage2\": 10,\n"
        "          \"prompt_stage1\": \"[COMPILA: empty room, rimuovi tutto, applica target_wall_finish+color, rispetta layout_constraints. Photorealistic, 8k.]\",\n"
        f"          \"prompt_stage2\": \"[COMPILA: stanza vuota con nuove pareti. Legno {palette['wood_material']}, metallo {palette['metal_material']}. Layering: tappeto, cuscini, pianta, mensola Edison, stampe, lampada. Rispetta layout_constraints. Consistent shadows, global illumination, ambient occlusion, f/8, HDR. Real estate photo, 24mm, cinematic warm light, 8k.]\",\n"
        "          \"interventi_lista\": [\n"
        f"            {{\"voce\": \"Finitura pareti [{palette['wall_colors'].split(',')[0]}]\", \"costo\": 400}},\n"
        f"            {{\"voce\": \"Tappeto {palette['accent'].split(',')[0]}\", \"costo\": 120}},\n"
        "            {{\"voce\": \"Cuscini 5 pz\", \"costo\": 80}},\n"
        f"            {{\"voce\": \"Mensola {palette['metal_material']} + Edison\", \"costo\": 90}},\n"
        "            {{\"voce\": \"2 stampe + cornici\", \"costo\": 40}},\n"
        "            {{\"voce\": \"Pianta + vaso terracotta\", \"costo\": 45}},\n"
        f"            {{\"voce\": \"Lampada stelo {palette['metal_material']}\", \"costo\": 60}}\n"
        "          ],\n"
        "          \"costo_simulato\": 835\n"
        "        }},\n"
        "        {{\n"
        "          \"logic_id\": \"E_WALL_FORCE\",\n"
        "          \"guidance_scale_stage1\": 14,\n"
        "          \"guidance_scale_stage2\": 10,\n"
        "          \"is_kitchen\": false,\n"
        "          \"current_wall_color\": \"[colore esatto pareti attuali]\",\n"
        "          \"target_wall_color\": \"[DA global_style_profile.wall_color_choice]\",\n"
        "          \"target_wall_finish\": \"[DA global_style_profile.wall_finish_choice]\",\n"
        "          \"prompt_stage1\": \"[COMPILA seguendo KitchenBrutalistProtocol se cucina, altrimenti E_WALL_FORCE standard. Include layout_constraints.]\",\n"
        "          \"prompt_stage2\": \"[COMPILA: INIZIA con 'Working on the newly painted [color] walls from Stage 1:'. Aggiungi layering. Rispetta layout_constraints. Consistent shadows, global illumination, ambient occlusion, f/8, HDR. Real estate photo, 24mm, cinematic warm light, 8k.]\",\n"
        "          \"interventi_lista\": [\n"
        f"            {{\"voce\": \"Tinteggiatura {palette['wall_colors'].split(',')[0]}\", \"costo\": 400}},\n"
        f"            {{\"voce\": \"Tappeto {style} design\", \"costo\": 120}},\n"
        "            {{\"voce\": \"Cuscini 5 pz Zara Home\", \"costo\": 80}},\n"
        f"            {{\"voce\": \"Mensola {palette['metal_material']} + Edison\", \"costo\": 90}},\n"
        "            {{\"voce\": \"2 stampe + cornici\", \"costo\": 40}},\n"
        "            {{\"voce\": \"Pianta + vaso terracotta\", \"costo\": 45}},\n"
        f"            {{\"voce\": \"Lampada stelo {palette['metal_material']}\", \"costo\": 60}}\n"
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
        "    {{\"categoria\": \"Layering e Decor\",\n"
        f"     \"articoli\": [\"item specifico {style}\"],\n"
        "     \"budget_stimato\": 0,\n"
        f"     \"negozi_consigliati\": \"IKEA, H&M Home, Zara Home, Westwing — {location}\"}}\n"
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

    n_stanze = len(result.get("stanze", []))
    print(f"[Gemini] stanze={n_stanze}/{n} | "
          f"adjacencies={len(result.get('spatial_map', {}).get('adjacencies', []))}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# STAGED PHOTOS
# ═══════════════════════════════════════════════════════════════════════════════

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
                p1 = esp.get("prompt_stage1", "")
                p2 = esp.get("prompt_stage2", "")
                if not p1 or not p2 or not photo_bytes:
                    continue
                all_futures.append(("D_FULL_SMART", i,
                    loop.run_in_executor(
                        _imagen_executor, _approach_D_two_stage,
                        photo_bytes, p1, p2,
                        GUIDANCE["D_STAGE1_CLEAN"], GUIDANCE["D_STAGE2_STAGE"]
                    )
                ))

            elif logic_id == "E_WALL_FORCE":
                p1  = esp.get("prompt_stage1", "")
                p2  = esp.get("prompt_stage2", "")
                cwc = esp.get("current_wall_color", "")
                is_kitchen = esp.get("is_kitchen", False) or room.get("is_kitchen", False)
                if not p1 or not p2 or not photo_bytes:
                    continue
                all_futures.append(("E_WALL_FORCE", i,
                    loop.run_in_executor(
                        _imagen_executor, _approach_E_two_stage,
                        photo_bytes, p1, p2,
                        GUIDANCE["E_STAGE1_WALL"], GUIDANCE["E_STAGE2_STAGE"],
                        cwc, is_kitchen
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


# ── Core: singola (C4, C5) ────────────────────────────────────────────────────

def _approach_single(photo_bytes: bytes, prompt: str,
                     guidance: int, label: str) -> str | None:
    try:
        compressed = compress_image(photo_bytes, max_width=1024, quality=80)
        print(f"[{label}] {len(compressed)//1024}KB  guidance={guidance}")
        client = _get_vertex_client()

        negative_base = (
            "distorted architecture, blurry textures, changing window positions, "
            "moving doors, wrong room proportions, deformed walls, different ceiling height, "
            "watermark, low quality, unrealistic, flat lighting, cartoon, illustration, "
            "floating objects, sticker effect, inconsistent shadows"
        )
        negative = (
            negative_base + ", clutter, old bulky furniture, yellowish tint, "
            "original wall color retained, same walls as original"
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
            print(f"[{label}] SUCCESS")
            return base64.b64encode(response.generated_images[0].image.image_bytes).decode()
        print(f"[{label}] 0 immagini (guidance={guidance})")
        return None
    except Exception as e:
        print(f"[{label}] ERRORE: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return None


# ── Core: D two-stage (INVARIATA) ────────────────────────────────────────────

def _approach_D_two_stage(photo_bytes: bytes,
                           prompt_stage1: str, prompt_stage2: str,
                           guidance1: int, guidance2: int) -> str | None:
    try:
        compressed = compress_image(photo_bytes, max_width=1024, quality=80)
        print(f"[D Stage1] {len(compressed)//1024}KB  guidance={guidance1}")
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
            r1.generated_images[0].image.image_bytes if r1.generated_images
            else compress_image(photo_bytes, max_width=1024, quality=80)
        )
        print(f"[D Stage1] {'SUCCESS' if r1.generated_images else '0 immagini → fallback'}")

        print(f"[D Stage2] guidance={guidance2}")
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
                negative_prompt=(
                    "empty room, bare walls, clutter, trash bags, messy cables, "
                    "dark shadows, bad exposure, harsh lighting, noise, grainy, "
                    "bad framing, overcrowded, cartoon style, flat lighting, floating objects, "
                    "sticker effect, inconsistent shadows, distorted architecture, watermark"
                ),
                safety_filter_level="block_only_high",
            ),
        )
        if r2.generated_images:
            print("[D Stage2] SUCCESS")
            return base64.b64encode(r2.generated_images[0].image.image_bytes).decode()
        print(f"[D Stage2] 0 immagini (guidance={guidance2})")
        return None
    except Exception as e:
        print(f"[D two-stage] ERRORE: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return None


# ── Core: E two-stage (WALL FORCE + KitchenBrutalistProtocol) ────────────────

def _approach_E_two_stage(photo_bytes: bytes,
                           prompt_stage1: str, prompt_stage2: str,
                           guidance1: int, guidance2: int,
                           current_wall_color: str = "",
                           is_kitchen: bool = False) -> str | None:
    """
    Stage1 guidance=14 — aggressivo sul cambio colore pareti.
    Per cucine (is_kitchen=True): negative prompt include colori ante originali.
    Stage2: inizia con conferma risultato Stage1.
    """
    try:
        compressed = compress_image(photo_bytes, max_width=1024, quality=80)
        label = "E[kitchen]" if is_kitchen else "E[room]"
        print(f"[{label} Stage1] {len(compressed)//1024}KB  guidance={guidance1}")
        client = _get_vertex_client()

        # Negative Stage1: generico + colore specifico + extra per cucine
        neg_e1 = (
            "furniture, objects, clutter, trash bags, messy cables, "
            "distorted architecture, watermark, low quality, unrealistic, "
            "original wall color, bleed-through, transparent paint, "
            "previous paint color showing through, dirty walls"
        )
        if current_wall_color:
            neg_e1 += f", {current_wall_color}, {current_wall_color} tinted walls"
        if is_kitchen:
            neg_e1 += (
                ", beige cabinets, cream cabinets, old cabinet color retained, "
                "wooden cabinet texture visible, original kitchen color"
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
                negative_prompt=neg_e1,
                safety_filter_level="block_only_high",
            ),
        )
        stage1_bytes = (
            r1.generated_images[0].image.image_bytes if r1.generated_images
            else compress_image(photo_bytes, max_width=1024, quality=80)
        )
        print(f"[{label} Stage1] {'SUCCESS' if r1.generated_images else '0 immagini → fallback'}")

        # Negative Stage2
        neg_e2 = (
            "empty room, bare walls, clutter, trash bags, messy cables, "
            "dark shadows, bad exposure, overexposed, harsh lighting, noise, grainy, "
            "bad framing, overcrowded, cartoon style, flat lighting, floating objects, "
            "sticker effect, inconsistent shadows, distorted architecture, watermark, "
            "yellowish tint, original wall color retained, bleed-through"
        )
        if current_wall_color:
            neg_e2 += f", {current_wall_color} on walls"
        if is_kitchen:
            neg_e2 += ", beige cabinets, old cabinet color, original kitchen appearance"

        print(f"[{label} Stage2] guidance={guidance2}")
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
                negative_prompt=neg_e2,
                safety_filter_level="block_only_high",
            ),
        )
        if r2.generated_images:
            print(f"[{label} Stage2] SUCCESS")
            return base64.b64encode(r2.generated_images[0].image.image_bytes).decode()
        print(f"[{label} Stage2] 0 immagini (guidance={guidance2})")
        return None
    except Exception as e:
        print(f"[E two-stage] ERRORE: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return None
