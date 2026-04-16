"""
AI Service v24 — Master Stylist Engine

Unica variante: E_WALL_FORCE two-stage. Zero librerie extra.
Tutto il delta qualitativo è nel prompt engineering.

Novità v24:
  SCENIC_CONFIG  — keyword fotografiche professionali iniettate in ogni chiamata Imagen.
  get_style_dna  — database stili: Boho/Scandinavo/Minimalista/Japandi/Industrial/ecc.
                   Ogni stile → palette, materiali, arredi IKEA, piante, tessili specifici.
  Stage 1 v24    — decluttering esplicito per categoria (cavi, oggetti personali,
                   mobili brutti identificati da Gemini), force-paint aggressivo.
  Stage 2 v24    — Catchy Airbnb Layer fisso: cuscini stratificati, pianta statement,
                   props scenografici (riviste design, vasi, quadri).
                   Budget-driven: reface se budget basso, replace IKEA se budget alto.
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
    "E_STAGE1_CANVAS":  16,   # Force-paint: copre colori originali al 100%
    "E_STAGE2_STAGING": 12,   # Catchy layering: fotorealismo + arredo
}

# ── SCENIC CONFIG — keyword fotografiche iniettate in ogni chiamata Imagen ────
SCENIC_CONFIG = (
    "24mm architectural wide-angle lens, f/11 aperture, "
    "cinematic warm light 2700K, HDR perfectly balanced exposure, "
    "soft bounced light from ceiling, global illumination with IES profiles, "
    "architectural digest photography style, 8k resolution, ultra sharp"
)

# ── STYLE DNA — database stili: palette + materiali + arredi + piante ─────────
_STYLE_DB: dict[str, dict] = {
    "scandinavo": {
        "palette":    "warm white walls, light ash wood tones, soft grey accents",
        "materials":  "light oak, white-painted wood, linen, wool, natural stone",
        "furniture":  "IKEA MALM bed white, IKEA LISABO table ash, IKEA ADDE chairs white",
        "textiles":   "chunky knit throw in cream, linen cushions in dusty blue and oat",
        "plants":     "Monstera deliciosa in light grey ceramic pot",
        "props":      "white ceramic vases, minimalist wooden clock, simple framed prints",
        "lighting":   "IKEA REGOLIT pendant lamp, warm Edison bulb 2700K",
    },
    "minimalista": {
        "palette":    "pure white walls, warm grey accents, black steel details",
        "materials":  "concrete, white-painted plaster, polished stone, black metal",
        "furniture":  "IKEA MALM dresser black-brown, IKEA LACK side table white, low-profile sofa",
        "textiles":   "white linen bedding, single grey throw, no pattern cushions",
        "plants":     "tall Sansevieria in matte black pot",
        "props":      "single large art print black frame, architectural coffee table book",
        "lighting":   "IKEA HEKTAR floor lamp black, directional spotlight 2700K",
    },
    "japandi": {
        "palette":    "warm greige walls, natural wood, muted earth tones, wabi-sabi accents",
        "materials":  "bamboo, rattan, rice paper, unfinished wood, linen, ceramic",
        "furniture":  "low platform bed natural wood, rattan side table, floor cushion zabuton",
        "textiles":   "linen duvet in oat, woven cotton blanket in terracotta, no pattern",
        "plants":     "Bonsai tree or tall Bamboo in terracotta pot",
        "props":      "ceramic tea set, single dried pampas stem, zen rock garden tray",
        "lighting":   "paper lantern pendant MUJI style, warm 2200K candle light",
    },
    "boho": {
        "palette":    "warm white walls, terracotta accents, sage green, mustard yellow",
        "materials":  "jute, rattan, macramé, distressed wood, ethnic textiles",
        "furniture":  "rattan armchair, low wooden coffee table, IKEA LOHALS jute rug",
        "textiles":   "mix of patterned cushions (ethnic, ikat, velvet), fringed throw in mustard",
        "plants":     "Strelitzia nicolai or large Fiddle-leaf fig in terracotta pot",
        "props":      "macramé wall hanging, vintage ceramic vases, ethnic woven basket",
        "lighting":   "rattan pendant lamp, string fairy lights warm 2200K",
    },
    "industrial": {
        "palette":    "exposed concrete grey, dark charcoal walls, warm amber accents",
        "materials":  "raw steel, reclaimed wood, exposed brick, leather, concrete",
        "furniture":  "IKEA FJÄLLBO shelving black steel, leather sofa dark brown, metal bar stools",
        "textiles":   "dark grey wool throw, leather cushions, no pattern",
        "plants":     "large Cactus or Rubber plant in raw concrete pot",
        "props":      "vintage Edison bulb pendant, industrial clock, reclaimed wood shelf",
        "lighting":   "IKEA HEKTAR pendant black, exposed Edison bulb 2200K warm amber",
    },
    "mediterraneo": {
        "palette":    "warm white walls, cobalt blue accents, terracotta floors, ochre",
        "materials":  "whitewashed plaster, terracotta tiles, wrought iron, ceramic",
        "furniture":  "whitewashed wood furniture, wrought iron bed frame, ceramic tile table",
        "textiles":   "white cotton bedding, blue and white striped cushions, linen curtains",
        "plants":     "Olive tree in large terracotta pot, Lavender, Bougainvillea",
        "props":      "blue ceramic vases, woven baskets, hand-painted ceramic plates on wall",
        "lighting":   "wrought iron pendant, warm 2700K candlelight lanterns",
    },
}

def get_style_dna(style: str) -> dict:
    """
    Restituisce il dizionario DNA per lo stile richiesto.
    Fuzzy match: cerca la keyword dello stile nel database.
    Fallback: minimalista.
    """
    style_lower = style.lower()
    # Exact match
    if style_lower in _STYLE_DB:
        return _STYLE_DB[style_lower]
    # Fuzzy: cerca se una chiave è contenuta nello stile scritto dall'utente
    for key in _STYLE_DB:
        if key in style_lower or style_lower in key:
            return _STYLE_DB[key]
    # Fallback
    return _STYLE_DB["minimalista"]



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
        "Sei un Interior Designer e Fotografo Real Estate senior. "
        "Il tuo obiettivo è massimizzare il CTR su Airbnb/Booking.\n\n"

        "STEP 1 — ANALISI SPAZIALE per ogni foto:\n"
        "  a) Identifica la sorgente luminosa principale e la direzione delle ombre.\n"
        "  b) Mappa i volumi: se un mobile blocca un elemento fisso (es. letto davanti "
        "all'armadio), pianifica il riposizionamento.\n"
        "  c) Identifica il COLORE ESATTO delle pareti attuali (es. 'giallo paglierino opaco') "
        "— critico per il Force-Paint dello Stage 1.\n"
        "  d) Identifica se è un bagno (is_bathroom=true): piastrelle = struttura, mai rimuoverle.\n\n"

        "STEP 2 — VALUTAZIONE ESTETICA E BUDGET:\n"
        "  Per ogni mobile/elemento visibile, decidi una budget_action:\n"
        "  • 'remove'  → mobile brutto o ingombrante, budget basso: eliminalo. "
        "Meglio vuoto e luminoso che pieno di obsoleto.\n"
        "  • 'replace' → budget sufficiente: rimpiazza con modello IKEA specifico.\n"
        "  • 'reface'  → cucine/bagni: cambia colore ante o applica micro-cement. "
        "Non rimuovere mai frigo, lavandini, docce sotto €5000 budget stanza.\n"
        "  • 'keep'    → elemento neutro o di qualità: mantieni.\n\n"

        "STEP 3 — GENERA mandatory_visual_keywords:\n"
        "  Lista di keyword Imagen che DERIVANO DIRETTAMENTE dagli interventi del preventivo. "
        "Se nel preventivo c'è 'Pittura grigio perla', la keyword DEVE essere "
        "'matte warm grey walls'. Se c'è 'Divano IKEA FRIHETEN grigio chiaro', la keyword "
        "DEVE essere 'light grey fabric sofa IKEA style'. "
        "REGOLA: ogni voce di costo ha una keyword visiva corrispondente.\n\n"

        "STEP 4 — COSTRUISCI I PROMPT:\n"
        "  Stage 1 (The Canvas): solo pareti + pavimento + rimozione mobili. "
        "Usa SEMPRE 'solid opaque [color] matte paint, zero bleed-through, 100% opacity'.\n"
        "  Stage 2 (The Staging): aggiungi SOLO arredi da mandatory_visual_keywords. "
        "Chiudi SEMPRE con: 'professional real estate photography, 24mm wide angle, "
        "HDR, perfectly balanced exposure, architectural digest style'.\n\n"

        "Rispondi ESCLUSIVAMENTE con un oggetto JSON valido, senza markdown né backtick."
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
        "      \"current_wall_color\": \"es. giallo paglierino opaco\",\n"
        "      \"force_paint_color\": \"es. warm grey matte\",\n"
        "      \"force_paint_finish\": \"es. matte|satin|micro-cement\",\n"
        "      \"floor_description\": \"es. light oak parquet\",\n"
        "      \"light_source\": \"es. window left, diffuse daylight\",\n"
        "      \"shadow_direction\": \"es. shadows toward right\",\n"
        "      \"spatial_map\": \"es. wardrobe blocks window — move bed to right wall\",\n"
        "      \"detected_elements\": [\"elemento visibile 1\", \"elemento visibile 2\"],\n"
        "      \"mandatory_visual_keywords\": [\n"
        "        \"solid opaque warm grey matte paint on all walls\",\n"
        "        \"light grey fabric sofa IKEA style\",\n"
        "        \"Monstera plant in terracotta pot\",\n"
        "        \"black metal floor lamp warm 2700K light\",\n"
        "        \"minimalist framed print above sofa\"\n"
        "      ],\n"
        "      \"detected_elements\": [\"elemento visibile\"],\n"
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
        "          \"budget_action\": \"remove|replace|reface|keep\",\n"
        f"          \"dove_comprare\": \"negozio coerente con {style}\"\n"
        "        }}\n"
        "      ],\n"
        "      \"costo_totale_stanza\": 350,\n"
        "      \"esperimenti_staged\": [\n"
        "        {{\n"
        "          \"logic_id\": \"E_WALL_FORCE\",\n"
        "          \"prompt_stage1\": \"Empty room. The original [current_wall_color] walls are now covered with solid opaque [force_paint_finish] [force_paint_color] coating. Zero bleed-through. 100% opacity. No original color visible. Floor is [floor_description], clean. No furniture. Light from [light_source]. Consistent lighting, photorealistic, 8k.\",\n"
        "          \"prompt_stage2\": \"[mandatory_visual_keywords joined as scene description]. [spatial_map arrangement]. Layering: textured cushions mix (linen, velvet, cotton), statement plant (Monstera or Strelitzia) in ceramic pot, warm 2700K floor lamp, minimalist framed art print. professional real estate photography, 24mm wide angle, HDR, perfectly balanced exposure, architectural digest style.\",\n"
        "          \"interventi_lista\": [\n"
        "            {{\"voce\": \"Pittura pareti [force_paint_color]\", \"costo\": 400}},\n"
        "            {{\"voce\": \"Arredo principale\", \"costo\": 500}},\n"
        "            {{\"voce\": \"Tessili e decor\", \"costo\": 200}}\n"
        "          ],\n"
        "          \"costo_simulato\": 1100\n"
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

    import time

    for attempt in range(1, 4):  # max 3 tentativi
        try:
            response = httpx.post(GEMINI_URL, json=payload, timeout=180.0,
                                  headers={"Content-Type": "application/json"})
            response.raise_for_status()
            break  # successo — esci dal loop
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status in (429, 500, 502, 503, 504) and attempt < 3:
                wait = 20 * attempt  # 20s, poi 40s
                print(f"[Gemini] HTTP {status} attempt={attempt} — retry tra {wait}s…")
                time.sleep(wait)
            else:
                raise  # rilancia dopo 3 tentativi o su errori non retriable (es. 400, 401)
        except httpx.TimeoutException:
            if attempt < 3:
                print(f"[Gemini] timeout attempt={attempt} — retry tra 30s…")
                time.sleep(30)
            else:
                raise

    data = response.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    result = _extract_json(text)
    print(f"[Gemini] stanze restituite: {len(result.get('stanze', []))} / {n} foto")
    return result


# ── STAGED PHOTOS ─────────────────────────────────────────────────────────────

async def generate_staged_photos(photos: list, analysis: dict,
                                  prefs: dict | None = None) -> list:
    stanze  = analysis.get("stanze", [])
    results = [{} for _ in stanze]
    sem     = _get_semaphore()
    budget  = (prefs or {}).get("budget", 3000)
    style   = (prefs or {}).get("style", "minimalista")
    style_dna = get_style_dna(style)

    async def _run_room(i: int, room: dict):
        idx         = room.get("indice_foto", i)
        photo_bytes = photos[idx]["content"] if idx < len(photos) else None
        if not photo_bytes:
            return

        is_bathroom = room.get("is_bathroom", False) or \
                      room.get("room_type", "").lower() in ("bagno", "bathroom")
        room_type   = room.get("room_type", "")
        cwc         = room.get("current_wall_color", "")
        fpc         = room.get("force_paint_color", style_dna.get("palette", "").split(",")[0].strip() or "warm white")
        fpf         = room.get("force_paint_finish", "matte")
        floor       = room.get("floor_description", "existing floor, clean")
        light       = room.get("light_source", "natural daylight from window")
        spatial     = room.get("spatial_map", "")
        keywords    = room.get("mandatory_visual_keywords", [])
        detected    = room.get("detected_elements", [])
        interventions = room.get("interventi", [])

        esp = next(
            (e for e in room.get("esperimenti_staged", []) if e.get("logic_id") == "E_WALL_FORCE"),
            {}
        )

        stage1_prompt = _build_stage1_prompt(
            esp.get("prompt_stage1", ""),
            cwc, fpc, fpf, floor, light, is_bathroom,
            detected_elements=detected,
            style_dna=style_dna,
            room_type=room_type,
            interventions=interventions,
        )
        stage2_prompt = _build_stage2_prompt(
            esp.get("prompt_stage2", ""),
            fpc, fpf, keywords, spatial, is_bathroom,
            style_dna=style_dna,
            budget=budget,
            room_type=room_type,
            interventions=interventions,
        )

        loop = asyncio.get_running_loop()
        async with sem:
            result = await loop.run_in_executor(
                _imagen_executor, _approach_E_two_stage_v22,
                photo_bytes, stage1_prompt, stage2_prompt, is_bathroom, cwc
            )
        results[i]["E_WALL_FORCE"] = result

    await asyncio.gather(*[_run_room(i, room) for i, room in enumerate(stanze)])
    return results


def _build_stage1_prompt(base: str, cwc: str, fpc: str, fpf: str,
                          floor: str, light: str, is_bathroom: bool,
                          detected_elements: list | None = None,
                          style_dna: dict | None = None,
                          room_type: str = "",
                          interventions: list | None = None) -> str:
    """
    Stage 1 v24.1 — The Canvas (Hard-Refurbishment):
    - Pareti: force-paint con solid opaque coating
    - Bagno: STRUCTURAL OVERWRITE piastrelle con micro-cement
    - Cucina: STRUCTURAL OVERWRITE ante + piastrelle backsplash
    - Anti-patterns: elimina esplicitamente colori obsoleti nominati
    - Decluttering per categoria
    """
    dna  = style_dna or {}
    ivs  = interventions or []
    det  = detected_elements or []
    is_kitchen = "cucina" in room_type.lower() or "kitchen" in room_type.lower()

    # ── Individua colori/materiali obsoleti da eliminare ─────────────────────
    anti_colors = []
    if cwc:
        anti_colors.append(cwc)
    for el in det:
        el_l = el.lower()
        if any(c in el_l for c in ["blu", "blue", "azzurr", "giallo", "yellow",
                                    "verde", "marrone", "arancio", "floral",
                                    "striped", "righe", "fantasia", "pattern"]):
            anti_colors.append(el)
    anti_block = ""
    if anti_colors:
        anti_block = (
            "ANTI-PATTERN ELIMINATION: The following colors and patterns are "
            "COMPLETELY ELIMINATED — zero trace visible anywhere: "
            + ", ".join(anti_colors[:5]) + ". "
            "Replace everything with clean neutral surfaces. "
        )

    # ── Individua se ci sono interventi su piastrelle/ante dal preventivo ────
    has_microcement = any(
        any(k in (iv.get("titolo", "") + iv.get("dettaglio", "")).lower()
            for k in ["micro", "piastrelle", "tile", "rivestimento"])
        for iv in ivs
    )
    has_cabinet_reface = any(
        any(k in (iv.get("titolo", "") + iv.get("dettaglio", "")).lower()
            for k in ["ante", "cabinet", "vernicia", "rifacimento"])
        for iv in ivs
    )
    has_sanitari = any(
        any(k in (iv.get("titolo", "") + iv.get("dettaglio", "")).lower()
            for k in ["wc", "bidet", "sanitari", "lavandino", "vasca"])
        for iv in ivs
    )

    # ── Paint target dal DNA di stile se Gemini non ha specificato ───────────
    paint_target = fpc or (dna.get("palette", "").split(",")[0].strip()) or "warm white"
    paint_finish = fpf or "matte"

    # ═══════════════════════════════════════════════════════════════════════════
    # BAGNO
    # ═══════════════════════════════════════════════════════════════════════════
    if is_bathroom:
        tile_cmd = (
            "STRUCTURAL OVERWRITE — TILES: ALL existing tiles (blue, aqua, any color) "
            "are COMPLETELY HIDDEN under a solid opaque layer of light warm grey "
            "micro-cement finish (Isoplam Microverlay style). "
            "Zero original tile color visible. Zero original tile texture visible. "
            "The tile grid lines are COVERED. Only smooth matte grey surface remains. "
            "Upper walls above tiles: solid opaque warm white matte paint. "
        ) if has_microcement else (
            "TILE RECOLOR: Existing tiles are painted over with solid opaque white "
            "tile paint. Zero original blue/aqua color showing through. "
        )

        sanitari_cmd = (
            "FIXTURE REMOVAL: Remove old toilet, bidet, sink, vanity cabinet, "
            "mirror, and all chrome fixtures. Result: empty positions on wall. "
        ) if has_sanitari else (
            "Remove all personal items, bottles, soaps, bath mats from surfaces. "
        )

        floor_cmd = (
            "Floor: existing terrazzo/graniglia clean and polished. "
            "Ceiling: freshly painted white. "
        )

        declutter = (
            "Remove: all personal objects, shampoo bottles, soap, towels on floor, "
            "shower curtain, old chrome rails and accessories, toilet brush holder, "
            "metal shelf contents. "
            "Keep: bathtub structure, window, radiator. "
            "DO NOT add windows not present in original. "
        )

        return (
            anti_block + tile_cmd + sanitari_cmd + declutter + floor_cmd
            + f"Lighting: {light}, bright and clean. "
            + f"Empty renovated bathroom shell. {SCENIC_CONFIG}."
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # CUCINA
    # ═══════════════════════════════════════════════════════════════════════════
    if is_kitchen:
        cabinet_color = "sage green matte" if "verde" in " ".join(
            iv.get("dettaglio", "") for iv in ivs).lower() else "warm white matte"

        cabinet_cmd = (
            f"CABINET OVERWRITE: ALL existing cabinet doors (blue, any color) are "
            f"NOW repainted with solid high-opacity {cabinet_color} paint. "
            "Zero original blue color showing through. "
            "Original cabinet hardware/handles replaced with natural wood pulls. "
        ) if has_cabinet_reface else (
            "Cabinet doors: repainted white matte. Zero original color visible. "
        )

        backsplash_cmd = (
            "BACKSPLASH OVERWRITE: All existing tile backsplash "
            "is COMPLETELY COVERED with light grey micro-cement finish. "
            "Zero original white tile texture or grout lines visible. "
            "Smooth matte grey surface. "
        ) if has_microcement else (
            "Backsplash tiles: cleaned and white, grout lines visible. "
        )

        wall_cmd = (
            f"Upper walls: solid opaque warm white matte paint. "
            "Zero original yellowed or stained paint visible. "
        )

        appliance_cmd = (
            "Remove: old white oven, old range hood, old dishwasher. "
            "Keep: refrigerator position, sink position. "
            "Result: clean empty appliance spaces ready for new units. "
        )

        declutter = (
            "Remove: plaid tablecloth, old chairs, personal items on surfaces, "
            "magnets from fridge, dish rack, old curtains. "
            "Keep: window, door to balcony, fixed cabinets. "
        )

        floor_cmd = "Floor: existing graniglia/terrazzo clean. Ceiling: white. "

        return (
            anti_block + cabinet_cmd + backsplash_cmd + wall_cmd
            + appliance_cmd + declutter + floor_cmd
            + f"Lighting: {light}, bright natural light from balcony door. "
            + f"Empty renovated kitchen shell. {SCENIC_CONFIG}."
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # ALTRI AMBIENTI (soggiorno, camera, ecc.)
    # ═══════════════════════════════════════════════════════════════════════════
    paint_cmd = (
        f"WALL TRANSFORMATION: Original {cwc} walls are now " if cwc
        else "WALL TRANSFORMATION: All walls are now "
    )
    paint_cmd += (
        f"completely covered with solid high-opacity {paint_target} {paint_finish} paint. "
        "Zero bleed-through. 100% solid coverage. No trace of previous color. "
    )

    ugly_items = ""
    bad_keywords = ["old", "datato", "brutto", "ingombrante", "obsoleto",
                    "spazzatura", "cavo", "sacco", "bag", "metallico", "ventilatore"]
    targets = [e for e in det if any(b in e.lower() for b in bad_keywords)]
    if targets:
        ugly_items = "Remove specifically: " + ", ".join(targets[:6]) + ". "

    declutter = (
        "DECLUTTERING: Remove ALL furniture, rugs, curtains, and decorative objects. "
        "Explicitly remove: ceiling fan, bookshelf, dining table with chairs, "
        "old sofas with striped/floral fabric, wall-mounted shelves with objects, "
        "framed religious images, floor lamps. "
        + ugly_items +
        "Keep ONLY: windows, doors, radiator, built-in elements. "
        "Result: bright empty room. "
    )

    return (
        anti_block + paint_cmd + declutter
        + f"Floor: {floor}, clean. Ceiling: white. "
        + f"Natural light from {light}. "
        + f"Empty clean room. {SCENIC_CONFIG}."
    )


def _build_stage2_prompt(base: str, fpc: str, fpf: str,
                          keywords: list, spatial: str, is_bathroom: bool,
                          style_dna: dict | None = None,
                          budget: int = 3000,
                          room_type: str = "",
                          interventions: list | None = None) -> str:
    """
    Stage 2 v24.1 — Catchy Airbnb Staging:
    Budget-driven + Style DNA + Catchy Layer + Lifestyle Staging
    """
    dna    = style_dna or {}
    ivs    = interventions or []
    is_kitchen = "cucina" in room_type.lower() or "kitchen" in room_type.lower()
    paint_target = fpc or "warm white"

    # Conferma superfici rifatte (ancora il modello allo Stage 2)
    surface_confirm = (
        f"The room surfaces are freshly renovated: {paint_target} {fpf} walls, "
        "all tiles covered with micro-cement, all cabinet doors repainted. "
        "Zero traces of previous blue, aqua, yellow, or patterned finishes. "
    )

    # Cost-to-Prompt: keyword dal preventivo
    kw_block = ""
    if keywords:
        kw_block = "VISUALIZE THESE BUDGET ITEMS: " + ". ".join(keywords[:6]) + ". "

    spatial_block = f"LAYOUT: {spatial}. " if spatial else ""

    # ═══════════════════════════════════════════════════════════════════════════
    # BAGNO
    # ═══════════════════════════════════════════════════════════════════════════
    if is_bathroom:
        has_sanitari = any(
            any(k in (iv.get("titolo","") + iv.get("dettaglio","")).lower()
                for k in ["wc", "bidet", "sanitari", "lavandino"])
            for iv in ivs
        )
        fixtures = (
            "NEW FIXTURES: Wall-hung white ceramic toilet and bidet (modern design). "
            "Floating wood-finish vanity unit with integrated white ceramic sink. "
            "Matte black designer faucets on sink and bathtub. "
        ) if has_sanitari else (
            "Existing bathtub repainted white. White sink with new matte black faucet. "
        )

        staging = (
            "BATHROOM STAGING: "
            "Large round frameless mirror or round black-framed mirror above vanity. "
            "3 neatly rolled white fluffy towels stacked on shelf. "
            "Small succulent or air plant in white ceramic pot on vanity. "
            "Matte black soap dispenser and toothbrush holder. "
            "Scented candle in glass jar on edge of bathtub. "
            "Minimal wooden bath tray across bathtub. "
        )
        end = f"Bright, clean, spa-like atmosphere. {SCENIC_CONFIG}."
        return surface_confirm + kw_block + fixtures + staging + end

    # ═══════════════════════════════════════════════════════════════════════════
    # CUCINA
    # ═══════════════════════════════════════════════════════════════════════════
    if is_kitchen:
        cabinet_color = "sage green matte" if "verde" in " ".join(
            iv.get("dettaglio","") for iv in ivs).lower() else "warm white matte"

        has_new_appliances = any(
            any(k in (iv.get("titolo","") + iv.get("dettaglio","")).lower()
                for k in ["forno", "piano cottura", "induzione", "lavello"])
            for iv in ivs
        )
        appliances = (
            "NEW APPLIANCES: Built-in stainless steel induction hob and oven. "
            "Stainless steel undermount sink with matte black faucet. "
        ) if has_new_appliances else (
            "Existing appliances kept, fridge clean and white. "
        )

        has_bar = any(
            "bancone" in (iv.get("titolo","") + iv.get("dettaglio","")).lower()
            for iv in ivs
        )
        bar_cmd = (
            "Small natural wood breakfast bar with 2 rattan bar stools. "
        ) if has_bar else ""

        staging = (
            "KITCHEN LIFESTYLE STAGING: "
            f"Cabinets now {cabinet_color} with natural wood handles. "
            + appliances +
            "Open wooden shelves with: white ceramic bowls stacked, "
            "small potted fresh herbs (basil, rosemary) on windowsill, "
            "wooden cutting board leaning against backsplash. "
            + bar_cmd +
            "Rattan pendant lamp above work area. "
            "Natural linen curtain on balcony door. "
        )
        end = f"Modern functional kitchen, magazine-ready. {SCENIC_CONFIG}."
        return surface_confirm + kw_block + staging + end

    # ═══════════════════════════════════════════════════════════════════════════
    # ALTRI AMBIENTI
    # ═══════════════════════════════════════════════════════════════════════════
    wall_confirm = (
        f"Fresh {paint_target} {fpf} painted walls. "
        "Zero traces of previous yellowed or patterned finishes. "
    )

    furniture = dna.get("furniture", "modern IKEA furniture")
    textiles  = dna.get("textiles",  "linen cushions, soft throw")
    plants    = dna.get("plants",    "Monstera in ceramic pot")
    props     = dna.get("props",     "ceramic vase, art print")
    lighting  = dna.get("lighting",  "warm floor lamp 2700K")

    if budget >= 3000:
        budget_directive = (
            f"FULL REPLACEMENT: {furniture}. "
            "Remove all dated pieces, place only new modern equivalents. "
        )
    else:
        budget_directive = (
            "REFACE MODE: Keep structural furniture, recolor surfaces. "
            "Add accessories and textiles to refresh. "
        )

    catchy = (
        "CATCHY AIRBNB LAYER: "
        f"Sofa with {textiles} — minimum 5 cushions mixed linen/velvet/bouclé. "
        f"Statement plant: {plants} in corner. "
        f"Props: {props} on coffee table — open art book, ceramic vase with pampas grass. "
        f"Lighting: {lighting} switched on, warm amber glow. "
        "Natural jute rug anchoring seating area. "
        "Macramé wall hanging above sofa. "
        "Sheer linen curtains on windows. "
        "Everything magazine-styled, Airbnb hero-shot ready. "
    )

    end = f"Scene fully styled and photoshoot-ready. {SCENIC_CONFIG}."
    return wall_confirm + kw_block + spatial_block + budget_directive + catchy + end







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


# ── Core: E two-stage v22 ────────────────────────────────────────────────────

def _approach_E_two_stage_v22(photo_bytes: bytes,
                               stage1_prompt: str, stage2_prompt: str,
                               is_bathroom: bool = False,
                               current_wall_color: str = "") -> str | None:
    """
    Stage 1 (guidance=16): Force-Paint Canvas — svuota + ridipinge pareti.
    Stage 2 (guidance=12): Catchy Layering — arreda con mandatory_visual_keywords.
    Retry 2× con backoff 15s per ogni stage.
    """
    import time

    neg_s1 = (
        "furniture, objects, clutter, trash bags, cables, "
        "original wall color, bleed-through, transparent paint, "
        "color bleeding, previous paint showing through, "
        "distorted architecture, watermark, low quality, unrealistic"
    )
    if current_wall_color:
        neg_s1 += f", {current_wall_color}, {current_wall_color} walls"
    if is_bathroom:
        neg_s1 += ", added windows, fake windows, exterior view, removed tiles"

    neg_s2 = (
        "empty room, bare walls, clutter, trash bags, "
        "original wall color retained, bleed-through, "
        "dark shadows, overexposed, noise, cartoon, "
        "floating objects, sticker effect, inconsistent shadows, "
        "distorted architecture, watermark"
    )
    if is_bathroom:
        neg_s2 += ", added windows, fake windows, exterior view, missing tiles"

    bath_tag = " [BAGNO]" if is_bathroom else ""
    g1 = GUIDANCE["E_STAGE1_CANVAS"]
    g2 = GUIDANCE["E_STAGE2_STAGING"]

    def _call_imagen(prompt: str, ref_bytes: bytes, guidance: float,
                     neg: str, tag: str) -> bytes | None:
        for attempt in range(1, 3):
            try:
                compressed = compress_image(ref_bytes, max_width=1024, quality=80)
                print(f"[{tag}] attempt={attempt} {len(compressed)//1024}KB g={guidance}")
                client = _get_vertex_client()
                resp = client.models.edit_image(
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
                        negative_prompt=neg,
                        safety_filter_level="block_only_high",
                    ),
                )
                if resp.generated_images:
                    print(f"[{tag}] SUCCESS attempt={attempt}")
                    return resp.generated_images[0].image.image_bytes
                print(f"[{tag}] 0 immagini (g={guidance}) — safety filter o quota")
                return None
            except Exception as e:
                print(f"[{tag}] ERRORE attempt={attempt}: {type(e).__name__}: {e}")
                if attempt < 2:
                    print(f"[{tag}] retry tra 15s…")
                    time.sleep(15)
                else:
                    return None

    # Stage 1
    s1_bytes = _call_imagen(
        stage1_prompt, photo_bytes, g1, neg_s1,
        f"E-S1{bath_tag}"
    )
    if s1_bytes is None:
        print(f"[E-S1] FALLITO — uso foto originale come fallback per Stage2")
        s1_bytes = compress_image(photo_bytes, max_width=1024, quality=80)

    # Stage 2
    s2_bytes = _call_imagen(
        stage2_prompt, s1_bytes, g2, neg_s2,
        f"E-S2{bath_tag}"
    )
    if s2_bytes is None:
        print("[E-S2] FALLITO — nessuna immagine per questa stanza")
        return None

    return base64.b64encode(s2_bytes).decode()



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
