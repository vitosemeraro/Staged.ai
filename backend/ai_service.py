"""
AI Service v19 — Template-First + Structural Tagging

Strategia: Gemini restituisce SOLO variabili (colori, elementi, vincoli).
Python costruisce i prompt Imagen con template hardcodati in Python.
Questo elimina la "LLM Laziness" dove Gemini restituisce [COMPILA...].

Moduli:
  PhotoValidator        — invariato da v18
  GlobalSpatialAnchor  — Gemini mappa adiacenze e STRUCTURAL_FIXED
  TemplateFirst        — Python compone i prompt Imagen, non Gemini
  StructuralTagging    — elementi STRUCTURAL_FIXED mai rimossi in Stage 1
  KitchenBrutalistProtocol — template specifico per cucine
  E guidance aggiornato: Stage1=16, Stage2=12
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

# ── Guidance ──────────────────────────────────────────────────────────────────
# EDIT_MODE_DEFAULT: >30 = 0 immagini silenzioso
# E Stage1=16 (più aggressivo per pittura), Stage2=12 (mantiene struttura)
GUIDANCE = {
    "D_STAGE1_CLEAN":   8,
    "D_STAGE2_STAGE":  10,
    "E_STAGE1_WALL":   16,   # alzato da 14 → 16 per forzare cambio colore
    "E_STAGE2_STAGE":  12,   # alzato da 10 → 12 per mantenere nuove pareti
}

# ── Palette per stile ─────────────────────────────────────────────────────────
STYLE_PALETTES = {
    "Scandinavian": {
        "wall_color":            "warm white",
        "wall_finish":           "matte",
        "wall_forbidden":        "orange, terracotta, red, electric blue, yellow, dark colors",
        "wood":                  "Light Oak",
        "metal":                 "Matte Black",
        "textiles":              "natural linen, soft grey, cream cotton",
        "kitchen_cabinet_color": "matte white",
        "kitchen_counter":       "white quartz",
        "kitchen_accent":        "Matte Black pendant lamp, potted herbs in white ceramic",
    },
    "Industrial": {
        "wall_color":            "warm off-white",
        "wall_finish":           "matte concrete-effect",
        "wall_forbidden":        "pink, pastel, bright orange, yellow, bright colors",
        "wood":                  "Reclaimed Dark Wood",
        "metal":                 "Dark Steel",
        "textiles":              "dark grey linen, charcoal canvas, warm beige",
        "kitchen_cabinet_color": "matte charcoal grey",
        "kitchen_counter":       "dark industrial concrete-effect laminate",
        "kitchen_accent":        "Dark Steel Edison pendant lamp, potted herbs in terracotta",
    },
    "Japandi": {
        "wall_color":            "warm greige",
        "wall_finish":           "matte clay",
        "wall_forbidden":        "saturated colors, neon, orange, electric blue",
        "wood":                  "Blonde Bamboo",
        "metal":                 "Brushed Brass",
        "textiles":              "warm linen, undyed cotton, wabi-sabi textures",
        "kitchen_cabinet_color": "warm linen white",
        "kitchen_counter":       "natural light stone",
        "kitchen_accent":        "Brushed Brass pendant lamp, ceramic pots with herbs",
    },
    "Minimalista": {
        "wall_color":            "light greige",
        "wall_finish":           "matte",
        "wall_forbidden":        "saturated colors, orange, red, dark colors",
        "wood":                  "Light Natural Wood",
        "metal":                 "Matte Black",
        "textiles":              "natural linen, soft white, neutral grey",
        "kitchen_cabinet_color": "matte white",
        "kitchen_counter":       "light natural stone",
        "kitchen_accent":        "Matte Black pendant lamp, minimalist ceramic vases",
    },
}

def _get_palette(style: str) -> dict:
    key = style.strip().title()
    if "Scandi" in key:   key = "Scandinavian"
    elif "Japandi" in key or "Japan" in key: key = "Japandi"
    elif "Industri" in key: key = "Industrial"
    elif "Minimal" in key:  key = "Minimalista"
    return STYLE_PALETTES.get(key, {
        "wall_color":            "light greige",
        "wall_finish":           "matte",
        "wall_forbidden":        "saturated colors, bright orange, neon",
        "wood":                  "Light Natural Wood",
        "metal":                 "Matte Black",
        "textiles":              "natural linen, neutral grey",
        "kitchen_cabinet_color": "matte white",
        "kitchen_counter":       "light natural stone",
        "kitchen_accent":        "Matte Black pendant lamp",
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
# PhotoValidator
# ═══════════════════════════════════════════════════════════════════════════════

async def validate_input_photos(photos: list) -> dict:
    if not photos:
        return {"valid": [], "issues": [], "warnings": [], "all_valid": False}
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_gemini_executor, _validate_photos_sync, photos)


def _validate_photos_sync(photos: list) -> dict:
    n = len(photos)
    parts = []
    for i, p in enumerate(photos):
        parts.append({"text": f"[FOTO {i}]"})
        parts.append({"inline_data": {
            "mime_type": p["content_type"],
            "data": base64.b64encode(p["content"]).decode()
        }})

    parts.append({"text": (
        f"Analizza queste {n} foto per AI home staging. Per ogni foto:\n"
        "1. È sufficientemente illuminata? (scarta se buia)\n"
        "2. Si vede abbastanza della stanza? (scarta se troppo stretta)\n"
        "3. Elementi strutturali fissi chiaramente visibili? (frigo, armadio, finestre)\n"
        "4. Tipo di stanza?\n"
        "Restituisci SOLO JSON:\n"
        "{\"photos\":[{\"index\":0,\"valid\":true,\"room_type\":\"soggiorno\","
        "\"issue\":\"ok\",\"suggestion\":\"\","
        "\"structural_elements_visible\":[\"frigorifero angolo nord-est\",\"armadio parete ovest\"]}],"
        "\"global_warnings\":[],\"layout_hint\":\"\"}"
    )})

    payload = {
        "system_instruction": {"parts": [{"text":
            "Sei un esperto di fotografia immobiliare. Rispondi SOLO con JSON valido."
        }]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 4096,
                             "responseMimeType": "application/json"}
    }
    try:
        resp = httpx.post(GEMINI_URL, json=payload, timeout=60.0,
                          headers={"Content-Type": "application/json"})
        resp.raise_for_status()
        parsed = _extract_json(resp.json()["candidates"][0]["content"]["parts"][0]["text"])
        photos_data = parsed.get("photos", [])
        print(f"[PhotoValidator] {sum(p.get('valid',True) for p in photos_data)}/{n} valide")
        return {
            "valid":       [p.get("valid", True) for p in photos_data],
            "issues":      [p.get("issue", "ok") for p in photos_data],
            "room_types":  [p.get("room_type", "unknown") for p in photos_data],
            "suggestions": [p.get("suggestion", "") for p in photos_data],
            "structural":  [p.get("structural_elements_visible", []) for p in photos_data],
            "warnings":    parsed.get("global_warnings", []),
            "layout_hint": parsed.get("layout_hint", ""),
            "all_valid":   all(p.get("valid", True) for p in photos_data),
        }
    except Exception as e:
        print(f"[PhotoValidator] ERRORE: {e} — procedo comunque")
        return {"valid":[True]*n, "issues":["skip"]*n, "room_types":["unknown"]*n,
                "suggestions":[""]*n, "structural":[[]]*n,
                "warnings":[], "layout_hint":"", "all_valid":True}


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
    if len(stanze) != n_photos:
        print(f"[WARNING] Gemini: {len(stanze)} stanze vs {n_photos} foto")
    for i, room in enumerate(stanze):
        idx = room.get("indice_foto")
        if idx is None or not isinstance(idx, int) or idx >= n_photos:
            room["indice_foto"] = i
    return analysis


# ═══════════════════════════════════════════════════════════════════════════════
# GEMINI ANALYSIS — restituisce solo VARIABILI, non prompt
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
    """
    Template-First: Gemini restituisce SOLO le variabili necessarie.
    I prompt Imagen vengono costruiti in Python con _build_prompt_*.
    Questo elimina la "LLM Laziness" (placeholder non compilati).
    """
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

    foto_index_list = "\n".join(
        f"  FOTO {i} = stanza {i+1}" for i in range(n)
    )

    system_instruction = (
        "Sei un Architetto Digitale specializzato in home staging. "
        "Analizza le foto e restituisci SOLO le variabili richieste — "
        "NON scrivere prompt Imagen. "
        f"PALETTE AUTORIZZATA per {style}: pareti = {palette['wall_color']} {palette['wall_finish']}. "
        f"VIETATO: {palette['wall_forbidden']}. "
        f"Materiali fissi: legno={palette['wood']}, metallo={palette['metal']}. "
        "STRUCTURAL_FIXED: elementi che NON devono mai essere rimossi o spostati "
        "(frigorifero, armadio a muro, finestre, colonne strutturali). "
        "Rispondi ESCLUSIVAMENTE con JSON valido, senza markdown."
    )

    prompt = (
        f"Analizza {n} foto di un appartamento per home staging {dest_label}.\n"
        f"Budget \u20ac{budget} | Stile: {style} | Citta': {location}\n"
        f"Distribuzione: Arredi \u20ac{alloc['arredi']} | "
        f"Tinteggiatura \u20ac{alloc['tinteggiatura']} | "
        f"Materiali \u20ac{alloc['materiali']} | Montaggio \u20ac{alloc['montaggio']}\n\n"
        f"FOTO:\n{foto_index_list}\n\n"

        f"STEP 0 — MAPPA SPAZIALE:\n"
        f"Identifica stanze adiacenti e crea spatial_adjacencies.\n"
        f"Per ogni adiacenza nota i vincoli spaziali critici.\n\n"

        f"STEP 1 — PER OGNI FOTO, estrai queste variabili:\n"
        f"  room_type: soggiorno|cucina|camera|bagno|altro\n"
        f"  is_kitchen: true/false\n"
        f"  current_wall_color: colore esatto pareti (es. 'giallo paglierino')\n"
        f"  light_source: posizione sorgente luce (es. 'finestra sinistra in alto')\n"
        f"  shadow_direction: direzione ombre (es. 'verso destra e verso il basso')\n"
        f"  floor_material: materiale pavimento (es. 'laminato chiaro')\n"
        f"  structural_fixed: lista di elementi INAMOVIBILI con posizione:\n"
        f"    - formato: 'nome_elemento|posizione' (es. 'frigorifero|angolo nord-est', 'armadio|intera parete ovest')\n"
        f"    - INCLUDI SEMPRE: frigorifero, armadi a muro, finestre, colonne\n"
        f"    - Gli armadi a muro BLOCCANO la disposizione del letto davanti — segnalalo\n"
        f"  furniture_to_replace: lista mobili brutti da sostituire con il loro sostituto:\n"
        f"    - formato: 'vecchio|nuovo|posizione' (es. 'divano rosso|divano grigio {palette['wood']}|centro stanza')\n"
        f"    - Replacement logic cucina: mantieni tavolo nella stessa posizione\n"
        f"    - NON spostare elementi structural_fixed\n"
        f"  layering_add: lista elementi decorativi da aggiungere:\n"
        f"    - tappeto, cuscini, piante, lampade, quadri coerenti con {style}\n"
        f"    - cucina: aggiungi solo lampada a sospensione, piante aromatiche\n"
        f"  kitchen_vars (solo se is_kitchen=true):\n"
        f"    original_cabinet_color: colore ante attuali\n"
        f"    original_cabinet_texture: materiale ante attuali\n\n"

        f"STEP 2 — Genera interventi e costi per ogni stanza (in italiano).\n\n"

        f"Restituisci questo JSON ({n} oggetti in 'stanze'):\n"
        "{{\n"
        "  \"spatial_adjacencies\": [\n"
        "    {{\"rooms\":[0,1], \"note\":\"vincolo spaziale\"}}\n"
        "  ],\n"
        "  \"wall_color_global\": \"colore unico per tutte le stanze\",\n"
        "  \"wall_finish_global\": \"finitura unica per tutte le stanze\",\n"
        "  \"valutazione_generale\": \"analisi\",\n"
        "  \"punti_di_forza\": [\"p1\"],\n"
        "  \"criticita\": [\"c1\"],\n"
        f"  \"potenziale_str\": \"potenziale a {location}\",\n"
        "  \"tariffe\": {{\"attuale_notte\": \"\u20acXX-YY\", \"post_restyling_notte\": \"\u20acXX-YY\", \"incremento_percentuale\": \"XX%\"}},\n"
        f"  \"stanze\": [\n"
        f"    /* {n} oggetti, uno per foto */\n"
        "    {{\n"
        "      \"nome\": \"Nome stanza\",\n"
        "      \"indice_foto\": 0,\n"
        "      \"room_type\": \"soggiorno\",\n"
        "      \"is_kitchen\": false,\n"
        "      \"current_wall_color\": \"giallo paglierino\",\n"
        "      \"light_source\": \"finestra sinistra\",\n"
        "      \"shadow_direction\": \"ombre verso destra\",\n"
        "      \"floor_material\": \"laminato chiaro\",\n"
        "      \"structural_fixed\": [\"frigorifero|angolo nord-est\", \"armadio|intera parete ovest — NON posizionare letto davanti\"],\n"
        "      \"furniture_to_replace\": [\"divano rosso|divano grigio lineare tessuto neutro|centro stanza\"],\n"
        "      \"layering_add\": [\"tappeto neutro\", \"cuscini lino\", \"lampada stelo\", \"pianta angolo\", \"2 quadri sopra divano\"],\n"
        "      \"kitchen_vars\": null,\n"
        "      \"stato_attuale\": \"descrizione\",\n"
        "      \"interventi\": [\n"
        "        {{\"titolo\": \"nome\", \"dettaglio\": \"dettaglio\", \"costo_min\": 50, \"costo_max\": 100, \"priorita\": \"alta\", \"dove_comprare\": \"negozio\"}}\n"
        "      ],\n"
        "      \"costo_totale_stanza\": 350\n"
        "    }}\n"
        "  ],\n"
        "  \"riepilogo_costi\": {{\"manodopera_tinteggiatura\":0,\"materiali_pittura\":0,\"arredi_complementi\":0,\"montaggio_varie\":0,\"totale\":0,\"budget_residuo\":0,\"nota_budget\":\"\"}},\n"
        "  \"piano_acquisti\": [{{\"categoria\":\"Arredi\",\"articoli\":[\"item\"],\"budget_stimato\":0,\"negozi_consigliati\":\"IKEA, H&M Home\"}}],\n"
        "  \"titolo_annuncio_suggerito\": \"Titolo max 50 car\",\n"
        "  \"highlights_str\": [\"h1\",\"h2\",\"h3\"],\n"
        "  \"roi_restyling\": \"ROI...\"\n"
        "}}"
    )

    parts = []
    for i, p in enumerate(photos):
        parts.append({"text": f"[FOTO {i} — stanza {i+1}]"})
        parts.append({"inline_data": {
            "mime_type": p["content_type"],
            "data": base64.b64encode(p["content"]).decode()
        }})
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

    resp = httpx.post(GEMINI_URL, json=payload, timeout=180.0,
                      headers={"Content-Type": "application/json"})
    resp.raise_for_status()

    data = resp.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    result = _extract_json(text)
    print(f"[Gemini] stanze={len(result.get('stanze',[]))}/{n} | "
          f"adj={len(result.get('spatial_adjacencies',[]))}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPLATE-FIRST: Python costruisce i prompt Imagen
# ═══════════════════════════════════════════════════════════════════════════════

def _structural_fixed_clause(structural_fixed: list) -> str:
    """Genera la clausola DO NOT REMOVE per elementi strutturali."""
    if not structural_fixed:
        return ""
    clauses = []
    for item in structural_fixed:
        parts = item.split("|")
        name = parts[0].strip()
        pos  = parts[1].strip() if len(parts) > 1 else "its original position"
        clauses.append(
            f"DO NOT REMOVE OR CHANGE the {name} located at {pos}. "
            f"Keep at least 90cm of clear access space in front of {name}."
        )
    return " ".join(clauses)


def _build_d_stage1(room: dict, palette: dict, wall_color: str, wall_finish: str) -> str:
    """Template Stage 1 per D: pulisce la stanza, applica nuove pareti."""
    sf_clause = _structural_fixed_clause(room.get("structural_fixed", []))
    floor     = room.get("floor_material", "existing floor")
    light     = room.get("light_source", "natural light")
    return (
        f"Achromatic blank room. Remove all furniture, clutter, objects and decor. "
        f"{sf_clause} "
        f"Paint ALL walls in {wall_finish} {wall_color} — solid, opaque, complete coverage. "
        f"Keep original {floor} floor unchanged. "
        f"Empty clean room ready for new furnishing. "
        f"Consistent lighting from {light}. Photorealistic, 8k."
    )


def _build_d_stage2(room: dict, palette: dict, wall_color: str, wall_finish: str) -> str:
    """Template Stage 2 per D: aggiunge layering su stanza vuota."""
    sf_clause   = _structural_fixed_clause(room.get("structural_fixed", []))
    replacements = room.get("furniture_to_replace", [])
    layering     = room.get("layering_add", [])
    light        = room.get("light_source", "natural light")
    shadow       = room.get("shadow_direction", "soft shadows")
    floor        = room.get("floor_material", "existing floor")

    # Replacement logic
    repl_text = ""
    for r in replacements:
        parts = r.split("|")
        if len(parts) >= 3:
            old, new, pos = parts[0].strip(), parts[1].strip(), parts[2].strip()
            repl_text += (
                f"In place of the {old}, place a {new} in {palette['wood']} wood "
                f"at {pos}, same footprint as original. "
            )

    # Layering text
    layer_text = ", ".join(layering) if layering else "decorative cushions, small plant, area rug"

    return (
        f"The room features {wall_finish} {wall_color} walls on all surfaces. "
        f"{sf_clause} "
        f"{repl_text}"
        f"Add: {layer_text}. "
        f"All wood in {palette['wood']}, all metal in {palette['metal']}, "
        f"textiles in {palette['textiles']}. "
        f"Soft shadows matching {light}, falling {shadow}. "
        f"Consistent shadows, global illumination, ambient occlusion, depth of field f/8, HDR. "
        f"Professional real estate photography, 24mm wide angle lens, cinematic warm lighting, "
        f"balanced exposure, highly detailed, 8k."
    )


def _build_e_stage1(room: dict, palette: dict, wall_color: str, wall_finish: str) -> str:
    """Template Stage 1 per E: più aggressivo, forza cambio colore pareti."""
    sf_clause     = _structural_fixed_clause(room.get("structural_fixed", []))
    current_color = room.get("current_wall_color", "original color")
    light         = room.get("light_source", "natural light")
    floor         = room.get("floor_material", "existing floor")

    is_kitchen = room.get("is_kitchen", False)
    kitchen_vars = room.get("kitchen_vars") or {}
    orig_cab = kitchen_vars.get("original_cabinet_color", "beige")
    orig_tex = kitchen_vars.get("original_cabinet_texture", "")

    kitchen_clause = ""
    if is_kitchen:
        kitchen_clause = (
            f"Strip all {orig_cab} {orig_tex} cabinet doors from their original finish. "
            f"Repaint all cabinet fronts in {palette['kitchen_cabinet_color']} matte finish — "
            f"solid opaque coverage, no {orig_cab} showing through. "
            f"Replace countertop with {palette['kitchen_counter']}. "
        )

    return (
        f"Complete architectural renovation. Achromatic blank room. "
        f"The original {current_color} wall color MUST be entirely replaced. "
        f"ALL walls are now {wall_finish} {wall_color} — "
        f"solid opaque paint, ZERO bleed-through of {current_color}, "
        f"no traces of original color anywhere on any surface. "
        f"{kitchen_clause}"
        f"{sf_clause} "
        f"Keep original {floor} floor unchanged. "
        f"Remove all furniture and clutter except structural_fixed elements. "
        f"Consistent lighting from {light}. Photorealistic, 8k."
    )


def _build_e_stage2(room: dict, palette: dict, wall_color: str, wall_finish: str) -> str:
    """Template Stage 2 per E: conferma nuove pareti + layering + materiali."""
    sf_clause    = _structural_fixed_clause(room.get("structural_fixed", []))
    replacements = room.get("furniture_to_replace", [])
    layering     = room.get("layering_add", [])
    light        = room.get("light_source", "natural light")
    shadow       = room.get("shadow_direction", "soft shadows")
    current_col  = room.get("current_wall_color", "original color")

    is_kitchen   = room.get("is_kitchen", False)
    kitchen_vars = room.get("kitchen_vars") or {}
    orig_cab     = kitchen_vars.get("original_cabinet_color", "beige")

    # Conferma nuove pareti (forza il modello a "ricordare" lo Stage 1)
    wall_confirm = (
        f"Working on the newly painted {wall_finish} {wall_color} walls from Stage 1. "
        f"These walls show absolutely zero trace of the previous {current_col}. "
    )

    # Kitchen specific
    kitchen_layer = ""
    if is_kitchen:
        kitchen_layer = (
            f"The {palette['kitchen_cabinet_color']} kitchen cabinets from Stage 1 are now repainted. "
            f"No {orig_cab} visible anywhere on the cabinets. "
            f"Add {palette['kitchen_accent']}. "
            f"Replace dining table with {palette['wood']} wood table in same position. "
        )

    # Replacement logic per stanze non-cucina
    repl_text = ""
    if not is_kitchen:
        for r in replacements:
            parts = r.split("|")
            if len(parts) >= 3:
                old, new, pos = parts[0].strip(), parts[1].strip(), parts[2].strip()
                repl_text += (
                    f"In place of the {old}, place a {new} in {palette['wood']} wood "
                    f"at {pos}, same footprint as original. "
                )

    layer_text = ", ".join(layering) if layering else "area rug, cushions, plant, floor lamp"

    return (
        f"{wall_confirm}"
        f"{sf_clause} "
        f"{kitchen_layer}"
        f"{repl_text}"
        f"Add: {layer_text}. "
        f"MATERIAL ANCHORING — same as all other rooms: "
        f"all wood in {palette['wood']}, all metal in {palette['metal']}, "
        f"textiles in {palette['textiles']}. "
        f"Soft shadows matching {light}, falling {shadow}. "
        f"Consistent shadows, global illumination, ambient occlusion, "
        f"depth of field f/8, ray tracing, HDR. "
        f"Professional real estate photography, 24mm wide angle lens, "
        f"cinematic warm lighting, balanced exposure, highly detailed, 8k."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STAGED PHOTOS
# ═══════════════════════════════════════════════════════════════════════════════

async def generate_staged_photos(photos: list, analysis: dict) -> list:
    stanze = analysis.get("stanze", [])
    loop   = asyncio.get_running_loop()
    all_futures = []

    # Colore parete globale: preferisce il valore globale, fallback per stanza
    wall_color_global  = analysis.get("wall_color_global", "")
    wall_finish_global = analysis.get("wall_finish_global", "matte")

    for i, room in enumerate(stanze):
        idx         = room.get("indice_foto", i)
        photo_bytes = photos[idx]["content"] if idx < len(photos) else None
        if not photo_bytes:
            continue

        # Colore parete: usa globale se disponibile
        style   = analysis.get("global_style_profile", {}).get("style", "")
        palette = _get_palette(style or "")
        wc  = wall_color_global  or palette["wall_color"]
        wf  = wall_finish_global or palette["wall_finish"]

        # Costruzione prompt con template Python (non Gemini)
        d_p1 = _build_d_stage1(room, palette, wc, wf)
        d_p2 = _build_d_stage2(room, palette, wc, wf)
        e_p1 = _build_e_stage1(room, palette, wc, wf)
        e_p2 = _build_e_stage2(room, palette, wc, wf)

        cwc        = room.get("current_wall_color", "")
        is_kitchen = room.get("is_kitchen", False)

        all_futures.append(("D_FULL_SMART", i,
            loop.run_in_executor(_imagen_executor, _approach_D_two_stage,
                photo_bytes, d_p1, d_p2,
                GUIDANCE["D_STAGE1_CLEAN"], GUIDANCE["D_STAGE2_STAGE"])
        ))
        all_futures.append(("E_WALL_FORCE", i,
            loop.run_in_executor(_imagen_executor, _approach_E_two_stage,
                photo_bytes, e_p1, e_p2,
                GUIDANCE["E_STAGE1_WALL"], GUIDANCE["E_STAGE2_STAGE"],
                cwc, is_kitchen)
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


# ── D two-stage (INVARIATA) ───────────────────────────────────────────────────

def _approach_D_two_stage(photo_bytes: bytes,
                           p1: str, p2: str,
                           g1: int, g2: int) -> str | None:
    try:
        compressed = compress_image(photo_bytes, max_width=1024, quality=80)
        print(f"[D Stage1] {len(compressed)//1024}KB  g={g1}")
        client = _get_vertex_client()

        r1 = client.models.edit_image(
            model="imagen-3.0-capability-001",
            prompt=p1,
            reference_images=[genai_types.RawReferenceImage(
                reference_id=1,
                reference_image=genai_types.Image(image_bytes=compressed),
            )],
            config=genai_types.EditImageConfig(
                edit_mode="EDIT_MODE_DEFAULT", number_of_images=1,
                guidance_scale=float(g1),
                negative_prompt=(
                    "furniture, objects, clutter, distorted architecture, "
                    "watermark, low quality, unrealistic"
                ),
                safety_filter_level="block_only_high",
            ),
        )
        s1 = r1.generated_images[0].image.image_bytes if r1.generated_images \
             else compress_image(photo_bytes, max_width=1024, quality=80)
        print(f"[D Stage1] {'OK' if r1.generated_images else 'fallback'} → Stage2 g={g2}")

        r2 = client.models.edit_image(
            model="imagen-3.0-capability-001",
            prompt=p2,
            reference_images=[genai_types.RawReferenceImage(
                reference_id=1, reference_image=genai_types.Image(image_bytes=s1),
            )],
            config=genai_types.EditImageConfig(
                edit_mode="EDIT_MODE_DEFAULT", number_of_images=1,
                guidance_scale=float(g2),
                negative_prompt=(
                    "empty room, bare walls, clutter, cartoon, flat lighting, "
                    "floating objects, sticker effect, inconsistent shadows, "
                    "distorted architecture, watermark, low quality"
                ),
                safety_filter_level="block_only_high",
            ),
        )
        if r2.generated_images:
            print("[D Stage2] SUCCESS")
            return base64.b64encode(r2.generated_images[0].image.image_bytes).decode()
        print(f"[D Stage2] 0 immagini")
        return None
    except Exception as e:
        print(f"[D] ERRORE: {e}\n{traceback.format_exc()}")
        return None


# ── E two-stage (Wall Force + Kitchen) ───────────────────────────────────────

def _approach_E_two_stage(photo_bytes: bytes,
                           p1: str, p2: str,
                           g1: int, g2: int,
                           current_wall_color: str = "",
                           is_kitchen: bool = False) -> str | None:
    try:
        compressed = compress_image(photo_bytes, max_width=1024, quality=80)
        label = "E[kitchen]" if is_kitchen else "E[room]"
        print(f"[{label} Stage1] {len(compressed)//1024}KB  g={g1}")
        client = _get_vertex_client()

        neg1 = (
            "furniture, objects, clutter, distorted architecture, watermark, low quality, "
            "original wall color, bleed-through, transparent paint, "
            "previous color showing through, dirty walls"
        )
        if current_wall_color:
            neg1 += f", {current_wall_color} tint on walls"
        if is_kitchen:
            neg1 += ", beige cabinets, cream cabinets, old cabinet finish, original cabinet color"

        r1 = client.models.edit_image(
            model="imagen-3.0-capability-001",
            prompt=p1,
            reference_images=[genai_types.RawReferenceImage(
                reference_id=1,
                reference_image=genai_types.Image(image_bytes=compressed),
            )],
            config=genai_types.EditImageConfig(
                edit_mode="EDIT_MODE_DEFAULT", number_of_images=1,
                guidance_scale=float(g1),
                negative_prompt=neg1,
                safety_filter_level="block_only_high",
            ),
        )
        s1 = r1.generated_images[0].image.image_bytes if r1.generated_images \
             else compress_image(photo_bytes, max_width=1024, quality=80)
        print(f"[{label} Stage1] {'OK' if r1.generated_images else 'fallback'} → Stage2 g={g2}")

        neg2 = (
            "empty room, bare walls, clutter, cartoon, flat lighting, floating objects, "
            "sticker effect, inconsistent shadows, distorted architecture, watermark, "
            "original wall color retained, bleed-through, yellowish tint"
        )
        if current_wall_color:
            neg2 += f", {current_wall_color} on walls"
        if is_kitchen:
            neg2 += ", beige cabinets, old cabinet color, original kitchen appearance"

        r2 = client.models.edit_image(
            model="imagen-3.0-capability-001",
            prompt=p2,
            reference_images=[genai_types.RawReferenceImage(
                reference_id=1, reference_image=genai_types.Image(image_bytes=s1),
            )],
            config=genai_types.EditImageConfig(
                edit_mode="EDIT_MODE_DEFAULT", number_of_images=1,
                guidance_scale=float(g2),
                negative_prompt=neg2,
                safety_filter_level="block_only_high",
            ),
        )
        if r2.generated_images:
            print(f"[{label} Stage2] SUCCESS")
            return base64.b64encode(r2.generated_images[0].image.image_bytes).decode()
        print(f"[{label} Stage2] 0 immagini")
        return None
    except Exception as e:
        print(f"[E] ERRORE: {e}\n{traceback.format_exc()}")
        return None
