"""
AI Service v21 — The Debug Edition

Fix rispetto a v20:
  A. Diagnostica profonda: _call_imagen stampa traceback COMPLETO + PROJECT_ID/LOCATION
     → vedi se l'errore è 403 (credenziali) o 429 (quota) nei log Cloud Run
  B. Elaborazione sequenziale pura: BATCH=1, max_workers=1 per Imagen
     → una sola chiamata Imagen alla volta, elimina saturazione quota
  C. Type-safety: ogni variabile stringa nei builder è wrapped in _s()
     → evita KeyError/AttributeError se Gemini restituisce null o lista
  D. Outdoor hardened: negative prompt più esteso + "Maintain 100% original sky"
  E. Verifica credenziali al primo avvio con print esplicito
"""
import asyncio
import base64
import hashlib
import io
import json
import os
import re
import traceback as tb_module
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

print(f"[startup] PROJECT_ID   = {PROJECT_ID!r}  {'OK' if PROJECT_ID else '*** MANCANTE ***'}")
print(f"[startup] LOCATION     = {LOCATION!r}")
print(f"[startup] GEMINI_API_KEY = {'SET' if GEMINI_API_KEY else '*** MANCANTE ***'}")

GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
)

# ── Guidance ──────────────────────────────────────────────────────────────────
GUIDANCE = {
    "D_STAGE1_CLEAN":   8,
    "D_STAGE2_STAGE":  10,
    "E_STAGE1_WALL":   14,
    "E_STAGE2_STAGE":  10,
    "OUTDOOR_STAGE1":   6,
    "OUTDOOR_STAGE2":   8,
}

# ── Concurrency: 1 worker, 1 batch → SEQUENZIALE PURO ────────────────────────
# Questo elimina definitivamente qualsiasi problema di quota/rate-limit.
# Più lento, ma diagnosticamente certo.
_imagen_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="imagen")
_gemini_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="gemini")

IMAGEN_BATCH_SIZE = 1   # una stanza alla volta

# ── Palette ───────────────────────────────────────────────────────────────────
STYLE_PALETTES = {
    "Scandinavian": {
        "wall_color": "warm white", "wall_finish": "matte",
        "wall_forbidden": "orange, terracotta, red, electric blue, yellow",
        "wood": "Light Oak", "metal": "Matte Black",
        "textiles": "natural linen, soft grey cotton",
        "kitchen_cabinet": "matte white", "kitchen_counter": "white quartz",
        "kitchen_accent": "Matte Black pendant lamp, potted herbs in white ceramic",
    },
    "Industrial": {
        "wall_color": "warm off-white", "wall_finish": "matte concrete-effect",
        "wall_forbidden": "pink, pastel, bright orange, yellow",
        "wood": "Reclaimed Dark Wood", "metal": "Dark Steel",
        "textiles": "dark grey linen, charcoal canvas",
        "kitchen_cabinet": "matte charcoal grey",
        "kitchen_counter": "dark concrete-effect laminate",
        "kitchen_accent": "Dark Steel Edison pendant lamp, terracotta herb pots",
    },
    "Japandi": {
        "wall_color": "warm greige", "wall_finish": "matte clay",
        "wall_forbidden": "saturated colors, neon, orange, electric blue",
        "wood": "Blonde Bamboo", "metal": "Brushed Brass",
        "textiles": "warm linen, undyed cotton",
        "kitchen_cabinet": "warm linen white", "kitchen_counter": "natural light stone",
        "kitchen_accent": "Brushed Brass pendant lamp, ceramic herb pots",
    },
    "Minimalista": {
        "wall_color": "light greige", "wall_finish": "matte",
        "wall_forbidden": "saturated colors, orange, red, dark colors",
        "wood": "Light Natural Wood", "metal": "Matte Black",
        "textiles": "natural linen, soft white, neutral grey",
        "kitchen_cabinet": "matte white", "kitchen_counter": "light natural stone",
        "kitchen_accent": "Matte Black pendant lamp, minimalist ceramic vases",
    },
    "Mid-Century Modern": {
        "wall_color": "warm ivory", "wall_finish": "matte",
        "wall_forbidden": "cold grey, cold white, neon, electric blue",
        "wood": "Walnut Wood", "metal": "Brushed Gold",
        "textiles": "mustard yellow, terracotta, olive green, warm beige",
        "kitchen_cabinet": "sage green matte", "kitchen_counter": "white marble",
        "kitchen_accent": "Brushed Gold pendant lamp, retro ceramics",
    },
    "Boho Chic": {
        "wall_color": "warm sand", "wall_finish": "textured matte",
        "wall_forbidden": "cold grey, electric blue, neon",
        "wood": "Natural Rattan and Teak", "metal": "Brushed Copper",
        "textiles": "terracotta, rust, warm beige, macrame",
        "kitchen_cabinet": "cream white", "kitchen_counter": "warm wood butcher block",
        "kitchen_accent": "Brushed Copper pendant lamp, woven baskets",
    },
}

OUTDOOR_ROOM_TYPES = {"balcone", "terrazzo", "esterno", "giardino",
                       "balcony", "terrace", "outdoor", "loggia"}


def _get_palette(style: str) -> dict:
    key = (style or "").strip()
    for k in STYLE_PALETTES:
        if k.lower() in key.lower() or key.lower() in k.lower():
            return STYLE_PALETTES[k]
    kl = key.lower()
    if "scandi" in kl:                        return STYLE_PALETTES["Scandinavian"]
    if "industri" in kl:                      return STYLE_PALETTES["Industrial"]
    if "japandi" in kl or "japan" in kl:      return STYLE_PALETTES["Japandi"]
    if "minimal" in kl:                       return STYLE_PALETTES["Minimalista"]
    if "mid" in kl and "century" in kl:       return STYLE_PALETTES["Mid-Century Modern"]
    if "boho" in kl:                          return STYLE_PALETTES["Boho Chic"]
    return {
        "wall_color": "light greige", "wall_finish": "matte",
        "wall_forbidden": "saturated colors",
        "wood": "Light Natural Wood", "metal": "Matte Black",
        "textiles": "natural linen, neutral grey",
        "kitchen_cabinet": "matte white", "kitchen_counter": "light natural stone",
        "kitchen_accent": "Matte Black pendant lamp",
    }


def _s(val, fallback: str = "neutral") -> str:
    """Type-safe string: converte qualsiasi valore in stringa pulita."""
    if val is None:
        return fallback
    if isinstance(val, list):
        return ", ".join(str(v) for v in val) if val else fallback
    return str(val).strip() or fallback


_vertex_client = None
_analysis_cache: dict[str, dict] = {}


def _get_vertex_client():
    global _vertex_client
    if _vertex_client is None:
        print(f"[Imagen] init Vertex AI — project={PROJECT_ID!r} location={LOCATION!r}")
        if not PROJECT_ID:
            raise RuntimeError("GCP_PROJECT_ID non impostato — Imagen non può funzionare")
        _vertex_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
        print("[Imagen] client Vertex AI creato OK")
    return _vertex_client


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
        f"Analizza {n} foto per AI home staging. Per ogni foto: luminosità, inquadratura, "
        "tipo stanza, is_outdoor (balcone/terrazzo), elementi strutturali fissi.\n"
        'JSON: {"photos":[{"index":0,"valid":true,"room_type":"soggiorno","is_outdoor":false,'
        '"issue":"ok","suggestion":"","structural_elements":["frigo angolo nord-est"]}],'
        '"global_warnings":[],"layout_hint":""}'
    )})
    payload = {
        "system_instruction": {"parts": [{"text":
            "Esperto fotografia immobiliare. SOLO JSON valido senza markdown."
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
        pds = parsed.get("photos", [])
        print(f"[PhotoValidator] {sum(p.get('valid',True) for p in pds)}/{n} valide")
        return {
            "valid":      [p.get("valid", True) for p in pds],
            "issues":     [p.get("issue", "ok") for p in pds],
            "room_types": [p.get("room_type", "unknown") for p in pds],
            "is_outdoor": [p.get("is_outdoor", False) for p in pds],
            "suggestions":[p.get("suggestion", "") for p in pds],
            "structural": [p.get("structural_elements", []) for p in pds],
            "warnings":   parsed.get("global_warnings", []),
            "layout_hint":parsed.get("layout_hint", ""),
            "all_valid":  all(p.get("valid", True) for p in pds),
        }
    except Exception as e:
        print(f"[PhotoValidator] ERRORE: {e} — procedo comunque")
        return {"valid":[True]*n, "issues":["skip"]*n, "room_types":["unknown"]*n,
                "is_outdoor":[False]*n, "suggestions":[""]*n,
                "structural":[[]]*n, "warnings":[], "layout_hint":"", "all_valid":True}


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
# Gemini Analysis
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
    result["_prefs_style"] = prefs.get("style", "")
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
    foto_list = "\n".join(f"  FOTO {i} = stanza {i+1}" for i in range(n))

    system_instruction = (
        "Sei un Architetto Digitale per home staging. "
        "Restituisci SOLO variabili richieste — NON scrivere prompt Imagen. "
        f"Palette {style}: pareti={palette['wall_color']} {palette['wall_finish']}. "
        f"Vietato: {palette['wall_forbidden']}. "
        f"Legno={palette['wood']}, Metallo={palette['metal']}. "
        "STRUCTURAL_FIXED: frigorifero, armadi a muro, finestre — mai rimuovere. "
        "Balconi: is_outdoor=true, NON citare pareti. "
        "IMPORTANTE: restituisci TUTTI i valori come stringhe semplici, mai come liste. "
        "Rispondi ESCLUSIVAMENTE con JSON valido senza markdown."
    )

    prompt = (
        f"Analizza {n} foto per home staging {dest_label}.\n"
        f"Budget €{budget} | Stile: {style} | Città: {location}\n"
        f"Budget: Arredi €{alloc['arredi']} | Tinteggiatura €{alloc['tinteggiatura']} | "
        f"Materiali €{alloc['materiali']} | Montaggio €{alloc['montaggio']}\n\n"
        f"FOTO:\n{foto_list}\n\n"
        "Per ogni foto estrai:\n"
        "  room_type: soggiorno|cucina|camera|bagno|balcone|altro\n"
        "  is_kitchen: true/false\n"
        "  is_outdoor: true se balcone/terrazzo/esterno\n"
        "  current_wall_color: stringa (es. 'giallo paglierino')\n"
        "  light_source: stringa (es. 'finestra sinistra')\n"
        "  shadow_direction: stringa (es. 'verso destra')\n"
        "  floor_material: stringa (es. 'laminato chiaro')\n"
        "  structural_fixed: lista stringhe 'nome|posizione'\n"
        "    Cucine: SEMPRE includi 'frigorifero|posizione visibile'\n"
        "    Camere: armadio a muro -> segnala che blocca letto\n"
        "  furniture_to_replace: lista stringhe 'vecchio|nuovo|stessa_posizione'\n"
        "    Cucine: mantieni tavolo nella stessa posizione\n"
        "  layering_add: lista stringhe, max 4 voci\n"
        "    Balconi: solo sedie, tavolino, piante, tappeto esterno\n"
        "    Cucine: solo lampada sospensione, piante aromatiche\n"
        "  kitchen_vars: null oppure {\"original_cabinet_color\":\"beige\",\"original_cabinet_texture\":\"\"}\n"
        "  outdoor_floor: null o stringa\n"
        "  outdoor_view: null o stringa\n\n"
        "Genera interventi e costi per ogni stanza.\n\n"
        f"JSON ({n} oggetti in stanze):\n"
        "{{\n"
        "  \"wall_color_global\": \"stringa\",\n"
        "  \"wall_finish_global\": \"stringa\",\n"
        "  \"valutazione_generale\": \"analisi\",\n"
        "  \"punti_di_forza\": [\"p1\"],\n"
        "  \"criticita\": [\"c1\"],\n"
        f"  \"potenziale_str\": \"potenziale a {location}\",\n"
        "  \"tariffe\": {{\"attuale_notte\":\"\u20acXX\",\"post_restyling_notte\":\"\u20acXX\",\"incremento_percentuale\":\"XX%\"}},\n"
        f"  \"stanze\": [\n"
        "    {{\n"
        "      \"nome\": \"Nome\", \"indice_foto\": 0,\n"
        "      \"room_type\": \"soggiorno\", \"is_kitchen\": false, \"is_outdoor\": false,\n"
        "      \"current_wall_color\": \"giallo paglierino\",\n"
        "      \"light_source\": \"finestra sinistra\", \"shadow_direction\": \"verso destra\",\n"
        "      \"floor_material\": \"laminato chiaro\",\n"
        "      \"structural_fixed\": [\"armadio|parete ovest\"],\n"
        "      \"furniture_to_replace\": [\"divano rosso|divano grigio|stessa posizione\"],\n"
        "      \"layering_add\": [\"tappeto neutro\", \"cuscini lino\", \"lampada stelo\"],\n"
        "      \"kitchen_vars\": null, \"outdoor_floor\": null, \"outdoor_view\": null,\n"
        "      \"stato_attuale\": \"descrizione\",\n"
        "      \"interventi\": [{{"
        "\"titolo\":\"nome\",\"dettaglio\":\"dettaglio\","
        "\"costo_min\":50,\"costo_max\":100,\"priorita\":\"alta\",\"dove_comprare\":\"negozio\""
        "}}],\n"
        "      \"costo_totale_stanza\": 350\n"
        "    }}\n"
        "  ],\n"
        "  \"riepilogo_costi\": {{\"manodopera_tinteggiatura\":0,\"materiali_pittura\":0,"
        "\"arredi_complementi\":0,\"montaggio_varie\":0,\"totale\":0,\"budget_residuo\":0,\"nota_budget\":\"\"}},\n"
        "  \"piano_acquisti\": [{{\"categoria\":\"Arredi\",\"articoli\":[\"item\"],"
        "\"budget_stimato\":0,\"negozi_consigliati\":\"IKEA\"}}],\n"
        "  \"titolo_annuncio_suggerito\": \"Titolo\",\n"
        "  \"highlights_str\": [\"h1\"],\n"
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

    resp = httpx.post(GEMINI_URL,
                      json={"system_instruction": {"parts": [{"text": system_instruction}]},
                            "contents": [{"role": "user", "parts": parts}],
                            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 65536,
                                                 "responseMimeType": "application/json"}},
                      timeout=180.0, headers={"Content-Type": "application/json"})
    resp.raise_for_status()
    text   = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    result = _extract_json(text)
    print(f"[Gemini] stanze={len(result.get('stanze',[]))}/{n} "
          f"wall_color_global={result.get('wall_color_global','?')!r}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Template builders — type-safe con _s()
# ═══════════════════════════════════════════════════════════════════════════════

def _is_outdoor(room: dict) -> bool:
    rt = _s(room.get("room_type"), "").lower()
    return bool(room.get("is_outdoor", False)) or rt in OUTDOOR_ROOM_TYPES


def _structural_clause(structural_fixed, is_kitchen: bool) -> str:
    items = list(structural_fixed or [])
    if is_kitchen:
        fridge_present = any("frigo" in _s(s).lower() or "refriger" in _s(s).lower()
                             for s in items)
        if not fridge_present:
            items.insert(0, "frigorifero|current position")
    if not items:
        return ""
    parts_out = []
    for item in items[:4]:
        item_s = _s(item)
        name   = item_s.split("|")[0].strip()
        pos    = item_s.split("|")[1].strip() if "|" in item_s else "its position"
        parts_out.append(f"Keep {name} at {pos} unchanged.")
    return " ".join(parts_out)


def _repl_text(replacements, wood: str) -> str:
    out = []
    for r in (replacements or [])[:2]:
        r_s = _s(r)
        p   = r_s.split("|")
        if len(p) >= 3:
            old, new, pos = p[0].strip(), p[1].strip(), p[2].strip()
            out.append(f"Replace {old} with {new} in {wood}, same spot.")
    return " ".join(out)


def _build_d_stage1(room: dict, palette: dict, wc: str, wf: str) -> str:
    sf = _structural_clause(room.get("structural_fixed"), room.get("is_kitchen", False))
    fl = _s(room.get("floor_material"), "existing floor")
    lt = _s(room.get("light_source"), "natural light")
    return (
        f"Empty clean room, all furniture removed. "
        f"Walls repainted in {_s(wf)} {_s(wc)}. "
        f"Keep {fl} floor. {sf} "
        f"Consistent light from {lt}. Photorealistic interior, 8k."
    )


def _build_d_stage2(room: dict, palette: dict, wc: str, wf: str) -> str:
    sf    = _structural_clause(room.get("structural_fixed"), room.get("is_kitchen", False))
    repl  = _repl_text(room.get("furniture_to_replace"), palette["wood"])
    layer = _s(", ".join((room.get("layering_add") or [])[:4]), "area rug, cushions, plant")
    lt    = _s(room.get("light_source"), "natural light")
    sh    = _s(room.get("shadow_direction"), "soft shadows")
    return (
        f"{_s(wf)} {_s(wc)} walls. {sf} {repl} "
        f"Add: {layer}. "
        f"Wood: {palette['wood']}, Metal: {palette['metal']}, Textiles: {palette['textiles']}. "
        f"Light from {lt}, shadows {sh}. "
        f"Global illumination, ambient occlusion, depth of field f/8. "
        f"Professional real estate photo, 24mm, cinematic warm light, 8k."
    )


def _build_e_stage1(room: dict, palette: dict, wc: str, wf: str) -> str:
    sf    = _structural_clause(room.get("structural_fixed"), room.get("is_kitchen", False))
    cur   = _s(room.get("current_wall_color"), "original color")
    lt    = _s(room.get("light_source"), "natural light")
    fl    = _s(room.get("floor_material"), "existing floor")

    kitchen_part = ""
    if room.get("is_kitchen", False):
        kv      = room.get("kitchen_vars") or {}
        cab_col = _s(kv.get("original_cabinet_color"), "beige")
        kitchen_part = (
            f"Repaint kitchen cabinet fronts in {palette['kitchen_cabinet']} — "
            f"cover all {cab_col} surfaces. "
            f"Change countertop to {palette['kitchen_counter']}. "
        )

    return (
        f"Renovated room. Replace {cur} walls with {_s(wf)} {_s(wc)} paint, full coverage. "
        f"{kitchen_part}{sf} "
        f"Keep {fl} floor. Remove clutter and furniture. "
        f"Light from {lt}. Photorealistic, 8k."
    )


def _build_e_stage2(room: dict, palette: dict, wc: str, wf: str) -> str:
    sf    = _structural_clause(room.get("structural_fixed"), room.get("is_kitchen", False))
    cur   = _s(room.get("current_wall_color"), "original color")
    lt    = _s(room.get("light_source"), "natural light")
    sh    = _s(room.get("shadow_direction"), "soft shadows")

    kitchen_layer = ""
    if room.get("is_kitchen", False):
        kv      = room.get("kitchen_vars") or {}
        cab_col = _s(kv.get("original_cabinet_color"), "beige")
        kitchen_layer = (
            f"Kitchen has {palette['kitchen_cabinet']} cabinets, no {cab_col} visible. "
            f"Add {palette['kitchen_accent']}. Keep dining table in same position. "
        )
        repl = ""
    else:
        repl = _repl_text(room.get("furniture_to_replace"), palette["wood"])

    layer = _s(", ".join((room.get("layering_add") or [])[:4]), "area rug, cushions, plant")
    return (
        f"Newly painted {_s(wf)} {_s(wc)} walls, no {cur} remaining. {sf} "
        f"{kitchen_layer}{repl}"
        f"Add: {layer}. "
        f"Wood: {palette['wood']}, Metal: {palette['metal']}, Textiles: {palette['textiles']}. "
        f"Light from {lt}, shadows {sh}. "
        f"Global illumination, ambient occlusion, depth of field f/8, HDR. "
        f"Professional real estate photo, 24mm, cinematic warm light, 8k."
    )


def _build_outdoor_stage1(room: dict, palette: dict) -> str:
    fl   = _s(room.get("outdoor_floor") or room.get("floor_material"), "existing floor tiles")
    view = _s(room.get("outdoor_view"), "urban city view")
    return (
        f"Outdoor balcony photo editing. "
        f"Maintain 100% of the original sky and background {view} — do not alter sky or horizon. "
        f"Remove only: drying racks, old furniture, scattered objects. "
        f"Keep {fl} floor and building railings unchanged. "
        f"This is an open exterior space — no walls, no ceiling, no indoor elements added. "
        f"Bright natural daylight. Photorealistic exterior photo, 8k."
    )


def _build_outdoor_stage2(room: dict, palette: dict) -> str:
    layer = _s(", ".join((room.get("layering_add") or [])[:4]),
               "2 folding chairs, small round table, potted plants, outdoor rug")
    fl    = _s(room.get("outdoor_floor") or room.get("floor_material"), "existing tiles")
    view  = _s(room.get("outdoor_view"), "city view")
    return (
        f"Staged outdoor balcony. Maintain 100% original sky and {view} unchanged. "
        f"Keep {fl} floor and metal railings unchanged. "
        f"Add tasteful outdoor furniture: {layer}. "
        f"Wood: {palette['wood']}, Metal: {palette['metal']}. "
        f"No walls, no ceiling, no interior elements — purely exterior space. "
        f"Bright natural daylight. Professional exterior photo, wide angle, 8k."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STAGED PHOTOS — sequenziale puro (BATCH=1, 1 worker)
# ═══════════════════════════════════════════════════════════════════════════════

async def generate_staged_photos(photos: list, analysis: dict) -> list:
    """
    v21: elaborazione completamente sequenziale.
    Una stanza alla volta, una variante alla volta.
    Motivo: elimina ogni possibile problema di quota/rate-limit.
    """
    stanze  = analysis.get("stanze", [])
    loop    = asyncio.get_running_loop()
    style   = _s(analysis.get("_prefs_style") or
                 analysis.get("global_style_profile", {}).get("style"), "")
    palette = _get_palette(style)
    wc      = _s(analysis.get("wall_color_global"), palette["wall_color"])
    wf      = _s(analysis.get("wall_finish_global"), palette["wall_finish"])

    results = [{} for _ in stanze]

    for i, room in enumerate(stanze):
        idx = room.get("indice_foto", i)
        pb  = photos[idx]["content"] if idx < len(photos) else None
        if not pb:
            print(f"[stanza {i}] foto non trovata (indice_foto={idx})")
            continue

        outdoor    = _is_outdoor(room)
        is_kitchen = bool(room.get("is_kitchen", False))
        cwc        = _s(room.get("current_wall_color"), "")

        print(f"\n[stanza {i}] '{room.get('nome','?')}' outdoor={outdoor} kitchen={is_kitchen}")

        if outdoor:
            p1 = _build_outdoor_stage1(room, palette)
            p2 = _build_outdoor_stage2(room, palette)
            for variant in ("D_FULL_SMART", "E_WALL_FORCE"):
                print(f"  [{variant}] → outdoor")
                res = await loop.run_in_executor(
                    _imagen_executor, _approach_outdoor, pb, p1, p2,
                    GUIDANCE["OUTDOOR_STAGE1"], GUIDANCE["OUTDOOR_STAGE2"]
                )
                results[i][variant] = res
        else:
            # D
            d1 = _build_d_stage1(room, palette, wc, wf)
            d2 = _build_d_stage2(room, palette, wc, wf)
            print(f"  [D] S1: {d1[:80]!r}")
            res_d = await loop.run_in_executor(
                _imagen_executor, _approach_two_stage, pb, d1, d2,
                GUIDANCE["D_STAGE1_CLEAN"], GUIDANCE["D_STAGE2_STAGE"],
                "D", cwc, is_kitchen
            )
            results[i]["D_FULL_SMART"] = res_d

            # E (dopo D — sequenziale)
            e1 = _build_e_stage1(room, palette, wc, wf)
            e2 = _build_e_stage2(room, palette, wc, wf)
            print(f"  [E] S1: {e1[:80]!r}")
            res_e = await loop.run_in_executor(
                _imagen_executor, _approach_two_stage, pb, e1, e2,
                GUIDANCE["E_STAGE1_WALL"], GUIDANCE["E_STAGE2_STAGE"],
                "E", cwc, is_kitchen
            )
            results[i]["E_WALL_FORCE"] = res_e

        d_ok = results[i].get("D_FULL_SMART") is not None
        e_ok = results[i].get("E_WALL_FORCE") is not None
        print(f"[stanza {i}] completata — D={'OK' if d_ok else 'FAIL'} E={'OK' if e_ok else 'FAIL'}")

    return results


# ── Core: singola chiamata Imagen con traceback completo ──────────────────────

def _call_imagen(label: str, compressed: bytes, prompt: str,
                 guidance: float, negative: str) -> bytes | None:
    """
    Chiamata Imagen con diagnostica COMPLETA.
    Stampa PROJECT_ID, LOCATION, tipo errore e traceback intero.
    Questo permette di vedere nei log Cloud Run se l'errore è:
      - 403 PERMISSION_DENIED → manca ruolo "Vertex AI User" sul service account
      - 429 RESOURCE_EXHAUSTED → superata quota API
      - altro errore SDK
    """
    # Verifica credenziali prima di chiamare
    print(f"  [{label}] g={guidance} project={PROJECT_ID!r} location={LOCATION!r} "
          f"prompt_len={len(prompt)}")

    try:
        client = _get_vertex_client()
        resp   = client.models.edit_image(
            model="imagen-3.0-capability-001",
            prompt=prompt,
            reference_images=[genai_types.RawReferenceImage(
                reference_id=1,
                reference_image=genai_types.Image(image_bytes=compressed),
            )],
            config=genai_types.EditImageConfig(
                edit_mode="EDIT_MODE_DEFAULT",
                number_of_images=1,
                guidance_scale=float(guidance),
                negative_prompt=negative if negative else None,
                safety_filter_level="block_only_high",
            ),
        )
        if resp.generated_images:
            print(f"  [{label}] SUCCESS — immagine generata OK")
            return resp.generated_images[0].image.image_bytes

        print(f"  [{label}] 0 immagini (safety filter o guidance out-of-range?)")
        print(f"  [{label}] prompt full: {prompt!r}")
        return None

    except Exception as e:
        # TRACEBACK COMPLETO — essenziale per diagnosticare 403/429
        full_tb = tb_module.format_exc()
        print(f"  [{label}] *** ERRORE {type(e).__name__} ***")
        print(f"  [{label}] messaggio: {e}")
        print(f"  [{label}] traceback:\n{full_tb}")
        print(f"  [{label}] prompt: {prompt[:200]!r}")
        return None


def _approach_two_stage(photo_bytes: bytes,
                        p1: str, p2: str,
                        g1: int, g2: int,
                        label: str,
                        current_wall_color: str = "",
                        is_kitchen: bool = False) -> str | None:
    compressed = compress_image(photo_bytes, max_width=1024, quality=80)
    print(f"  [{label}] {len(compressed)//1024}KB")

    neg1 = "distorted architecture, watermark, low quality, unrealistic"
    if label == "E":
        neg1 += ", original wall color visible, paint bleed-through"
        cwc = _s(current_wall_color, "")
        if cwc:
            neg1 += f", {cwc} walls"
        if is_kitchen:
            neg1 += ", old cabinet color, beige kitchen cabinets"

    s1 = _call_imagen(f"{label}_S1", compressed, p1, g1, neg1)
    if not s1:
        print(f"  [{label}_S1] fallback su originale")
        s1 = compressed

    neg2 = "empty room, bare walls, cartoon, flat lighting, floating objects, watermark"
    if label == "E":
        neg2 += ", original wall color retained, paint bleed-through"
        cwc = _s(current_wall_color, "")
        if cwc:
            neg2 += f", {cwc}"
        if is_kitchen:
            neg2 += ", old cabinet color visible"

    s2 = _call_imagen(f"{label}_S2", s1, p2, g2, neg2)
    if s2:
        return base64.b64encode(s2).decode()
    return None


def _approach_outdoor(photo_bytes: bytes,
                      p1: str, p2: str,
                      g1: int, g2: int) -> str | None:
    compressed = compress_image(photo_bytes, max_width=1024, quality=80)
    print(f"  [OUTDOOR] {len(compressed)//1024}KB")

    neg = (
        "interior walls added, enclosed space, ceiling added, "
        "sky removed, sky replaced, horizon blocked, view blocked, "
        "indoor room, curtains, wallpaper, interior doors, indoor lighting, "
        "distorted architecture, watermark, low quality"
    )

    s1 = _call_imagen("OUT_S1", compressed, p1, g1, neg)
    if not s1:
        s1 = compressed
        print("  [OUTDOOR_S1] fallback su originale")

    s2 = _call_imagen("OUT_S2", s1, p2, g2, neg + ", clutter, drying rack, messy objects")
    if s2:
        return base64.b64encode(s2).decode()
    return None
