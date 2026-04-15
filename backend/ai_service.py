"""
AI Service v20 — Fix "Non disponibile" + Outdoor Protocol

Fix rispetto a v19:
  A. Outdoor Protocol: balcone/terrazzo usa template separato senza "walls"
  B. Guidance E: Stage1=14, Stage2=10 (16 produceva 0 immagini silenzioso)
  C. Concurrency: max_workers=3, esecuzione sequenziale a coppie (anti-quota)
  D. Prompt più corti e meno aggressivi (no "MUST", "ZERO") → meno safety blocks
  E. Fix lookup stile: recupero da prefs_style iniettato nell'analysis
  F. Fridge hardcoded in structural_fixed per cucine
  G. Log prompt primi 150 char quando 0 immagini → debug in Cloud Run logs
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

# ── Guidance — valori sicuri testati ──────────────────────────────────────────
# EDIT_MODE_DEFAULT: > ~15 può produrre 0 immagini silenzioso
GUIDANCE = {
    "D_STAGE1_CLEAN":   8,    # libertà massima per pulizia
    "D_STAGE2_STAGE":  10,    # layering moderato
    "E_STAGE1_WALL":   15,    # alzato a 15: forza cambio colore pareti
    "E_STAGE2_STAGE":  10,    # mantiene pareti, aggiunge arredo
    "OUTDOOR_STAGE1":   6,    # molto libero per spazi aperti
    "OUTDOOR_STAGE2":   8,    # aggiunge arredi esterni
}

# ── Concurrency — ridotto per evitare saturazione quota ──────────────────────
# Con 5 stanze × 2 varianti × 2 stage = 20 call simultanee → quota exceeded
# Con max_workers=3 e batching a 2 stanze per volta → ~6 call simultanee max
_imagen_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="imagen")
_gemini_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="gemini")

# ── Palette per stile ─────────────────────────────────────────────────────────
STYLE_PALETTES = {
    "Scandinavian": {
        "wall_color": "warm white",      "wall_finish": "matte",
        "wall_forbidden": "orange, terracotta, red, electric blue, yellow",
        "wood": "Light Oak",             "metal": "Matte Black",
        "textiles": "natural linen, soft grey cotton",
        "kitchen_cabinet": "matte white",
        "kitchen_counter": "white quartz",
        "kitchen_accent":  "Matte Black pendant lamp, potted herbs",
    },
    "Industrial": {
        "wall_color": "warm off-white",  "wall_finish": "matte concrete-effect",
        "wall_forbidden": "pink, pastel, bright orange, yellow",
        "wood": "Reclaimed Dark Wood",   "metal": "Dark Steel",
        "textiles": "dark grey linen, charcoal canvas",
        "kitchen_cabinet": "matte charcoal grey",
        "kitchen_counter": "dark concrete-effect laminate",
        "kitchen_accent":  "Dark Steel Edison pendant lamp, terracotta herb pots",
    },
    "Japandi": {
        "wall_color": "warm greige",     "wall_finish": "matte clay",
        "wall_forbidden": "saturated colors, neon, orange, electric blue",
        "wood": "Blonde Bamboo",         "metal": "Brushed Brass",
        "textiles": "warm linen, undyed cotton",
        "kitchen_cabinet": "warm linen white",
        "kitchen_counter": "natural light stone",
        "kitchen_accent":  "Brushed Brass pendant lamp, ceramic herb pots",
    },
    "Minimalista": {
        "wall_color": "light greige",    "wall_finish": "matte",
        "wall_forbidden": "saturated colors, orange, red, dark colors",
        "wood": "Light Natural Wood",    "metal": "Matte Black",
        "textiles": "natural linen, soft white, neutral grey",
        "kitchen_cabinet": "matte white",
        "kitchen_counter": "light natural stone",
        "kitchen_accent":  "Matte Black pendant lamp, minimalist ceramic vases",
    },
    "Mid-Century Modern": {
        "wall_color": "warm ivory",      "wall_finish": "matte",
        "wall_forbidden": "grey, cold white, neon, electric blue",
        "wood": "Walnut Wood",           "metal": "Brushed Gold",
        "textiles": "mustard yellow, terracotta, olive green, warm beige",
        "kitchen_cabinet": "sage green matte",
        "kitchen_counter": "white marble or butcher block",
        "kitchen_accent":  "Brushed Gold pendant lamp, retro ceramics",
    },
    "Boho Chic": {
        "wall_color": "warm sand",       "wall_finish": "textured matte",
        "wall_forbidden": "cold grey, electric blue, neon",
        "wood": "Natural Rattan and Teak",  "metal": "Brushed Copper",
        "textiles": "terracotta, rust, warm beige, macrame",
        "kitchen_cabinet": "cream white",
        "kitchen_counter": "warm wood butcher block",
        "kitchen_accent":  "Brushed Copper pendant lamp, woven baskets",
    },
}

OUTDOOR_ROOM_TYPES = {"balcone", "terrazzo", "esterno", "giardino", "loggia",
                       "balcony", "terrace", "outdoor", "garden"}


def _get_palette(style: str) -> dict:
    key = style.strip().title()
    for k in STYLE_PALETTES:
        if k.lower() in key.lower() or key.lower() in k.lower():
            return STYLE_PALETTES[k]
    # Normalizzazione comuni
    if "scandi" in key.lower():   return STYLE_PALETTES["Scandinavian"]
    if "industri" in key.lower(): return STYLE_PALETTES["Industrial"]
    if "japandi" in key.lower() or "japan" in key.lower(): return STYLE_PALETTES["Japandi"]
    if "minimal" in key.lower():  return STYLE_PALETTES["Minimalista"]
    if "mid" in key.lower() and "century" in key.lower(): return STYLE_PALETTES["Mid-Century Modern"]
    if "boho" in key.lower():     return STYLE_PALETTES["Boho Chic"]
    return {  # fallback neutro
        "wall_color": "light greige", "wall_finish": "matte",
        "wall_forbidden": "saturated colors",
        "wood": "Light Natural Wood", "metal": "Matte Black",
        "textiles": "natural linen, neutral grey",
        "kitchen_cabinet": "matte white", "kitchen_counter": "light natural stone",
        "kitchen_accent": "Matte Black pendant lamp",
    }


_vertex_client = None
_analysis_cache: dict[str, dict] = {}


def _get_vertex_client():
    global _vertex_client
    if _vertex_client is None:
        print(f"[Imagen] init client project={PROJECT_ID} location={LOCATION}")
        _vertex_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
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
        f"Analizza {n} foto per AI home staging. Per ogni foto:\n"
        "1. È sufficientemente illuminata?\n"
        "2. L'inquadratura mostra abbastanza della stanza?\n"
        "3. Tipo stanza (soggiorno/cucina/camera/bagno/balcone/altro)?\n"
        "4. Elementi strutturali fissi visibili (frigo, armadio, finestre)?\n"
        "5. È uno spazio esterno (balcone, terrazzo)?\n"
        'Restituisci JSON: {"photos":[{"index":0,"valid":true,"room_type":"soggiorno",'
        '"is_outdoor":false,"issue":"ok","suggestion":"",'
        '"structural_elements":["frigorifero angolo nord-est"]}],'
        '"global_warnings":[],"layout_hint":""}'
    )})
    payload = {
        "system_instruction": {"parts": [{"text":
            "Esperto di fotografia immobiliare. Rispondi SOLO con JSON valido."
        }]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 4096,
                             "responseMimeType": "application/json"}
    }
    try:
        resp = httpx.post(GEMINI_URL, json=payload, timeout=60.0,
                          headers={"Content-Type": "application/json"})
        resp.raise_for_status()
        parsed = _extract_json(
            resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        )
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
        print(f"[PhotoValidator] ERRORE: {e}")
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


def _sync_furniture_costs(analysis: dict, palette: dict) -> dict:
    """
    Allinea furniture_to_replace → interventi.
    Se Gemini ha messo un mobile in furniture_to_replace ma non ha
    un intervento corrispondente nel PDF, lo inietta automaticamente.
    Questo evita la discrepanza armadio-fantasma (visto nel render, non nel preventivo).
    """
    for room in analysis.get("stanze", []):
        replacements = room.get("furniture_to_replace") or []
        interventi   = room.get("interventi") or []
        # Testo degli interventi esistenti in minuscolo per confronto
        existing_text = " ".join(
            (iv.get("titolo", "") + " " + iv.get("dettaglio", "")).lower()
            for iv in interventi
        )
        for r in replacements:
            parts = r.split("|")
            if len(parts) < 2:
                continue
            old_item = parts[0].strip()
            new_item = parts[1].strip()
            # Controlla se è già menzionato negli interventi
            if old_item.lower() in existing_text or new_item.lower() in existing_text:
                continue
            # Stima costo grezzo basata sul tipo di mobile
            costo = 150  # default
            ol = old_item.lower()
            if any(k in ol for k in ("wardrobe", "armadio", "closet")):
                costo = 300
            elif any(k in ol for k in ("sofa", "divano", "couch")):
                costo = 250
            elif any(k in ol for k in ("bed", "letto")):
                costo = 200
            elif any(k in ol for k in ("table", "tavolo", "desk")):
                costo = 120
            elif any(k in ol for k in ("chair", "sedia")):
                costo = 60
            interventi.append({
                "titolo":       f"Sostituzione {old_item}",
                "dettaglio":    f"Sostituire {old_item} con {new_item} in {palette['wood']}",
                "costo_min":    int(costo * 0.8),
                "costo_max":    costo,
                "priorita":     "alta",
                "dove_comprare":"IKEA, JYSK",
            })
            room["costo_totale_stanza"] = room.get("costo_totale_stanza", 0) + costo
            print(f"[sync_costs] stanza '{room.get('nome','?')}': "
                  f"aggiunto intervento '{old_item}' €{int(costo*0.8)}–{costo}")
        room["interventi"] = interventi
    return analysis


    stanze = analysis.get("stanze", [])
    if len(stanze) != n_photos:
        print(f"[WARNING] Gemini: {len(stanze)} stanze vs {n_photos} foto")
    for i, room in enumerate(stanze):
        idx = room.get("indice_foto")
        if idx is None or not isinstance(idx, int) or idx >= n_photos:
            room["indice_foto"] = i
    return analysis


# ═══════════════════════════════════════════════════════════════════════════════
# Gemini Analysis — restituisce SOLO variabili
# ═══════════════════════════════════════════════════════════════════════════════

async def analyze_with_gemini(photos: list, prefs: dict) -> dict:
    key = _cache_key(photos, prefs)
    if key in _analysis_cache:
        print("[Gemini] cache hit")
        return _analysis_cache[key]
    loop   = asyncio.get_running_loop()
    result = await loop.run_in_executor(_gemini_executor, _gemini_sync, photos, prefs)
    result = _sync_furniture_costs(result, _get_palette(prefs.get("style", "")))
    result = validate_and_fix_costs(result, prefs["budget"])
    # Inietta stile nei metadata per recupero in generate_staged_photos
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
        "Restituisci SOLO le variabili richieste — NON scrivere prompt Imagen. "
        f"Palette autorizzata {style}: pareti={palette['wall_color']} {palette['wall_finish']}. "
        f"Vietato: {palette['wall_forbidden']}. "
        f"Legno={palette['wood']}, Metallo={palette['metal']}. "
        "STRUCTURAL_FIXED: frigorifero, armadi a muro, finestre — mai rimuovere. "
        "Per balconi/terrazzi: is_outdoor=true, NON menzionare pareti. "
        "CRITICO — LINGUA INGLESE OBBLIGATORIA: "
        "I campi current_wall_color, light_source, shadow_direction, floor_material, "
        "structural_fixed, furniture_to_replace, layering_add, outdoor_floor, "
        "outdoor_view e kitchen_vars DEVONO essere scritti SOLO IN INGLESE. "
        "Questi valori vengono inseriti direttamente in prompt per Imagen 3 che "
        "rifiuta prompt in lingue miste con errore 400. "
        "Esempi CORRETTI: 'left window' NON 'finestra sinistra', "
        "'light laminate floor' NON 'laminato chiaro', "
        "'red sofa' NON 'divano rosso', 'right wall' NON 'parete destra', "
        "'built-in wardrobe' NON 'armadio a muro', 'pale yellow' NON 'giallo paglierino'. "
        "Rispondi ESCLUSIVAMENTE con JSON valido senza markdown."
    )

    prompt = (
        f"Analizza {n} foto per home staging {dest_label}.\n"
        f"Budget €{budget} | Stile: {style} | Città: {location}\n"
        f"Budget: Arredi €{alloc['arredi']} | Tinteggiatura €{alloc['tinteggiatura']} | "
        f"Materiali €{alloc['materiali']} | Montaggio €{alloc['montaggio']}\n\n"
        f"FOTO:\n{foto_list}\n\n"
        f"Per ogni foto estrai (TUTTI I VALORI IN INGLESE):\n"
        f"  room_type: soggiorno|cucina|camera|bagno|balcone|altro\n"
        f"  is_kitchen: true/false\n"
        f"  is_outdoor: true se balcone/terrazzo/esterno\n"
        f"  current_wall_color: IN ENGLISH (es. 'pale yellow', 'sky blue', 'beige')\n"
        f"  light_source: IN ENGLISH (es. 'left window', 'right window', 'overhead')\n"
        f"  shadow_direction: IN ENGLISH (es. 'towards right', 'downward left')\n"
        f"  floor_material: IN ENGLISH (es. 'light laminate', 'terracotta tiles')\n"
        f"  structural_fixed: IN ENGLISH formato 'element|position'\n"
        f"    Cucine: SEMPRE includi 'refrigerator|current position'\n"
        f"    Camere: 'built-in wardrobe|entire west wall — do not place bed in front'\n"
        f"  furniture_to_replace: IN ENGLISH formato 'old item|new item|same position'\n"
        f"    Cucine: mantieni tavolo nella stessa posizione\n"
        f"  layering_add: IN ENGLISH, max 4 voci\n"
        f"    Balconi: solo sedie, tavolino, piante, tappeto esterno\n"
        f"    Cucine: solo pendant lamp, aromatic herbs\n"
        f"  kitchen_vars: null oppure {{\"original_cabinet_color\":\"beige\",\"original_cabinet_texture\":\"\"}}\n"
        f"  outdoor_floor: IN ENGLISH (es. 'red terracotta tiles')\n"
        f"  outdoor_view: IN ENGLISH (es. 'city rooftops and sky')\n\n"
        f"Genera anche interventi e costi (in italiano) per ogni stanza.\n\n"
        f"JSON da restituire ({n} oggetti in stanze):\n"
        "{{\n"
        "  \"wall_color_global\": \"colore unico per stanze interne\",\n"
        "  \"wall_finish_global\": \"finitura unica per stanze interne\",\n"
        "  \"valutazione_generale\": \"analisi complessiva\",\n"
        "  \"punti_di_forza\": [\"p1\"],\n"
        "  \"criticita\": [\"c1\"],\n"
        f"  \"potenziale_str\": \"potenziale a {location}\",\n"
        "  \"tariffe\": {{\n"
        "    \"attuale_notte\": \"\u20acXX-YY\",\n"
        "    \"post_restyling_notte\": \"\u20acXX-YY\",\n"
        "    \"incremento_percentuale\": \"XX%\"\n"
        "  }},\n"
        f"  \"stanze\": [\n"
        "    {{\n"
        "      \"nome\": \"Nome stanza\",\n"
        "      \"indice_foto\": 0,\n"
        "      \"room_type\": \"soggiorno\",\n"
        "      \"is_kitchen\": false,\n"
        "      \"is_outdoor\": false,\n"
        "      \"current_wall_color\": \"pale yellow\",\n"
        "      \"light_source\": \"left window\",\n"
        "      \"shadow_direction\": \"towards right\",\n"
        "      \"floor_material\": \"light laminate\",\n"
        "      \"structural_fixed\": [\"built-in wardrobe|entire west wall — no bed in front\", \"refrigerator|north-east corner\"],\n"
        "      \"furniture_to_replace\": [\"red sofa|grey linear sofa|same position\"],\n"
        "      \"layering_add\": [\"neutral area rug\", \"linen cushions\", \"floor lamp\", \"potted plant\"],\n"
        "      \"kitchen_vars\": null,\n"
        "      \"outdoor_floor\": null,\n"
        "      \"outdoor_view\": null,\n"
        "      \"stato_attuale\": \"descrizione\",\n"
        "      \"interventi\": [\n"
        "        {{\"titolo\": \"nome\", \"dettaglio\": \"dettaglio\","
        "\"costo_min\": 50, \"costo_max\": 100, \"priorita\": \"alta\","
        "\"dove_comprare\": \"negozio\"}}\n"
        "      ],\n"
        "      \"costo_totale_stanza\": 350\n"
        "    }}\n"
        "  ],\n"
        "  \"riepilogo_costi\": {{\"manodopera_tinteggiatura\":0,\"materiali_pittura\":0,"
        "\"arredi_complementi\":0,\"montaggio_varie\":0,\"totale\":0,"
        "\"budget_residuo\":0,\"nota_budget\":\"\"}},\n"
        "  \"piano_acquisti\": [{{\"categoria\":\"Arredi\",\"articoli\":[\"item\"],"
        "\"budget_stimato\":0,\"negozi_consigliati\":\"IKEA, H&M Home\"}}],\n"
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

    resp = httpx.post(GEMINI_URL,
                      json={"system_instruction": {"parts": [{"text": system_instruction}]},
                            "contents": [{"role": "user", "parts": parts}],
                            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 65536,
                                                 "responseMimeType": "application/json"}},
                      timeout=180.0, headers={"Content-Type": "application/json"})
    resp.raise_for_status()
    text   = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    result = _extract_json(text)
    # Inline: correggi indici stanza fuori range
    stanze = result.get("stanze", [])
    if len(stanze) != n:
        print(f"[WARNING] Gemini: {len(stanze)} stanze vs {n} foto")
    for i, room in enumerate(stanze):
        idx = room.get("indice_foto")
        if idx is None or not isinstance(idx, int) or idx >= n:
            room["indice_foto"] = i
    print(f"[Gemini] stanze={len(stanze)}/{n}  "
          f"wall_color_global={result.get('wall_color_global','?')!r}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Template builders — prompt corti e sicuri (no MUST/ZERO)
# ═══════════════════════════════════════════════════════════════════════════════

OUTDOOR_ROOM_TYPES = {"balcone", "terrazzo", "esterno", "giardino",
                       "balcony", "terrace", "outdoor", "loggia"}


def _is_outdoor(room: dict) -> bool:
    rt = (room.get("room_type") or "").lower()
    return room.get("is_outdoor", False) or rt in OUTDOOR_ROOM_TYPES


def _structural_clause(structural_fixed: list, is_kitchen: bool) -> str:
    """Genera clausole di protezione brevi per elementi strutturali."""
    items = list(structural_fixed or [])
    # Fridge hardcoded per cucine
    if is_kitchen:
        fridge_present = any("frigo" in s.lower() or "refriger" in s.lower()
                             for s in items)
        if not fridge_present:
            items.insert(0, "frigorifero|posizione attuale")

    if not items:
        return ""
    parts = []
    for item in items[:4]:  # max 4 per brevità
        name = item.split("|")[0].strip()
        pos  = item.split("|")[1].strip() if "|" in item else "its position"
        parts.append(f"Keep {name} at {pos} unchanged.")
    return " ".join(parts)


def _repl_text(replacements: list, wood: str) -> str:
    """Replacement logic: max 2 sostituzioni per brevità prompt."""
    out = []
    for r in (replacements or [])[:2]:
        p = r.split("|")
        if len(p) >= 3:
            old, new, pos = p[0].strip(), p[1].strip(), p[2].strip()
            out.append(f"Replace {old} with {new} in {wood}, same spot.")
    return " ".join(out)


# ── D templates ───────────────────────────────────────────────────────────────

def _build_d_stage1(room: dict, palette: dict, wc: str, wf: str) -> str:
    sf   = _structural_clause(room.get("structural_fixed", []), room.get("is_kitchen", False))
    fl   = room.get("floor_material", "existing floor")
    lt   = room.get("light_source", "natural light")
    return (
        f"Empty clean room, all furniture removed. "
        f"Walls repainted in {wf} {wc}. "
        f"Keep {fl} floor. {sf} "
        f"Consistent light from {lt}. Photorealistic interior, 8k."
    )


def _build_d_stage2(room: dict, palette: dict, wc: str, wf: str) -> str:
    sf    = _structural_clause(room.get("structural_fixed", []), room.get("is_kitchen", False))
    repl  = _repl_text(room.get("furniture_to_replace", []), palette["wood"])
    layer = ", ".join((room.get("layering_add") or [])[:4]) or "area rug, cushions, plant"
    lt    = room.get("light_source", "natural light")
    sh    = room.get("shadow_direction", "soft shadows")
    return (
        f"{wf} {wc} walls. {sf} {repl} "
        f"Add: {layer}. "
        f"Wood: {palette['wood']}, Metal: {palette['metal']}, "
        f"Textiles: {palette['textiles']}. "
        f"Light from {lt}, shadows {sh}. "
        f"Global illumination, ambient occlusion, depth of field f/8. "
        f"Professional real estate photo, 24mm, cinematic warm light, 8k."
    )


# ── E templates (wall force) ──────────────────────────────────────────────────

def _build_e_stage1(room: dict, palette: dict, wc: str, wf: str) -> str:
    sf    = _structural_clause(room.get("structural_fixed", []), room.get("is_kitchen", False))
    cur   = room.get("current_wall_color", "original color")
    lt    = room.get("light_source", "natural light")
    fl    = room.get("floor_material", "existing floor")

    kitchen_part = ""
    if room.get("is_kitchen", False):
        kv      = room.get("kitchen_vars") or {}
        cab_col = kv.get("original_cabinet_color", "beige")
        kitchen_part = (
            f"Repaint kitchen cabinet fronts in {palette['kitchen_cabinet']} — "
            f"cover all {cab_col} surfaces completely. "
            f"Change countertop to {palette['kitchen_counter']}. "
        )

    return (
        f"Architectural renovation. Brutal color replacement: completely sand and repaint "
        f"all {cur} wall surfaces with 3 coats of high-opacity {wf} {wc} paint. "
        f"Full wall coverage, no {cur} visible anywhere, achromatic base before new color. "
        f"{kitchen_part}{sf} "
        f"Keep {fl} floor. Remove all furniture and clutter. "
        f"Light from {lt}. Photorealistic, 8k."
    )


def _build_e_stage2(room: dict, palette: dict, wc: str, wf: str) -> str:
    sf    = _structural_clause(room.get("structural_fixed", []), room.get("is_kitchen", False))
    cur   = room.get("current_wall_color", "original color")
    lt    = room.get("light_source", "natural light")
    sh    = room.get("shadow_direction", "soft shadows")

    kitchen_layer = ""
    if room.get("is_kitchen", False):
        kv      = room.get("kitchen_vars") or {}
        cab_col = kv.get("original_cabinet_color", "beige")
        kitchen_layer = (
            f"Kitchen now has {palette['kitchen_cabinet']} cabinets, no {cab_col} visible. "
            f"Add {palette['kitchen_accent']}. "
            f"Keep dining table in same position. "
        )
        repl = ""
    else:
        repl = _repl_text(room.get("furniture_to_replace", []), palette["wood"])

    layer = ", ".join((room.get("layering_add") or [])[:4]) or "area rug, cushions, plant"

    return (
        f"Newly painted {wf} {wc} walls (no {cur} remaining). {sf} "
        f"{kitchen_layer}{repl}"
        f"Add: {layer}. "
        f"Wood: {palette['wood']}, Metal: {palette['metal']}, "
        f"Textiles: {palette['textiles']}. "
        f"Light from {lt}, shadows {sh}. "
        f"Global illumination, ambient occlusion, depth of field f/8, HDR. "
        f"Professional real estate photo, 24mm, cinematic warm light, 8k."
    )


# ── Outdoor templates (balcone/terrazzo) ──────────────────────────────────────

def _build_outdoor_stage1(room: dict, palette: dict) -> str:
    """Stage 1 outdoor: rimuove ingombri, mantiene cielo e vista, ripristina ringhiera."""
    fl   = room.get("outdoor_floor") or room.get("floor_material", "existing floor tiles")
    view = room.get("outdoor_view", "city view")
    metal = palette.get("metal", "Matte Black")
    return (
        f"Outdoor balcony staging. "
        f"Maintain 100% of the original sky and background {view} — do not alter sky or horizon. "
        f"Remove only: drying racks, old furniture, scattered objects, laundry. "
        f"Keep {fl} floor tiles unchanged. "
        f"Restore and clean all metal railings and balustrades — repaint with fresh {metal} finish, "
        f"remove rust and imperfections from railings. "
        f"This is an open exterior space — no walls, no ceiling, no indoor elements. "
        f"Bright natural daylight. Photorealistic exterior photo, 8k."
    )


def _build_outdoor_stage2(room: dict, palette: dict) -> str:
    """Stage 2 outdoor: aggiunge arredo esterno su spazio ripristinato."""
    layer = ", ".join((room.get("layering_add") or [])[:4]) or \
            "2 folding chairs, small round table, potted plants, outdoor rug"
    fl    = room.get("outdoor_floor") or room.get("floor_material", "existing tiles")
    view  = room.get("outdoor_view", "city view")
    metal = palette.get("metal", "Matte Black")
    wood  = palette.get("wood", "Light Natural Wood")
    return (
        f"Staged outdoor balcony. Maintain 100% original sky and {view} unchanged. "
        f"Keep {fl} floor and freshly painted {metal} railings unchanged. "
        f"Add tasteful outdoor furniture: {layer}. "
        f"Furniture in {wood} and {metal}. "
        f"No walls, no ceiling, no interior elements — purely exterior open space. "
        f"Bright natural daylight. Professional exterior photo, wide angle, 8k."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STAGED PHOTOS — batched sequentially to avoid quota saturation
# ═══════════════════════════════════════════════════════════════════════════════

async def generate_staged_photos(photos: list, analysis: dict) -> list:
    """
    Processa le stanze in batch sequenziali di 2 per evitare saturazione quota.
    Max 2 stanze × 2 varianti × 2 stage = 8 call per batch (con 3 workers).
    """
    stanze = analysis.get("stanze", [])
    loop   = asyncio.get_running_loop()

    # Fix lookup stile — v19 bug: cercava in global_style_profile ma non esiste
    style   = (analysis.get("_prefs_style") or
               analysis.get("global_style_profile", {}).get("style") or "")
    palette = _get_palette(style)
    wc      = analysis.get("wall_color_global") or palette["wall_color"]
    wf      = analysis.get("wall_finish_global") or palette["wall_finish"]

    results = [{} for _ in stanze]

    # Batch size 2: al massimo 2 stanze elaborate in parallelo alla volta
    BATCH = 2
    for batch_start in range(0, len(stanze), BATCH):
        batch_rooms = stanze[batch_start:batch_start + BATCH]
        batch_futures = []

        for i, room in enumerate(batch_rooms):
            room_idx = batch_start + i
            idx      = room.get("indice_foto", room_idx)
            pb       = photos[idx]["content"] if idx < len(photos) else None
            if not pb:
                continue

            outdoor = _is_outdoor(room)
            is_kitchen = room.get("is_kitchen", False)

            if outdoor:
                p1 = _build_outdoor_stage1(room, palette)
                p2 = _build_outdoor_stage2(room, palette)
                g1 = GUIDANCE["OUTDOOR_STAGE1"]
                g2 = GUIDANCE["OUTDOOR_STAGE2"]
                batch_futures.append(("D_FULL_SMART", room_idx,
                    loop.run_in_executor(_imagen_executor,
                        _approach_outdoor, pb, p1, p2, g1, g2)
                ))
                batch_futures.append(("E_WALL_FORCE", room_idx,
                    loop.run_in_executor(_imagen_executor,
                        _approach_outdoor, pb, p1, p2, g1, g2)
                ))
            else:
                d1 = _build_d_stage1(room, palette, wc, wf)
                d2 = _build_d_stage2(room, palette, wc, wf)
                e1 = _build_e_stage1(room, palette, wc, wf)
                e2 = _build_e_stage2(room, palette, wc, wf)
                cwc = room.get("current_wall_color", "")

                batch_futures.append(("D_FULL_SMART", room_idx,
                    loop.run_in_executor(_imagen_executor,
                        _approach_two_stage, pb, d1, d2,
                        GUIDANCE["D_STAGE1_CLEAN"], GUIDANCE["D_STAGE2_STAGE"],
                        "D", cwc, is_kitchen)
                ))
                batch_futures.append(("E_WALL_FORCE", room_idx,
                    loop.run_in_executor(_imagen_executor,
                        _approach_two_stage, pb, e1, e2,
                        GUIDANCE["E_STAGE1_WALL"], GUIDANCE["E_STAGE2_STAGE"],
                        "E", cwc, is_kitchen)
                ))

        if batch_futures:
            gathered = await asyncio.gather(
                *[f for _, _, f in batch_futures],
                return_exceptions=True
            )
            for (key, room_idx, _), result in zip(batch_futures, gathered):
                if isinstance(result, Exception):
                    print(f"[{key} stanza {room_idx}] EXC: {result}")
                    results[room_idx][key] = None
                else:
                    results[room_idx][key] = result

        print(f"[Batch] stanze {batch_start+1}–{batch_start+len(batch_rooms)} completate")

    return results


# ── Core Imagen call ──────────────────────────────────────────────────────────

def _call_imagen(label: str, compressed: bytes, prompt: str,
                 guidance: float, negative: str) -> bytes | None:
    """Singola chiamata Imagen con log del prompt se fallisce."""
    client = _get_vertex_client()
    try:
        resp = client.models.edit_image(
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
                negative_prompt=negative or None,
                safety_filter_level="block_only_high",
            ),
        )
        if resp.generated_images:
            return resp.generated_images[0].image.image_bytes
        # Log prompt per debug quando 0 immagini
        print(f"[{label}] 0 immagini (g={guidance}) — prompt[:150]: {prompt[:150]!r}")
        return None
    except Exception as e:
        print(f"[{label}] ERRORE: {type(e).__name__}: {e}")
        print(f"[{label}] prompt[:150]: {prompt[:150]!r}")
        return None


def _approach_two_stage(photo_bytes: bytes,
                        p1: str, p2: str,
                        g1: int, g2: int,
                        label: str,
                        current_wall_color: str = "",
                        is_kitchen: bool = False) -> str | None:
    compressed = compress_image(photo_bytes, max_width=1024, quality=80)
    print(f"[{label} S1] {len(compressed)//1024}KB g={g1}")

    # Negative prompts — corti e focalizzati
    neg1 = "distorted architecture, watermark, low quality, unrealistic"
    if label == "E":
        neg1 += ", original wall color, bleed-through"
        if current_wall_color:
            neg1 += f", {current_wall_color} walls"
        if is_kitchen:
            neg1 += ", old cabinet color, beige cabinets"

    s1 = _call_imagen(f"{label}S1", compressed, p1, g1, neg1)
    if not s1:
        print(f"[{label} S1] fallback su originale")
        s1 = compressed
    else:
        print(f"[{label} S1] OK → S2 g={g2}")

    neg2 = "empty room, bare walls, cartoon, flat lighting, floating objects, watermark"
    if label == "E":
        neg2 += ", original wall color retained, bleed-through"
        if current_wall_color:
            neg2 += f", {current_wall_color}"
        if is_kitchen:
            neg2 += ", old cabinet color"

    s2 = _call_imagen(f"{label}S2", s1, p2, g2, neg2)
    if s2:
        print(f"[{label} S2] SUCCESS")
        return base64.b64encode(s2).decode()
    return None


def _approach_outdoor(photo_bytes: bytes,
                      p1: str, p2: str,
                      g1: int, g2: int) -> str | None:
    """Template outdoor: preserva cielo e vista, aggiunge solo arredo esterno."""
    compressed = compress_image(photo_bytes, max_width=1024, quality=80)
    print(f"[OUTDOOR S1] {len(compressed)//1024}KB g={g1}")

    neg_outdoor = (
        "interior walls added, enclosed space, ceiling added over balcony, "
        "sky removed, view blocked, distorted architecture, watermark, low quality, "
        "indoor lighting, curtains, wallpaper, interior doors, roof added"
    )

    s1 = _call_imagen("OUTDOOR_S1", compressed, p1, g1, neg_outdoor)
    if not s1:
        s1 = compressed
        print("[OUTDOOR S1] fallback su originale")
    else:
        print("[OUTDOOR S1] OK → S2")

    neg2_outdoor = neg_outdoor + ", clutter, drying rack, laundry, rusty railings, dirty floor"
    s2 = _call_imagen("OUTDOOR_S2", s1, p2, g2, neg2_outdoor)
    if s2:
        print("[OUTDOOR S2] SUCCESS")
        return base64.b64encode(s2).decode()
    return None
