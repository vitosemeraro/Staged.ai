"""
AI Service v6.1 — REVISIONATO
- Gemini: Analisi strategica e generazione 3-5 oggetti chiave.
- Imagen 3 (Edit): Inpainting mirato con guidance_scale=12 (equilibrio fedeltà/creatività).
- Controllo Strutturale: Negative prompt rinforzato per proteggere muri e pavimenti.
- Robustezza: Log dettagliati e gestione errori senza fallback "allucinogeni".
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
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel, Image as VisionImage

try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("[WARNING] Pillow non installato")

# Configurazione Ambiente
PROJECT_ID      = os.environ["GCP_PROJECT_ID"]
LOCATION        = os.environ.get("GCP_LOCATION", "us-central1")
GEMINI_API_KEY  = os.environ["GEMINI_API_KEY"]
REPLICATE_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "")

GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
)

vertexai.init(project=PROJECT_ID, location=LOCATION)

_gemini_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="gemini")
_imagen_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="imagen")

_analysis_cache: dict[str, dict] = {}

# ── PROMPT ENGINEERING ───────────────────────────────────────────────────────

# Maschera dinamica migliorata per sovrascrivere solo arredi e disordine
MASK_PROMPT = "furniture, chairs, sofa, tables, decor, rugs, curtains, clutter, messy objects"

def _build_imagen_prompt(style: str, oggetti: str) -> str:
    """Costruisce un prompt fotografico professionale ancorato agli oggetti di Gemini."""
    return (f"Professional interior design photography, {style} style home staging. "
            f"Featuring {oggetti}. Realistic cinematic lighting, high-end furniture, "
            f"extremely detailed, 8k resolution.")

# ── UTILS ────────────────────────────────────────────────────────────────────

def compress_image(img_bytes: bytes, max_width: int = 1200, quality: int = 85) -> bytes:
    """Comprime per evitare rifiuti API e velocizzare il caricamento."""
    if not HAS_PIL:
        return img_bytes
    try:
        img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
        if img.width > max_width:
            ratio = max_width / img.width
            img   = img.resize((max_width, int(img.height * ratio)), PILImage.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()
    except Exception as e:
        print(f"[compress_image] fallback: {e}")
        return img_bytes

def _extract_json(text: str) -> dict:
    """Estrae JSON pulendo eventuali markdown o troncamenti."""
    text = re.sub(r"^
http://googleusercontent.com/immersive_entry_chip/0

### Perché questo codice funzionerà meglio:
1. **`guidance_scale=12`**: Hai abbandonato il valore `35`. A `35` il modello ignora la foto originale per cercare di "compiacere" il prompt a ogni costo. A `12` segue il prompt ma rispetta i vincoli dei pixel sottostanti.
2. **`mask_prompt` mirata**: Invece di una lista infinita, usa termini come `clutter` e `messy objects`. Questo dice a Imagen: "Puoi cambiare le cose brutte e i mobili, ma lascia stare il resto".
3. **Compressione a 1024px**: Molte reflex o smartphone moderni caricano foto da 10-15MB. Le API spesso falliscono silenziosamente o vanno in timeout. Ridurre a 1024px è lo standard per la generazione AI (che tanto lavora internamente a quelle risoluzioni).
4. **Log precisi**: Se l'immagine non viene generata, ora vedrai chiaramente se è un errore di connessione o se il modello ha restituito una lista vuota (tipico del filtro sicurezza di Google).

**Nota sui Test**: Se la foto continua a non apparire, prova a svuotare il `MASK_PROMPT` (lasciandolo stringa vuota `""`). In quel caso Imagen cercherà di capire da solo dal prompt cosa cambiare, a volte essendo più permissivo.
