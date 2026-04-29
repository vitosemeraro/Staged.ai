"""
HomeStager AI — FastAPI Backend
Pipeline: Upload → Gemini 2.5 Flash → Imagen 3 → WeasyPrint PDF → SendGrid
"""
import traceback
import uuid
import asyncio
import os
import httpx
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from ai_service import analyze_with_gemini, generate_staged_photos
from pdf_service import generate_pdf
from email_service import send_report_email

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
STAGED_SECRET  = os.environ.get("STAGED_SECRET", "hs_beta_2025_v1")

app = FastAPI(title="HomeStager AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://staged-ai-six.vercel.app",
        "https://gestione-affitti-brevi-milano.it",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs: dict = {}


def _check_token(request: Request):
    token = request.headers.get("X-Staged-Token", "")
    if token != STAGED_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/health")
def health():
    return {"status": "ok"}


# ── Gemini text proxy ─────────────────────────────────────────────────────────
# Replaces direct frontend calls to generativelanguage.googleapis.com
# Key never leaves the server.

@app.post("/gemini-analyze")
async def gemini_analyze(request: Request):
    """Proxy for all Gemini text/JSON generation calls."""
    _check_token(request)
    body = await request.json()
    model   = body.get("model", "gemini-2.5-flash")
    payload = body.get("payload", {})
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={GEMINI_API_KEY}")
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(url, json=payload,
                              headers={"Content-Type": "application/json"})
    if not r.is_success:
        raise HTTPException(status_code=r.status_code, detail=r.text[:500])
    return r.json()


@app.post("/gemini-translate")
async def gemini_translate(request: Request):
    """Translate Italian text to English for image prompts."""
    _check_token(request)
    body = await request.json()
    text = body.get("text", "")
    if not text.strip():
        return {"translated": text}
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")
    payload = {
        "contents": [{"role": "user", "parts": [
            {"text": f"Translate this Italian interior design instruction to English. Reply ONLY with the translation, no explanation:\n\n{text}"}
        ]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 500}
    }
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}")
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(url, json=payload,
                              headers={"Content-Type": "application/json"})
    if not r.is_success:
        raise HTTPException(status_code=r.status_code, detail=r.text[:500])
    data = r.json()
    translated = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    return {"translated": translated}


@app.post("/stage-image")
async def stage_image(request: Request):
    """Proxy for Gemini image generation (gemini-3.1-flash-image-preview etc.)."""
    _check_token(request)
    body = await request.json()
    model   = body.get("model", "gemini-3.1-flash-image-preview")
    payload = body.get("payload", {})
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")
    # Use v1beta for image generation models
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={GEMINI_API_KEY}")
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(url, json=payload,
                              headers={"Content-Type": "application/json"})
    if not r.is_success:
        raise HTTPException(status_code=r.status_code, detail=r.text[:500])
    return r.json()


@app.post("/analyze")
async def analyze(
    background_tasks: BackgroundTasks,
    photos: list[UploadFile] = File(...),
    budget: int = Form(...),
    style: str = Form(...),
    location: str = Form(...),
    destination: str = Form(...),
    email: str = Form(...),
):
    if not photos:
        raise HTTPException(status_code=400, detail="Almeno una foto è richiesta")
    if len(photos) > 10:
        raise HTTPException(status_code=400, detail="Massimo 10 foto")

    photo_data = []
    for photo in photos:
        content = await photo.read()
        photo_data.append({
            "content": content,
            "filename": photo.filename,
            "content_type": photo.content_type or "image/jpeg",
        })

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "progress": 0, "step": "Inizializzazione…"}

    prefs = {
        "budget": budget,
        "style": style,
        "location": location,
        "destination": destination,
    }
    background_tasks.add_task(_process_job, job_id, photo_data, prefs, email)
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job non trovato")
    return job


async def _process_job(job_id: str, photos: list, prefs: dict, email: str):
    def update(progress: int, step: str):
        jobs[job_id].update({"progress": progress, "step": step})

    try:
        update(10, "Gemini analizza le foto…")
        analysis = await analyze_with_gemini(photos, prefs)

        update(35, "Imagen 3 genera le foto staged (4 approcci)…")
        staged_results = await generate_staged_photos(photos, analysis)

        update(70, "Compilazione scheda…")
        # staged_results è ora una lista di dict con chiavi A_base, B_geometric, C_reference, D_edit
        # Non serve più attaccare staged_photo_b64 alle stanze — lo fa generate_pdf

        update(80, "Generazione PDF…")
        pdf_bytes = generate_pdf(analysis, prefs, photos, staged_results=staged_results)

        update(92, "Invio email…")
        send_report_email(email, pdf_bytes, analysis, prefs)

        jobs[job_id] = {
            "status": "completed",
            "progress": 100,
            "step": "Completato",
            "summary": {
                "titolo": analysis.get("titolo_annuncio_suggerito", ""),
                "totale_costi": analysis.get("riepilogo_costi", {}).get("totale", 0),
                "incremento": analysis.get("tariffe", {}).get("incremento_percentuale", ""),
            },
        }

    except Exception as exc:
        # Traceback completo salvato nel job — visibile nel frontend per debug
        tb = traceback.format_exc()
        print(tb)  # anche nei log Cloud Run
        jobs[job_id] = {
            "status": "error",
            "progress": 0,
            "step": "Errore",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": tb,  # ← NUOVO: traceback completo
        }