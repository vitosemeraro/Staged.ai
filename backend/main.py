"""
HomeStager AI — FastAPI Backend
Pipeline: Upload → Gemini 2.5 Flash → Imagen 3 → WeasyPrint PDF → SendGrid
"""
import base64
import traceback
import uuid
import asyncio
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ai_service import analyze_with_gemini, generate_staged_photos, compress_image, _get_vertex_client
from pdf_service import generate_pdf
from email_service import send_report_email
from google.genai import types as genai_types

app = FastAPI(title="HomeStager AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # wildcard: permette chiamate dalla sandbox Claude.ai e da localhost
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs: dict = {}


@app.get("/health")
def health():
    return {"status": "ok"}


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


@app.post("/test-variant")
async def test_variant(
    photo: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    guidance_scale: float = Form(25.0),
):
    """
    Endpoint di test per la sandbox di configurazione.
    Riceve una foto + parametri di una variante Imagen, ritorna l'immagine generata in base64.
    Bypassa la pipeline completa (niente Gemini, PDF, email).

    Limite: guidance_scale viene cappato a 30 (valori > 30 restituiscono
    silenziosamente 0 immagini in EDIT_MODE_DEFAULT).
    """
    guidance_scale = min(float(guidance_scale), 30.0)
    content = await photo.read()

    try:
        compressed = compress_image(content, max_width=1024, quality=80)
        client = _get_vertex_client()

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
                negative_prompt=negative_prompt or None,
                safety_filter_level="block_only_high",
            ),
        )

        if response.generated_images:
            b64 = base64.b64encode(
                response.generated_images[0].image.image_bytes
            ).decode()
            return {"success": True, "image_b64": b64, "guidance_used": guidance_scale}

        return {
            "success": False,
            "error": (
                f"Imagen ha restituito 0 immagini (guidance={guidance_scale}). "
                "Prova un valore <= 28."
            ),
        }

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[test-variant] ERRORE:\n{tb}")
        return {"success": False, "error": f"{type(e).__name__}: {e}"}


async def _process_job(job_id: str, photos: list, prefs: dict, email: str):
    def update(progress: int, step: str):
        jobs[job_id].update({"progress": progress, "step": step})

    try:
        update(10, "Gemini analizza le foto…")
        analysis = await analyze_with_gemini(photos, prefs)

        update(35, "Imagen 3 genera le foto staged…")
        staged_results = await generate_staged_photos(photos, analysis)

        update(70, "Compilazione scheda…")

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
        tb = traceback.format_exc()
        print(tb)
        jobs[job_id] = {
            "status": "error",
            "progress": 0,
            "step": "Errore",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": tb,
        }
