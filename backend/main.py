"""
HomeStager AI — FastAPI Backend
Pipeline: Upload → Gemini 1.5 Pro analysis → Imagen 3 staging (parallel) → WeasyPrint PDF → SendGrid
"""
import uuid
import asyncio
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ai_service import analyze_with_gemini, generate_staged_photos
from pdf_service import generate_pdf
from email_service import send_report_email

app = FastAPI(title="HomeStager AI")

# Update allow_origins with your Vercel URL after first deploy
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        # "https://your-project.vercel.app",  # <-- uncomment and set after Vercel deploy
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store. Works for single-instance Cloud Run (min-instances=1).
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

    # Read all bytes before handing off — UploadFile is not safe after response
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

        update(35, "Imagen 3 genera le foto staged…")
        staged_photos = await generate_staged_photos(photos, analysis)

        update(70, "Compilazione scheda…")
        for i, room in enumerate(analysis.get("stanze", [])):
            if i < len(staged_photos):
                room["staged_photo_b64"] = staged_photos[i]

        update(80, "Generazione PDF…")
        pdf_bytes = generate_pdf(analysis, prefs, photos)

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
        import traceback
        traceback.print_exc()
        jobs[job_id] = {
            "status": "error",
            "progress": 0,
            "step": "Errore",
            "error": str(exc),
        }
