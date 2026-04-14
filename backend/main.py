"""
HomeStager AI — FastAPI Backend v18
Pipeline: Upload → PhotoValidator → Gemini 2.5 Flash → Imagen 3 → WeasyPrint PDF → SendGrid
"""
import base64
import traceback
import uuid
import asyncio
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ai_service import (
    analyze_with_gemini, generate_staged_photos,
    validate_input_photos, compress_image, _get_vertex_client,
)
from pdf_service import generate_pdf
from email_service import send_report_email
from google.genai import types as genai_types

app = FastAPI(title="HomeStager AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # wildcard per lab sandbox + localhost + Vercel
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

    # ── PhotoValidator — analisi qualità prima di avviare il job ──────────────
    validation = await validate_input_photos(photo_data)
    invalid_photos = [
        {"index": i, "issue": validation["issues"][i], "suggestion": validation.get("suggestions", [""])[i]}
        for i, ok in enumerate(validation["valid"])
        if not ok
    ]

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "processing",
        "progress": 0,
        "step": "Inizializzazione…",
        # Esponi i warning di validazione subito nel job — il frontend può mostrarli
        "validation": {
            "warnings":       validation.get("warnings", []),
            "invalid_photos": invalid_photos,
            "layout_hint":    validation.get("layout_hint", ""),
        }
    }

    prefs = {
        "budget":      budget,
        "style":       style,
        "location":    location,
        "destination": destination,
    }
    background_tasks.add_task(_process_job, job_id, photo_data, prefs, email)
    return {
        "job_id":    job_id,
        "validation": jobs[job_id]["validation"],   # restituisce subito i warning
    }


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
    Chiama Imagen direttamente senza pipeline completa.
    guidance_scale cappato a 30 (>30 = 0 immagini silenzioso).
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
            "error": f"Imagen ha restituito 0 immagini (guidance={guidance_scale}). "
                     "Prova un valore <= 28.",
        }

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[test-variant] ERRORE:\n{tb}")
        return {"success": False, "error": f"{type(e).__name__}: {e}"}


async def _process_job(job_id: str, photos: list, prefs: dict, email: str):
    def update(progress: int, step: str):
        jobs[job_id].update({"progress": progress, "step": step})

    try:
        update(10, "Gemini analizza le foto (spatial anchor + style DNA)…")
        analysis = await analyze_with_gemini(photos, prefs)

        update(35, "Imagen 3 genera D + E (KitchenBrutalistProtocol attivo)…")
        staged_results = await generate_staged_photos(photos, analysis)

        update(80, "Generazione PDF…")
        pdf_bytes = generate_pdf(analysis, prefs, photos, staged_results=staged_results)

        update(92, "Invio email…")
        send_report_email(email, pdf_bytes, analysis, prefs)

        jobs[job_id] = {
            "status":   "completed",
            "progress": 100,
            "step":     "Completato",
            "summary": {
                "titolo":       analysis.get("titolo_annuncio_suggerito", ""),
                "totale_costi": analysis.get("riepilogo_costi", {}).get("totale", 0),
                "incremento":   analysis.get("tariffe", {}).get("incremento_percentuale", ""),
                "spatial_map":  analysis.get("spatial_map", {}),
            },
            "validation": jobs[job_id].get("validation", {}),
        }

    except Exception as exc:
        tb = traceback.format_exc()
        print(tb)
        jobs[job_id] = {
            "status":    "error",
            "progress":  0,
            "step":      "Errore",
            "error":     f"{type(exc).__name__}: {exc}",
            "traceback": tb,
            "validation": jobs[job_id].get("validation", {}),
        }