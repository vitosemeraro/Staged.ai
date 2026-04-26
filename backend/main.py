"""
HomeStager AI — FastAPI Backend (Secure v2)
- X-Staged-Token header authentication
- Firestore credit verification before every AI call (server-side)
- Gemini API key never leaves the server
- /stage-image : Imagen/Gemini image generation
- /gemini-analyze : Gemini analysis proxy (consumes 1 credit)
- /gemini-translate : translation proxy (no credit)
- /gemini-products : product suggestions proxy (no credit)
"""
import os
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Firebase Admin ────────────────────────────────────────────────────────
import firebase_admin
from firebase_admin import credentials, firestore as fb_firestore

if not firebase_admin._apps:
    firebase_admin.initialize_app()   # uses Cloud Run default service account
db = fb_firestore.client()

# ── Env config ────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
STAGED_SECRET  = os.environ.get("STAGED_SECRET", "")
GEMINI_BASE    = "https://generativelanguage.googleapis.com/v1beta/models"

app = FastAPI(title="HomeStager AI Secure")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://gestione-affitti-brevi-milano.it",
        "https://www.gestione-affitti-brevi-milano.it",
        "https://staged-ai-six.vercel.app",
    ],
    allow_methods=["*"],
    allow_headers=["*", "X-Staged-Token", "X-User-UID"],
)

# ── Helpers ───────────────────────────────────────────────────────────────

def verify_token(request: Request):
    if not STAGED_SECRET:
        return  # skip in dev if not set
    if request.headers.get("X-Staged-Token", "") != STAGED_SECRET:
        raise HTTPException(status_code=401, detail="Token non valido.")


def consume_credit(uid: str):
    """Decrement credit in Firestore transaction. Raises 402 if out of credits."""
    if not uid:
        raise HTTPException(status_code=401, detail="UID utente mancante.")
    user_ref = db.collection("users").document(uid)

    @fb_firestore.transactional
    def _txn(transaction, ref):
        snap = ref.get(transaction=transaction)
        if not snap.exists:
            raise HTTPException(status_code=403, detail="Utente non trovato.")
        credits = snap.to_dict().get("credits", 0)
        if credits <= 0:
            raise HTTPException(status_code=402,
                detail="Crediti esauriti. Contatta l'amministratore.")
        transaction.update(ref, {"credits": fb_firestore.Increment(-1)})
        return credits - 1

    txn = db.transaction()
    return _txn(txn, user_ref)


async def call_gemini(model: str, payload: dict, timeout: float = 120.0) -> dict:
    url = f"{GEMINI_BASE}/{model}:generateContent?key={GEMINI_API_KEY}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload,
                                 headers={"Content-Type": "application/json"})
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text[:500])
    return resp.json()


# ── Models ────────────────────────────────────────────────────────────────

class GeminiReq(BaseModel):
    payload: dict
    model: str = "gemini-2.5-flash"
    consume_credit: bool = True

class StageImageReq(BaseModel):
    photo_b64: str
    photo_mime: str = "image/jpeg"
    prompt: str

class TranslateReq(BaseModel):
    text: str

# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok",
            "gemini_key": bool(GEMINI_API_KEY),
            "secret": bool(STAGED_SECRET)}


@app.post("/gemini-analyze")
async def gemini_analyze(req: GeminiReq, request: Request):
    """Main analysis — consumes 1 credit. Gemini key stays server-side."""
    verify_token(request)
    uid = request.headers.get("X-User-UID", "")
    if req.consume_credit:
        consume_credit(uid)
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY non configurata.")
    return await call_gemini(req.model, req.payload)


@app.post("/gemini-translate")
async def gemini_translate(req: TranslateReq, request: Request):
    """Translation — no credit consumed."""
    verify_token(request)
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY non configurata.")
    payload = {
        "contents": [{"role": "user", "parts": [{"text":
            f"Translate to English. Return ONLY the translation:\n\n{req.text}"
        }]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 2048}
    }
    data = await call_gemini("gemini-2.5-flash", payload, timeout=30.0)
    try:
        return {"translation": data["candidates"][0]["content"]["parts"][0]["text"]}
    except (KeyError, IndexError):
        raise HTTPException(status_code=500, detail="Risposta traduzione malformata.")


@app.post("/stage-image")
async def stage_image(req: StageImageReq, request: Request):
    """Image generation — token verified, credit already consumed in /gemini-analyze."""
    verify_token(request)
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY non configurata.")

    MODELS = [
        "gemini-3.1-flash-image-preview",
        "gemini-2.5-flash-image",
        "gemini-2.0-flash-exp",
    ]
    payload = {
        "contents": [{"parts": [
            {"inline_data": {"mime_type": req.photo_mime, "data": req.photo_b64}},
            {"text": req.prompt}
        ]}],
        "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]}
    }
    last_err = ""
    async with httpx.AsyncClient(timeout=120.0) as client:
        for model in MODELS:
            url = f"{GEMINI_BASE}/{model}:generateContent?key={GEMINI_API_KEY}"
            try:
                resp = await client.post(url, json=payload,
                                         headers={"Content-Type": "application/json"})
                data = resp.json()
                if "error" in data:
                    last_err = f"{model}: {data['error'].get('message','')}"; continue
                parts = (data.get("candidates",[{}])[0]
                         .get("content",{}).get("parts",[]))
                img = next((p for p in parts
                            if (p.get("inline_data") or p.get("inlineData",{}))
                               .get("mime_type","").startswith("image/")), None)
                if not img:
                    last_err = f"{model}: no image"; continue
                id_ = img.get("inline_data") or img.get("inlineData",{})
                return {"image_b64": id_.get("data",""),
                        "mime": id_.get("mime_type","image/png"), "model": model}
            except Exception as e:
                last_err = f"{model}: {e}"
    raise HTTPException(status_code=500, detail=f"Immagine fallita: {last_err}")


@app.post("/gemini-products")
async def gemini_products(req: GeminiReq, request: Request):
    """Product suggestions — no credit consumed."""
    verify_token(request)
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY non configurata.")
    req.consume_credit = False
    return await call_gemini(req.model, req.payload, timeout=30.0)