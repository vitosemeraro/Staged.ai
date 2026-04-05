"""
Test degli endpoint FastAPI.
Usa TestClient di FastAPI — non richiede server in ascolto.
Tutte le dipendenze esterne (Gemini, Imagen, SendGrid) sono mockate.
"""
import io
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("GCP_PROJECT_ID", "test-project")
os.environ.setdefault("GCP_LOCATION", "us-central1")
os.environ.setdefault("GEMINI_API_KEY", "fake-key-for-tests")
os.environ.setdefault("SENDGRID_API_KEY", "SG.fake")
os.environ.setdefault("FROM_EMAIL", "test@test.com")

# Mock tutte le dipendenze esterne
with patch.dict("sys.modules", {
    "google": MagicMock(),
    "google.genai": MagicMock(),
    "google.genai.types": MagicMock(),
    "vertexai": MagicMock(),
    "vertexai.preview": MagicMock(),
    "vertexai.preview.vision_models": MagicMock(),
    "sendgrid": MagicMock(),
    "sendgrid.helpers": MagicMock(),
    "sendgrid.helpers.mail": MagicMock(),
    "weasyprint": MagicMock(),
    "jinja2": MagicMock(),
}):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi.testclient import TestClient

FAKE_ANALYSIS = {
    "valutazione_generale": "Test appartamento",
    "punti_di_forza": ["Luminoso"],
    "criticita": ["Pareti da ridipingere"],
    "potenziale_str": "Buon potenziale",
    "tariffe": {
        "attuale_notte": "€60",
        "post_restyling_notte": "€90",
        "incremento_percentuale": "50%"
    },
    "stanze": [{
        "nome": "Soggiorno",
        "indice_foto": 0,
        "stato_attuale": "OK",
        "interventi": [{
            "titolo": "Nuovo divano",
            "dettaglio": "IKEA",
            "costo_min": 300,
            "costo_max": 500,
            "priorita": "alta",
            "dove_comprare": "IKEA"
        }],
        "costo_totale_stanza": 400,
        "prompt_imagen": "Test prompt"
    }],
    "riepilogo_costi": {
        "manodopera_tinteggiatura": 200,
        "materiali_pittura": 100,
        "arredi_complementi": 400,
        "montaggio_varie": 50,
        "totale": 750,
        "budget_residuo": 750,
        "nota_budget": "OK"
    },
    "piano_acquisti": [{
        "categoria": "Arredi",
        "items": ["Divano"],
        "budget_stimato": 400,
        "negozi_consigliati": "IKEA"
    }],
    "titolo_annuncio_suggerito": "Test appartamento Milano",
    "highlights_str": ["WiFi"],
    "roi_restyling": "Break-even 10 notti"
}


def make_test_image() -> bytes:
    """Crea un'immagine JPEG minima valida (1x1 pixel)."""
    # JPEG minimo valido in bytes
    return bytes([
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
        0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
        0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
        0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
        0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
        0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
        0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
        0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
        0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00,
        0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
        0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01, 0x03,
        0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D,
        0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00, 0xFB, 0xD2,
        0x8A, 0x28, 0xFF, 0xD9
    ])


@pytest.fixture
def client():
    """TestClient con tutte le dipendenze AI mockate."""
    with patch("ai_service.analyze_with_gemini", new_callable=AsyncMock, return_value=FAKE_ANALYSIS), \
         patch("ai_service.generate_staged_photos", new_callable=AsyncMock, return_value=[None]), \
         patch("pdf_service.generate_pdf", return_value=b"%PDF-fake"), \
         patch("email_service.send_report_email", return_value=None):
        import main
        yield TestClient(main.app)


class TestHealthEndpoint:

    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestAnalyzeEndpoint:

    def test_missing_photos_returns_400(self, client):
        resp = client.post("/analyze", data={
            "budget": 1500,
            "style": "Scandinavo",
            "location": "Milano",
            "destination": "STR",
            "email": "test@test.com"
        })
        assert resp.status_code == 422  # FastAPI validation error

    def test_valid_request_returns_job_id(self, client):
        img = make_test_image()
        resp = client.post("/analyze",
            data={
                "budget": 1500,
                "style": "Scandinavo",
                "location": "Milano",
                "destination": "STR",
                "email": "test@test.com"
            },
            files=[("photos", ("test.jpg", img, "image/jpeg"))]
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert len(data["job_id"]) == 36  # UUID format

    def test_too_many_photos_returns_400(self, client):
        img = make_test_image()
        files = [("photos", (f"photo{i}.jpg", img, "image/jpeg")) for i in range(11)]
        resp = client.post("/analyze",
            data={
                "budget": 1500,
                "style": "Scandinavo",
                "location": "Milano",
                "destination": "STR",
                "email": "test@test.com"
            },
            files=files
        )
        assert resp.status_code == 400

    def test_invalid_email_accepted(self, client):
        """FastAPI non valida il formato email — solo la presenza."""
        img = make_test_image()
        resp = client.post("/analyze",
            data={
                "budget": 1500,
                "style": "Scandinavo",
                "location": "Milano",
                "destination": "STR",
                "email": "not-an-email"
            },
            files=[("photos", ("test.jpg", img, "image/jpeg"))]
        )
        assert resp.status_code == 200


class TestStatusEndpoint:

    def test_unknown_job_returns_404(self, client):
        resp = client.get("/status/nonexistent-job-id")
        assert resp.status_code == 404

    def test_known_job_returns_status(self, client):
        img = make_test_image()
        post_resp = client.post("/analyze",
            data={
                "budget": 1500,
                "style": "Scandinavo",
                "location": "Milano",
                "destination": "STR",
                "email": "test@test.com"
            },
            files=[("photos", ("test.jpg", img, "image/jpeg"))]
        )
        job_id = post_resp.json()["job_id"]
        status_resp = client.get(f"/status/{job_id}")
        assert status_resp.status_code == 200
        data = status_resp.json()
        assert data["status"] in ("processing", "completed", "error")
        assert "progress" in data
