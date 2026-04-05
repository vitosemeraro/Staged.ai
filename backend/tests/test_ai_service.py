"""
Test unitari per ai_service.py
Coprono: _extract_json, validate_and_fix_costs, compress_image
Non richiedono credenziali Google — tutte le chiamate API sono mockate.
"""
import base64
import io
import json
import os
import sys

# Imposta variabili d'ambiente fake prima di importare il modulo
os.environ.setdefault("GCP_PROJECT_ID", "test-project")
os.environ.setdefault("GCP_LOCATION", "us-central1")
os.environ.setdefault("GEMINI_API_KEY", "fake-key-for-tests")

# Mock delle librerie Google prima dell'import
from unittest.mock import MagicMock, patch
import pytest

# Patch le librerie esterne prima che ai_service le importi
with patch.dict("sys.modules", {
    "google": MagicMock(),
    "google.genai": MagicMock(),
    "google.genai.types": MagicMock(),
    "vertexai": MagicMock(),
    "vertexai.preview": MagicMock(),
    "vertexai.preview.vision_models": MagicMock(),
}):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from ai_service import _extract_json, validate_and_fix_costs, compress_image


# ── Test _extract_json ────────────────────────────────────────────────────────

class TestExtractJson:

    def test_clean_json(self):
        """JSON pulito senza markdown."""
        raw = '{"nome": "test", "valore": 42}'
        result = _extract_json(raw)
        assert result["nome"] == "test"
        assert result["valore"] == 42

    def test_json_with_markdown_fence(self):
        """JSON avvolto in backtick markdown."""
        raw = '```json\n{"nome": "test"}\n```'
        result = _extract_json(raw)
        assert result["nome"] == "test"

    def test_json_with_text_before(self):
        """Testo spurio prima del JSON."""
        raw = 'Ecco il JSON richiesto:\n{"nome": "test", "ok": true}'
        result = _extract_json(raw)
        assert result["nome"] == "test"
        assert result["ok"] is True

    def test_json_with_text_after(self):
        """Testo spurio dopo il JSON."""
        raw = '{"nome": "test"}\n\nSpero sia utile!'
        result = _extract_json(raw)
        assert result["nome"] == "test"

    def test_nested_json(self):
        """JSON con oggetti annidati."""
        raw = '{"stanze": [{"nome": "Soggiorno", "costo": 500}], "totale": 500}'
        result = _extract_json(raw)
        assert len(result["stanze"]) == 1
        assert result["stanze"][0]["nome"] == "Soggiorno"

    def test_truncated_json_repaired(self):
        """JSON troncato (come da max_output_tokens) — deve essere riparato."""
        # Simula un JSON troncato a metà di un array
        raw = '{"stanze": [{"nome": "Soggiorno", "costo": 500}, {"nome": "Cucina"'
        # Ci aspettiamo che tenti la riparazione senza crashare con ValueError
        try:
            result = _extract_json(raw)
            # Se riesce, deve avere almeno il primo elemento
            assert "stanze" in result
        except ValueError as e:
            # Accettabile — l'importante è che non sia un json.JSONDecodeError nudo
            assert "troncato" in str(e).lower() or "riparabile" in str(e).lower()

    def test_no_json_raises(self):
        """Testo senza JSON deve sollevare ValueError."""
        with pytest.raises(ValueError, match="Nessun JSON"):
            _extract_json("Questo testo non contiene JSON valido")

    def test_real_gemini_response_structure(self):
        """Struttura JSON completa come quella attesa da Gemini."""
        sample = {
            "valutazione_generale": "Appartamento in buono stato",
            "punti_di_forza": ["Luminoso", "Ben posizionato"],
            "criticita": ["Pareti da ridipingere"],
            "potenziale_str": "Alto potenziale per STR",
            "tariffe": {
                "attuale_notte": "€60-80",
                "post_restyling_notte": "€90-110",
                "incremento_percentuale": "35%"
            },
            "stanze": [
                {
                    "nome": "Soggiorno",
                    "indice_foto": 0,
                    "stato_attuale": "Divano datato",
                    "interventi": [
                        {
                            "titolo": "Nuovo divano",
                            "dettaglio": "IKEA SÖDERHAMN grigio €699",
                            "costo_min": 600,
                            "costo_max": 750,
                            "priorita": "alta",
                            "dove_comprare": "IKEA"
                        }
                    ],
                    "costo_totale_stanza": 700,
                    "prompt_imagen": "Photorealistic interior photo..."
                }
            ],
            "riepilogo_costi": {
                "manodopera_tinteggiatura": 300,
                "materiali_pittura": 100,
                "arredi_complementi": 700,
                "montaggio_varie": 150,
                "totale": 1250,
                "budget_residuo": 250,
                "nota_budget": "Budget ben allocato"
            },
            "piano_acquisti": [
                {
                    "categoria": "Arredi",
                    "items": ["Divano IKEA"],
                    "budget_stimato": 700,
                    "negozi_consigliati": "IKEA"
                }
            ],
            "titolo_annuncio_suggerito": "Luminoso bilocale Milano centro",
            "highlights_str": ["WiFi fibra", "Balcone"],
            "roi_restyling": "Break-even in 15 notti"
        }
        raw = json.dumps(sample)
        result = _extract_json(raw)
        assert result["stanze"][0]["nome"] == "Soggiorno"
        assert result["riepilogo_costi"]["totale"] == 1250


# ── Test validate_and_fix_costs ───────────────────────────────────────────────

class TestValidateAndFixCosts:

    def _make_analysis(self, stanze_costi, totale_dichiarato=None, budget_residuo=0):
        stanze = []
        for i, costo in enumerate(stanze_costi):
            stanze.append({
                "nome": f"Stanza {i}",
                "costo_totale_stanza": costo,
                "interventi": [
                    {"costo_min": int(costo * 0.4), "costo_max": int(costo * 0.6)}
                ]
            })
        totale = totale_dichiarato if totale_dichiarato is not None else sum(stanze_costi)
        return {
            "stanze": stanze,
            "riepilogo_costi": {
                "manodopera_tinteggiatura": 200,
                "materiali_pittura": 100,
                "arredi_complementi": totale - 300,
                "montaggio_varie": 0,
                "totale": totale,
                "budget_residuo": budget_residuo,
                "nota_budget": "test"
            }
        }

    def test_valid_budget_unchanged(self):
        """Budget rispettato → nessuna modifica."""
        analysis = self._make_analysis([300, 400, 300])  # totale 1000
        result = validate_and_fix_costs(analysis, budget=1500)
        assert result["riepilogo_costi"]["totale"] == 1000

    def test_over_budget_scaled_down(self):
        """Totale > budget → tutto scalato al 95% del budget."""
        analysis = self._make_analysis([600, 600, 600])  # totale 1800
        result = validate_and_fix_costs(analysis, budget=1500)
        assert result["riepilogo_costi"]["totale"] <= 1500
        # Deve essere circa 95% del budget
        assert result["riepilogo_costi"]["totale"] <= int(1500 * 0.95) + 1

    def test_budget_residuo_correct(self):
        """budget_residuo = budget - totale."""
        analysis = self._make_analysis([400, 300])  # totale 700
        result = validate_and_fix_costs(analysis, budget=1000)
        rc = result["riepilogo_costi"]
        assert rc["budget_residuo"] == 1000 - rc["totale"]

    def test_discrepancy_fixed(self):
        """Discrepanza > €50 tra somma stanze e totale dichiarato → corretto."""
        analysis = self._make_analysis([300, 400], totale_dichiarato=800)  # 100 di discrepanza
        result = validate_and_fix_costs(analysis, budget=1500)
        stanze_sum = sum(r["costo_totale_stanza"] for r in result["stanze"])
        assert result["riepilogo_costi"]["totale"] == stanze_sum

    def test_min_max_swapped_fixed(self):
        """costo_min > costo_max → vengono scambiati."""
        analysis = {
            "stanze": [{
                "nome": "Test",
                "costo_totale_stanza": 500,
                "interventi": [{"costo_min": 300, "costo_max": 100}]  # invertiti
            }],
            "riepilogo_costi": {"totale": 500, "budget_residuo": 500}
        }
        result = validate_and_fix_costs(analysis, budget=1000)
        iv = result["stanze"][0]["interventi"][0]
        assert iv["costo_min"] <= iv["costo_max"]

    def test_exact_budget_ok(self):
        """Totale esattamente uguale al budget → accettato."""
        analysis = self._make_analysis([500, 500])  # totale 1000
        result = validate_and_fix_costs(analysis, budget=1000)
        assert result["riepilogo_costi"]["totale"] <= 1000

    def test_nota_budget_appended_when_fixed(self):
        """Quando i costi vengono corretti, la nota viene aggiornata."""
        analysis = self._make_analysis([1000, 1000])  # totale 2000, budget 1000
        result = validate_and_fix_costs(analysis, budget=1000)
        assert "ricalibrati" in result["riepilogo_costi"]["nota_budget"]


# ── Test compress_image ───────────────────────────────────────────────────────

class TestCompressImage:

    def _make_jpeg(self, width=2000, height=1500) -> bytes:
        """Crea un'immagine JPEG in memoria."""
        try:
            from PIL import Image as PILImage
            img = PILImage.new("RGB", (width, height), color=(128, 64, 32))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=95)
            return buf.getvalue()
        except ImportError:
            pytest.skip("Pillow non installato")

    def test_large_image_resized(self):
        """Immagine > 1400px viene ridimensionata."""
        original = self._make_jpeg(width=3000, height=2000)
        compressed = compress_image(original, max_width=1400)

        from PIL import Image as PILImage
        img = PILImage.open(io.BytesIO(compressed))
        assert img.width <= 1400

    def test_small_image_not_enlarged(self):
        """Immagine < 1400px NON viene ingrandita."""
        original = self._make_jpeg(width=800, height=600)
        compressed = compress_image(original, max_width=1400)

        from PIL import Image as PILImage
        img = PILImage.open(io.BytesIO(compressed))
        assert img.width == 800

    def test_compression_reduces_size(self):
        """La compressione riduce il peso del file."""
        original = self._make_jpeg(width=2000, height=1500)
        compressed = compress_image(original, max_width=1400, quality=82)
        assert len(compressed) < len(original)

    def test_returns_bytes(self):
        """Il risultato è sempre bytes."""
        original = self._make_jpeg()
        result = compress_image(original)
        assert isinstance(result, bytes)

    def test_invalid_input_returns_original(self):
        """Input non valido → restituisce i bytes originali senza crash."""
        garbage = b"not an image at all"
        result = compress_image(garbage)
        assert result == garbage
