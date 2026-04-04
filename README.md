# HomeStager AI — v2

Pipeline completa: **React (Vercel)** → **FastAPI (Cloud Run)** → **Gemini 1.5 Pro** + **Imagen 3 / ControlNet** → **WeasyPrint PDF** → **SendGrid email**

---

## Cosa fa

L'utente carica fino a 10 foto del proprio appartamento, imposta budget, stile e città, e riceve via email un PDF professionale con:

- **Analisi visiva** delle criticità attuali stanza per stanza
- **Tabella costi localizzata** per città (prezzi reali di mercato calcolati da Gemini, non hardcoded)
- **Foto Prima / Dopo** per ogni stanza: Imagen 3 sostituisce mobili, tessili e complementi preservando pareti, pavimenti e finestre dell'appartamento originale
- **Piano acquisti** con brand e negozi coerenti con lo stile scelto
- **Titolo e highlights Airbnb** ottimizzati + stima ROI

---

## Struttura del progetto

```
homestager/
├── backend/
│   ├── main.py             FastAPI: endpoint + job queue in-memoria
│   ├── ai_service.py       Gemini 1.5 Pro + Imagen 3 + ControlNet + validazione costi
│   ├── pdf_service.py      WeasyPrint: PDF con foto prima/dopo compresse
│   ├── email_service.py    SendGrid: invio PDF allegato
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
└── frontend/
    ├── src/
    │   ├── App.jsx         UI: upload → preferenze → polling → done
    │   ├── main.jsx        Entry point React
    │   └── index.css       CSS variables + reset (necessario fuori claude.ai)
    ├── index.html
    ├── package.json
    ├── vite.config.js
    ├── vercel.json
    └── .env.example
```

---

## Pipeline tecnica

```
Utente (browser)
  │  foto (JPEG/PNG, max 10) + budget + stile + città + destinazione + email
  ▼
React frontend  →  POST /analyze  (multipart/form-data)
  ▼
FastAPI  →  legge bytes, crea job_id, risponde immediatamente
  │
  └─ BackgroundTask: _process_job()
        │
        ├─ [1] Gemini 1.5 Pro (tutte le foto in una chiamata)
        │       → JSON scheda: valutazione, interventi, costi localizzati,
        │         piano acquisti, prompt_imagen per ogni stanza
        │       → validate_and_fix_costs(): verifica matematica budget
        │       → cache MD5: evita chiamate doppie su stesse foto
        │
        ├─ [2] Generazione foto staged (parallelo, una task per stanza)
        │       Path A — Imagen 3 inpainting (default)
        │         mask_prompt:     cosa sostituire (mobili, tessili, decor)
        │         negative_prompt: cosa preservare (pareti, pavimenti, finestre)
        │       Path B — ControlNet Depth via Replicate (se REPLICATE_API_TOKEN impostato)
        │         depth map → garantisce fedeltà geometrica assoluta
        │       Path C — Imagen 3 text-to-image (fallback automatico)
        │
        ├─ [3] WeasyPrint → PDF A4 multi-pagina
        │       Foto originali compresse con Pillow (1400px, JPEG 82%)
        │       → PDF tipicamente < 8 MB, deliverable via email
        │
        └─ [4] SendGrid → email con PDF allegato

React → GET /status/{job_id} ogni 2.5s → progress bar
```

---

## Stile: campo libero, nessuna lista chiusa

Il campo stile accetta qualsiasi testo: `Japandi`, `Boho Chic`, `Art Deco`, `Cottagecore`, `Wabi-Sabi`, `Mid-Century Modern`, ecc. Gemini interpreta lo stile e genera arredi, colori, tessili e decorazioni coerenti. I chip visibili nell'UI sono solo suggerimenti rapidi, non opzioni obbligatorie.

---

## Prezzi localizzati: nessun dato hardcoded

I prezzi per manodopera, materiali e arredi sono determinati da Gemini in base alla città indicata. Gemini conosce la differenza tra le tariffe di Milano (€10-12/mq tinteggiatura) e quelle di Napoli (€5-7/mq). Dopo la risposta di Gemini, `validate_and_fix_costs()` verifica che la somma dei costi non superi il budget e corregge proporzionalmente in caso di errore.

---

## Deploy — solo interfaccia web

### Prerequisiti una tantum (Google Cloud)

1. [console.cloud.google.com](https://console.cloud.google.com) → crea o seleziona un progetto → annota il **Project ID**
2. **APIs & Services → Library** → abilita:
   - `Vertex AI API`
   - `Cloud Run API`
   - `Cloud Build API`

### 1. Backend su Cloud Run

1. [console.cloud.google.com/run](https://console.cloud.google.com/run) → **Create Service**
2. **Continuously deploy from a repository** → collega GitHub → seleziona il repo
   - Branch: `main` · Build type: **Dockerfile** · Dockerfile: `/backend/Dockerfile`
3. Impostazioni servizio:

   | Campo | Valore |
   |---|---|
   | Service name | `homestager-backend` |
   | Region | `europe-west1` |
   | Authentication | **Allow unauthenticated invocations** |
   | Request timeout | `3600` |
   | Memory | `2 GiB` |
   | CPU | `2` |
   | **Minimum instances** | **`1`** ← mantiene il container vivo per i BackgroundTask |
   | Maximum instances | `3` |

4. **Container, Variables & Secrets → Variables**:

   | Variabile | Valore | Obbligatoria |
   |---|---|---|
   | `GCP_PROJECT_ID` | es. `my-project-123456` | Sì |
   | `GCP_LOCATION` | `us-central1` | Sì |
   | `SENDGRID_API_KEY` | `SG.xxxxxxxxxx` | Sì |
   | `FROM_EMAIL` | `noreply@tuodominio.it` | Sì |
   | `FROM_NAME` | `HomeStager AI` | No (default: HomeStager AI) |
   | `REPLICATE_API_TOKEN` | `r8_xxxxxxxxxx` | No — attiva ControlNet Depth |

5. **Create** → attendi 3-5 min → copia l'URL: `https://homestager-backend-xxxx-ew.a.run.app`

> **Nota `min-instances=1`:** FastAPI `BackgroundTasks` gira nello stesso processo del server. Con `min-instances=0` Cloud Run può spegnere il container dopo aver inviato la risposta HTTP, killando il job prima che finisca. Con `min-instances=1` il container rimane sempre attivo. Costo aggiuntivo: ~€7/mese.

> **Nota ControlNet:** se imposti `REPLICATE_API_TOKEN`, la generazione delle foto usa ControlNet Depth via Replicate invece di Imagen 3 inpainting. La depth map preserva geometria 3D della stanza in modo assoluto. Costo aggiuntivo: ~€0.05/immagine su Replicate vs €0.04 su Imagen 3. Per un prototipo Imagen 3 è sufficiente.

### 2. Frontend su Vercel

1. [vercel.com](https://vercel.com) → **Add New Project** → importa il repo GitHub
2. Configurazione:

   | Campo | Valore |
   |---|---|
   | Root Directory | `frontend` |
   | Framework Preset | **Vite** (rilevato automatico) |
   | Build Command | `npm run build` |
   | Output Directory | `dist` |

3. **Environment Variables**:

   | Nome | Valore |
   |---|---|
   | `VITE_API_URL` | URL Cloud Run del passo precedente |

4. **Deploy** → in ~2 minuti il frontend è live

### 3. Aggiorna CORS dopo il primo deploy

In `backend/main.py` decommenta e aggiorna:

```python
# "https://your-project.vercel.app",  # ← decommenta e metti il tuo URL Vercel
```

Push → Cloud Run si aggiorna automaticamente.

---

## Aggiornamenti futuri

Push su `main` → Vercel e Cloud Run si aggiornano automaticamente. Nessun terminale.

---

## Costi stimati

| Voce | Costo |
|---|---|
| Cloud Run (1 istanza minima) | ~€7/mese |
| Gemini 1.5 Pro per report | ~€0.05 |
| Imagen 3 inpainting × 5 stanze | ~€0.20 |
| ControlNet via Replicate × 5 stanze (opzionale) | ~€0.25 |
| Pillow + WeasyPrint | €0 |
| SendGrid (free tier) | €0 fino a 100 email/giorno |
| Vercel (free tier) | €0 |
| **Totale fisso mensile** | **~€7** |
| **Costo per report (Imagen 3)** | **~€0.25** |
| **Costo per report (ControlNet)** | **~€0.30** |

---

## Test in locale

```bash
# Backend
cd backend
pip install -r requirements.txt
cp .env.example .env        # compila con le tue credenziali
uvicorn main:app --reload --port 8000

# Frontend (nuovo terminale)
cd frontend
npm install
echo "VITE_API_URL=http://localhost:8000" > .env.local
npm run dev
```

---

## Note tecniche

**Validazione costi (`validate_and_fix_costs`):**
Dopo ogni risposta di Gemini vengono verificate tre invarianti: `costo_min ≤ costo_max` per ogni intervento, `SOMMA(costo_totale_stanza) == riepilogo_costi.totale`, `totale ≤ budget`. Se una è violata, tutti i costi vengono scalati proporzionalmente al 95% del budget.

**Compressione foto (`compress_image`):**
Le foto originali vengono ridimensionate a max 1400px e ricompresse a JPEG 82% prima dell'embedding nel PDF. Tipicamente da 3-5 MB a 200-400 KB per foto. Il PDF finale resta sotto 8 MB anche con 10 stanze.

**Cache Gemini:**
Hash MD5 di contenuto foto + stile + budget. Se lo stesso appartamento viene analizzato due volte con gli stessi parametri, Gemini non viene richiamato.

**Thread pool separati:**
`_gemini_executor` (3 worker) per le chiamate Gemini, `_imagen_executor` (8 worker) per le chiamate Imagen in parallelo. Evita che Gemini e Imagen si contendano gli stessi thread.
