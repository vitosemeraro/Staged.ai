import { useState, useRef, useCallback, useEffect } from "react";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

// Suggerimenti rapidi — non sono un elenco chiuso.
// L'utente può digitare qualsiasi stile nel campo testo.
const STYLE_SUGGESTIONS = [
  "Scandinavo", "Minimalista", "Boho Chic", "Industrial", "Japandi",
  "Lusso Contemporaneo", "Mediterraneo", "Art Deco", "Provenzale", "Mid-Century Modern",
];

const CITIES = ["Milano", "Roma", "Firenze", "Venezia", "Bologna", "Torino", "Napoli", "Altra città"];

function readPreview(file) {
  return new Promise((res) => {
    const r = new FileReader();
    r.onload = () => res(r.result);
    r.readAsDataURL(file);
  });
}

const s = {
  wrap: { fontFamily: "var(--font-sans)", padding: "1.5rem 1rem", maxWidth: 720, margin: "0 auto" },
  h2: { fontSize: 20, fontWeight: 500, color: "var(--color-text-primary)", margin: "0 0 8px" },
  sub: { fontSize: 14, color: "var(--color-text-secondary)", lineHeight: 1.6, marginBottom: 20 },
  card: { background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-tertiary)", borderRadius: "var(--border-radius-lg)", padding: "1rem 1.25rem", marginBottom: 12 },
  btn: { background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-secondary)", borderRadius: "var(--border-radius-md)", padding: "8px 18px", fontSize: 14, cursor: "pointer", color: "var(--color-text-primary)" },
  btnPrimary: { background: "var(--color-text-primary)", border: "none", borderRadius: "var(--border-radius-md)", padding: "10px 24px", fontSize: 14, cursor: "pointer", color: "var(--color-background-primary)", fontWeight: 500 },
  label: { fontSize: 13, color: "var(--color-text-secondary)", marginBottom: 6, display: "block" },
  row: { display: "flex", gap: 12, alignItems: "center" },
  divider: { borderTop: "0.5px solid var(--color-border-tertiary)", margin: "16px 0" },
  h3: { fontSize: 15, fontWeight: 500, color: "var(--color-text-primary)", margin: "0 0 10px" },
};

// ─── Step 0: Intro ──────────────────────────────────────────────────────────
function IntroStep({ onStart }) {
  return (
    <div style={s.wrap}>
      <p style={{ fontSize: 12, color: "var(--color-text-tertiary)", letterSpacing: 1.5, textTransform: "uppercase", marginBottom: 8 }}>Powered by Gemini + Imagen 3</p>
      <h1 style={{ fontSize: 24, fontWeight: 500, color: "var(--color-text-primary)", marginBottom: 8 }}>Home Staging AI</h1>
      <p style={s.sub}>Carica le foto del tuo appartamento. Ricevi via email un PDF con le foto <em>dopo staging</em> generate da Imagen 3 e la scheda completa con preventivi localizzati.</p>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 24 }}>
        {[
          ["Carica foto", "Soggiorno, cucina, camere, bagno, balcone"],
          ["Gemini analizza", "Scheda stanza per stanza con preventivi"],
          ["Imagen 3 crea", "Foto realistiche post-staging per ogni stanza"],
          ["PDF via email", "Report completo con prima/dopo in allegato"],
        ].map(([t, d], i) => (
          <div key={i} style={s.card}>
            <p style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginBottom: 3 }}>0{i + 1}</p>
            <p style={{ fontWeight: 500, fontSize: 14, marginBottom: 2, color: "var(--color-text-primary)" }}>{t}</p>
            <p style={{ fontSize: 13, color: "var(--color-text-secondary)", margin: 0 }}>{d}</p>
          </div>
        ))}
      </div>

      <button style={s.btnPrimary} onClick={onStart}>Inizia →</button>
    </div>
  );
}

// ─── Step 1: Upload ──────────────────────────────────────────────────────────
function PhotoStep({ files, setFiles, onNext }) {
  const [dragOver, setDragOver] = useState(false);
  const [previews, setPreviews] = useState([]);
  const inputRef = useRef(null);

  const addFiles = useCallback(async (newFiles) => {
    const valid = Array.from(newFiles).filter(f => f.type.startsWith("image/")).slice(0, 10 - files.length);
    if (!valid.length) return;
    const newPreviews = await Promise.all(valid.map(readPreview));
    setFiles(f => [...f, ...valid].slice(0, 10));
    setPreviews(p => [...p, ...newPreviews].slice(0, 10));
  }, [files.length, setFiles]);

  return (
    <div style={s.wrap}>
      <p style={{ fontSize: 12, color: "var(--color-text-tertiary)", marginBottom: 4 }}>Passo 1 di 2</p>
      <h2 style={s.h2}>Carica le foto dell'appartamento</h2>
      <p style={s.sub}>Ogni stanza che carichi genererà una foto staged con Imagen 3. Più foto, più ricco il report.</p>

      <div
        onClick={() => inputRef.current?.click()}
        onDrop={(e) => { e.preventDefault(); setDragOver(false); addFiles(e.dataTransfer.files); }}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        style={{
          border: `1.5px dashed ${dragOver ? "var(--color-border-info)" : "var(--color-border-secondary)"}`,
          borderRadius: "var(--border-radius-lg)", padding: "2rem", textAlign: "center",
          cursor: "pointer", marginBottom: 16,
          background: dragOver ? "var(--color-background-info)" : "var(--color-background-secondary)",
        }}
      >
        <p style={{ fontSize: 14, color: "var(--color-text-secondary)", margin: 0 }}>
          {dragOver ? "Rilascia qui" : "Trascina le foto o clicca per selezionarle"}
        </p>
        <p style={{ fontSize: 12, color: "var(--color-text-tertiary)", marginTop: 4, marginBottom: 0 }}>JPG, PNG — max 10 foto</p>
        <input ref={inputRef} type="file" multiple accept="image/*" style={{ display: "none" }}
          onChange={e => addFiles(e.target.files)} />
      </div>

      {previews.length > 0 && (
        <div style={{ marginBottom: 16 }}>
          <p style={{ ...s.label, marginBottom: 8 }}>{previews.length} foto caricate</p>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(90px, 1fr))", gap: 8 }}>
            {previews.map((src, i) => (
              <div key={i} style={{ position: "relative", aspectRatio: "1", borderRadius: "var(--border-radius-md)", overflow: "hidden", border: "0.5px solid var(--color-border-tertiary)" }}>
                <img src={src} style={{ width: "100%", height: "100%", objectFit: "cover" }} alt="" />
                <button onClick={() => { setFiles(f => f.filter((_, j) => j !== i)); setPreviews(p => p.filter((_, j) => j !== i)); }}
                  style={{ position: "absolute", top: 3, right: 3, width: 18, height: 18, borderRadius: "50%", background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-secondary)", cursor: "pointer", fontSize: 12, color: "var(--color-text-primary)", display: "flex", alignItems: "center", justifyContent: "center" }}>×</button>
              </div>
            ))}
          </div>
        </div>
      )}

      <button style={s.btnPrimary} onClick={onNext} disabled={files.length === 0}>Continua →</button>
    </div>
  );
}

// ─── Step 2: Preferences ─────────────────────────────────────────────────────
function PrefsStep({ prefs, setPrefs, files, onBack, onSubmit, error, loading }) {
  return (
    <div style={s.wrap}>
      <p style={{ fontSize: 12, color: "var(--color-text-tertiary)", marginBottom: 4 }}>Passo 2 di 2</p>
      <h2 style={s.h2}>Preferenze progetto</h2>
      <p style={s.sub}>{files.length} foto caricate · le foto staged saranno generate da Imagen 3 per ogni stanza</p>

      {error && (
        <div style={{ ...s.card, background: "var(--color-background-danger)", borderColor: "var(--color-border-danger)", marginBottom: 16 }}>
          <p style={{ fontSize: 13, color: "var(--color-text-danger)", margin: 0 }}>{error}</p>
        </div>
      )}

      <div style={s.card}>
        <h3 style={s.h3}>Budget</h3>
        <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginBottom: 10 }}>
          <span style={{ fontSize: 26, fontWeight: 500 }}>€{prefs.budget.toLocaleString("it-IT")}</span>
        </div>
        <input type="range" min={500} max={10000} step={250} value={prefs.budget}
          onChange={e => setPrefs(p => ({ ...p, budget: +e.target.value }))} style={{ width: "100%" }} />
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "var(--color-text-tertiary)", marginTop: 4 }}>
          <span>€500</span><span>€10.000</span>
        </div>
      </div>

      <div style={s.card}>
        <h3 style={s.h3}>Stile</h3>
        <p style={{ ...s.label, marginBottom: 10 }}>
          Scrivi lo stile che vuoi (es. Japandi, Boho Chic, Art Deco…) oppure clicca un suggerimento
        </p>
        <input
          type="text"
          value={prefs.style}
          onChange={e => setPrefs(p => ({ ...p, style: e.target.value }))}
          placeholder="es. Scandinavo, Boho Chic, Industrial, Japandi…"
          style={{ width: "100%", fontSize: 14, marginBottom: 12 }}
        />
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
          {STYLE_SUGGESTIONS.map(sug => (
            <button
              key={sug}
              onClick={() => setPrefs(p => ({ ...p, style: sug }))}
              style={{
                ...s.btn,
                fontSize: 12,
                padding: "4px 12px",
                borderColor: prefs.style === sug
                  ? "var(--color-border-primary)"
                  : "var(--color-border-tertiary)",
                background: prefs.style === sug
                  ? "var(--color-background-secondary)"
                  : "var(--color-background-primary)",
                fontWeight: prefs.style === sug ? 500 : 400,
              }}
            >
              {sug}
            </button>
          ))}
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>
        <div style={s.card}>
          <h3 style={s.h3}>Città</h3>
          <select value={prefs.location} onChange={e => setPrefs(p => ({ ...p, location: e.target.value }))} style={{ width: "100%", fontSize: 14 }}>
            {CITIES.map(c => <option key={c}>{c}</option>)}
          </select>
        </div>
        <div style={{ ...s.card, gridColumn: "2/4" }}>
          <h3 style={s.h3}>Destinazione</h3>
          <div style={{ display: "flex", gap: 8 }}>
            {["STR", "Casa Vacanza"].map(d => (
              <button key={d} onClick={() => setPrefs(p => ({ ...p, destination: d }))}
                style={{ ...s.btn, flex: 1, borderColor: prefs.destination === d ? "var(--color-border-primary)" : "var(--color-border-tertiary)", background: prefs.destination === d ? "var(--color-background-secondary)" : "var(--color-background-primary)", fontWeight: prefs.destination === d ? 500 : 400, fontSize: 13 }}>{d}</button>
            ))}
          </div>
        </div>
      </div>

      <div style={s.card}>
        <h3 style={s.h3}>La tua email</h3>
        <p style={{ ...s.label, marginBottom: 8 }}>Il PDF con le foto staged e la scheda completa verrà inviato a questo indirizzo</p>
        <input type="email" value={prefs.email} onChange={e => setPrefs(p => ({ ...p, email: e.target.value }))}
          placeholder="tua@email.com" style={{ width: "100%", fontSize: 14 }} />
      </div>

      <div style={s.row}>
        <button style={s.btn} onClick={onBack}>← Indietro</button>
        <button style={s.btnPrimary} onClick={onSubmit} disabled={loading || !prefs.email}>
          {loading ? "Invio in corso…" : "Genera report →"}
        </button>
      </div>
    </div>
  );
}

// ─── Step 3: Processing ──────────────────────────────────────────────────────
function ProcessingStep({ job }) {
  const steps = [
    "Gemini analizza le foto…",
    "Imagen 3 genera le foto staged per ogni stanza…",
    "Compilazione scheda e preventivi…",
    "Generazione PDF…",
    "Invio email…",
  ];

  return (
    <div style={{ ...s.wrap, textAlign: "center", paddingTop: "3rem" }}>
      <div style={{ width: 40, height: 40, border: "2px solid var(--color-border-tertiary)", borderTop: "2px solid var(--color-text-primary)", borderRadius: "50%", margin: "0 auto 24px", animation: "spin 1s linear infinite" }} />
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      <h2 style={{ ...s.h2, marginBottom: 8 }}>Elaborazione in corso</h2>
      <p style={{ fontSize: 14, color: "var(--color-text-secondary)", marginBottom: 24 }}>{job?.step || "…"}</p>

      <div style={{ maxWidth: 400, margin: "0 auto", textAlign: "left" }}>
        <div style={{ height: 4, background: "var(--color-background-secondary)", borderRadius: 2, marginBottom: 20, overflow: "hidden" }}>
          <div style={{ height: "100%", background: "var(--color-text-primary)", borderRadius: 2, width: `${job?.progress || 0}%`, transition: "width 0.6s ease" }} />
        </div>
        {steps.map((step, i) => {
          const done = (job?.progress || 0) > (i + 1) * 18;
          const active = !done && (job?.progress || 0) > i * 18;
          return (
            <div key={i} style={{ display: "flex", gap: 10, alignItems: "center", marginBottom: 8, opacity: done || active ? 1 : 0.3 }}>
              <div style={{ width: 16, height: 16, borderRadius: "50%", background: done ? "var(--color-text-success)" : active ? "var(--color-text-primary)" : "var(--color-border-secondary)", flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center" }}>
                {done && <span style={{ color: "var(--color-background-primary)", fontSize: 10 }}>✓</span>}
              </div>
              <span style={{ fontSize: 13, color: done ? "var(--color-text-success)" : active ? "var(--color-text-primary)" : "var(--color-text-tertiary)" }}>{step}</span>
            </div>
          );
        })}
      </div>

      <p style={{ fontSize: 12, color: "var(--color-text-tertiary)", marginTop: 28 }}>Può richiedere 1–3 minuti · Riceverai il PDF via email</p>
    </div>
  );
}

// ─── Step 4: Done ────────────────────────────────────────────────────────────
function DoneStep({ summary, email, onReset }) {
  return (
    <div style={{ ...s.wrap, textAlign: "center", paddingTop: "2.5rem" }}>
      <div style={{ width: 48, height: 48, borderRadius: "50%", background: "var(--color-background-success)", display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 20px", fontSize: 20 }}>✓</div>
      <h2 style={{ ...s.h2, textAlign: "center", marginBottom: 8 }}>Report inviato!</h2>
      <p style={{ fontSize: 14, color: "var(--color-text-secondary)", marginBottom: 28 }}>
        Controlla la casella <strong>{email}</strong> — trovi il PDF con le foto staged e la scheda completa.
      </p>

      {summary && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, maxWidth: 480, margin: "0 auto 28px", textAlign: "center" }}>
          {[
            ["Costo stimato", `€${Number(summary.totale_costi).toLocaleString("it-IT")}`],
            ["Incremento", summary.incremento],
            ["Stile", summary.titolo?.split(" ").slice(0, 2).join(" ")],
          ].map(([l, v]) => (
            <div key={l} style={{ background: "var(--color-background-secondary)", padding: "1rem", borderRadius: "var(--border-radius-md)" }}>
              <p style={{ fontSize: 12, color: "var(--color-text-secondary)", margin: "0 0 4px" }}>{l}</p>
              <p style={{ fontSize: 16, fontWeight: 500, margin: 0 }}>{v}</p>
            </div>
          ))}
        </div>
      )}

      <button style={s.btnPrimary} onClick={onReset}>Analizza un altro appartamento</button>
    </div>
  );
}

// ─── Main App ────────────────────────────────────────────────────────────────
export default function App() {
  const [step, setStep] = useState(0);
  const [files, setFiles] = useState([]);
  const [prefs, setPrefs] = useState({ budget: 2500, style: "Scandinavo", location: "Milano", destination: "STR", email: "" });
  const [jobId, setJobId] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const pollRef = useRef(null);

  // Poll job status
  useEffect(() => {
    if (!jobId) return;
    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/status/${jobId}`);
        const data = await res.json();
        setJobStatus(data);
        if (data.status === "completed") {
          clearInterval(pollRef.current);
          setStep(4);
        } else if (data.status === "error") {
          clearInterval(pollRef.current);
          setError(`Errore: ${data.error}`);
          setStep(2);
          setLoading(false);
        }
      } catch { /* network hiccup, retry */ }
    }, 2500);
    return () => clearInterval(pollRef.current);
  }, [jobId]);

  const handleSubmit = async () => {
    if (!prefs.email) return;
    setLoading(true);
    setError(null);

    try {
      const fd = new FormData();
      files.forEach(f => fd.append("photos", f));
      fd.append("budget", prefs.budget);
      fd.append("style", prefs.style);
      fd.append("location", prefs.location);
      fd.append("destination", prefs.destination);
      fd.append("email", prefs.email);

      const res = await fetch(`${API_BASE}/analyze`, { method: "POST", body: fd });
      if (!res.ok) throw new Error(`Server error ${res.status}`);
      const data = await res.json();
      setJobId(data.job_id);
      setStep(3);
    } catch (err) {
      setError(`Errore di connessione: ${err.message}`);
      setLoading(false);
    }
  };

  const reset = () => {
    clearInterval(pollRef.current);
    setStep(0); setFiles([]); setJobId(null); setJobStatus(null); setError(null); setLoading(false);
    setPrefs({ budget: 2500, style: "Scandinavo", location: "Milano", destination: "STR", email: "" });
  };

  if (step === 0) return <IntroStep onStart={() => setStep(1)} />;
  if (step === 1) return <PhotoStep files={files} setFiles={setFiles} onNext={() => setStep(2)} />;
  if (step === 2) return <PrefsStep prefs={prefs} setPrefs={setPrefs} files={files} onBack={() => setStep(1)} onSubmit={handleSubmit} error={error} loading={loading} />;
  if (step === 3) return <ProcessingStep job={jobStatus} />;
  if (step === 4) return <DoneStep summary={jobStatus?.summary} email={prefs.email} onReset={reset} />;
  return null;
}
