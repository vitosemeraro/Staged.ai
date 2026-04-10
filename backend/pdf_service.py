"""
PDF Service — WeasyPrint + Jinja2
Produces a multi-page A4 PDF with:
  - Cover page (metadata + tariffe)
  - Valutazione generale
  - One page per room: PRIMA / DOPO photos + interventi + costi
  - Riepilogo costi + Piano acquisti
  - Annuncio ottimizzato + ROI
"""
import base64
from datetime import date
from weasyprint import HTML, CSS
from jinja2 import Template
from ai_service import compress_image

PRIORITY_COLOR = {
    "alta": "#E24B4A",
    "media": "#BA7517",
    "bassa": "#639922",
}

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8"/>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Lato:wght@300;400;700&family=Playfair+Display:wght@700&display=swap');
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family:'Lato',sans-serif; color:#2c2c2a; font-size:11pt; line-height:1.6; }

  /* ── Cover ── */
  .cover { page-break-after:always; padding:60px 50px; min-height:297mm;
           display:flex; flex-direction:column; justify-content:space-between; }
  .brand  { font-size:9pt; letter-spacing:3px; text-transform:uppercase; color:#888; }
  .cover-title { font-family:'Playfair Display',serif; font-size:32pt; font-weight:700;
                 margin:40px 0 12px; line-height:1.1; }
  .cover-sub   { font-size:13pt; color:#555; margin-bottom:36px; }
  .cover-meta  { display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-bottom:36px; }
  .meta-box    { border:1px solid #ddd; padding:14px 18px; border-radius:6px; }
  .meta-label  { font-size:8pt; text-transform:uppercase; letter-spacing:1.5px; color:#888; margin-bottom:4px; }
  .meta-value  { font-size:14pt; font-weight:700; }
  .tariffe-grid { display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; margin:24px 0; }
  .tariffa-box  { background:#f5f5f3; padding:16px; border-radius:6px; text-align:center; }
  .tariffa-num  { font-size:18pt; font-weight:700; color:#2c2c2a; }
  .tariffa-label { font-size:8pt; color:#888; text-transform:uppercase; margin-top:4px; }
  .incremento-box { background:#2c2c2a; }
  .incremento-box .tariffa-num   { color:#fff; }
  .incremento-box .tariffa-label { color:#aaa; }
  .footer-cover { font-size:8pt; color:#bbb; }

  /* ── General section ── */
  .section { page-break-before:always; padding:50px; }
  .section-title { font-family:'Playfair Display',serif; font-size:22pt; margin-bottom:6px; }
  .divider { border:none; border-top:2px solid #2c2c2a; margin:10px 0 22px; width:40px; }
  .body-text { font-size:11pt; color:#444; margin-bottom:20px; line-height:1.8; }
  .two-col { display:grid; grid-template-columns:1fr 1fr; gap:24px; margin:20px 0; }
  .col-title { font-size:9pt; text-transform:uppercase; letter-spacing:1.5px;
               color:#888; margin-bottom:10px; font-weight:700; }
  .point { font-size:10pt; color:#444; margin-bottom:6px;
           padding-left:16px; position:relative; }
  .point::before { content:''; position:absolute; left:0; top:7px;
                   width:6px; height:6px; border-radius:50%; }
  .point.green::before  { background:#639922; }
  .point.amber::before  { background:#BA7517; }

  /* ── Room pages ── */
  .room-page { page-break-before:always; padding:50px; }
  .room-header { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:6px; }
  .room-name { font-family:'Playfair Display',serif; font-size:24pt; }
  .room-cost { font-size:14pt; font-weight:700; }
  .room-status { font-size:10.5pt; color:#777; margin-bottom:18px; }
  .photos-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:26px; }
  /* Layout C4/C5 */
  .staging-grid { display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; margin-bottom:20px; }
  .staging-col { }
  .staging-label { font-size:7.5pt; text-transform:uppercase; letter-spacing:1.5px;
                   color:#fff; font-weight:700; padding:4px 8px; border-radius:4px 4px 0 0; }
  .staging-img { width:100%; height:150px; object-fit:contain; background:#f0eeea;
                 display:block; border:1px solid #e8e8e8; border-top:none; border-radius:0 0 4px 4px; }
  .staging-placeholder { width:100%; height:150px; background:#f0eeea; display:flex;
                          align-items:center; justify-content:center; font-size:8pt; color:#ccc;
                          border:1px solid #e8e8e8; border-top:none; border-radius:0 0 4px 4px; }
  .staging-card-body { padding:6px 8px; border:1px solid #e8e8e8; border-top:none;
                        border-radius:0 0 4px 4px; background:#fafaf9; }
  .staging-voce { font-size:7.5pt; color:#555; padding:1px 0; display:flex;
                  justify-content:space-between; border-bottom:1px solid #f0f0f0; }
  .staging-voce:last-child { border-bottom:none; }
  .staging-total { font-size:8pt; font-weight:700; color:#2c2c2a;
                   padding-top:4px; text-align:right; }
  .photo-label { font-size:8pt; text-transform:uppercase; letter-spacing:1.5px;
                 color:#888; margin-bottom:5px; }
  .room-photo { width:100%; max-height:220px; object-fit:contain; background:#f5f5f3; border-radius:6px; display:block; }
  .photo-placeholder { width:100%; height:185px; background:#f0eeea; border-radius:6px;
                       display:flex; align-items:center; justify-content:center;
                       font-size:9pt; color:#bbb; }
  .int-section-title { font-size:9pt; text-transform:uppercase; letter-spacing:1.5px;
                       color:#888; margin-bottom:10px; font-weight:700; }
  .intervention { border-left:3px solid #eee; padding:8px 0 8px 14px; margin-bottom:10px; }
  .int-row   { display:flex; justify-content:space-between; align-items:flex-start; }
  .int-title { font-weight:700; font-size:10.5pt; }
  .int-badge { font-size:7.5pt; padding:2px 8px; border-radius:3px;
               color:#fff; font-weight:700; margin-left:8px; }
  .int-cost  { font-size:10pt; font-weight:700; white-space:nowrap; margin-left:12px; }
  .int-detail { font-size:10pt; color:#555; margin-top:3px; line-height:1.5; }
  .int-where  { font-size:9pt; color:#aaa; margin-top:2px; }

  /* ── Costs summary ── */
  .costs-section { page-break-before:always; padding:50px; }
  .cost-table { width:100%; border-collapse:collapse; margin:18px 0; font-size:10.5pt; }
  .cost-table td { padding:9px 0; border-bottom:1px solid #eee; }
  .cost-table td:last-child { text-align:right; font-weight:700; }
  .cost-total td { border-top:2px solid #2c2c2a; border-bottom:none;
                   font-size:13pt; font-weight:700; padding-top:12px; }
  .residuo td { color:#639922; font-size:10pt; border-bottom:none; }
  .nota-budget { font-size:10pt; color:#666; margin-top:8px; font-style:italic; }
  .shopping-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin:18px 0; }
  .shop-card { border:1px solid #eee; border-radius:6px; padding:12px 14px; }
  .shop-header { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:4px; }
  .shop-cat    { font-size:9pt; text-transform:uppercase; letter-spacing:1.5px;
                 color:#888; font-weight:700; }
  .shop-budget { font-size:11pt; font-weight:700; }
  .shop-store  { font-size:9pt; color:#aaa; margin-bottom:6px; }
  .shop-item   { font-size:9.5pt; color:#555; padding-left:10px; position:relative; margin-top:3px; }
  .shop-item::before { content:'–'; position:absolute; left:0; color:#ccc; }

  /* ── Listing page ── */
  .listing-section { page-break-before:always; padding:50px; }
  .listing-title-box { background:#2c2c2a; color:#fff; padding:20px 24px;
                       border-radius:8px; margin:18px 0; }
  .listing-title-text { font-family:'Playfair Display',serif; font-size:17pt; }
  .highlights { display:flex; flex-wrap:wrap; gap:8px; margin:16px 0; }
  .highlight  { background:#f0eeea; padding:5px 12px; border-radius:3px;
                font-size:9.5pt; color:#555; }
  .roi-box { border-left:4px solid #639922; padding:12px 16px;
             background:#f7fbf2; border-radius:0 6px 6px 0;
             margin-top:20px; font-size:10.5pt; color:#3B6D11; line-height:1.7; }

  /* ── Page numbers ── */
  @page { size:A4; margin:0;
    @bottom-center { content:counter(page); font-size:8pt; color:#ccc; margin-bottom:18px; } }
</style>
</head>
<body>

{# ──────────────── COVER ──────────────── #}
<div class="cover">
  <div>
    <div class="brand">Home Staging Report · AI-Powered</div>
    <div class="cover-title">{{ analysis.titolo_annuncio_suggerito or 'Report Home Staging' }}</div>
    <div class="cover-sub">
      Scheda professionale per
      {{ 'affitto breve (Airbnb / Booking)' if prefs.destination == 'STR' else 'casa vacanza' }}
    </div>

    <div class="cover-meta">
      <div class="meta-box">
        <div class="meta-label">Città</div>
        <div class="meta-value">{{ prefs.location }}</div>
      </div>
      <div class="meta-box">
        <div class="meta-label">Stile</div>
        <div class="meta-value">{{ prefs.style | title }}</div>
      </div>
      <div class="meta-box">
        <div class="meta-label">Budget totale</div>
        <div class="meta-value">€{{ "{:,}".format(prefs.budget).replace(",", ".") }}</div>
      </div>
      <div class="meta-box">
        <div class="meta-label">Data</div>
        <div class="meta-value">{{ today }}</div>
      </div>
    </div>

    <div class="tariffe-grid">
      <div class="tariffa-box">
        <div class="tariffa-num">{{ analysis.tariffe.attuale_notte }}</div>
        <div class="tariffa-label">Tariffa attuale / notte</div>
      </div>
      <div class="tariffa-box">
        <div class="tariffa-num">{{ analysis.tariffe.post_restyling_notte }}</div>
        <div class="tariffa-label">Dopo restyling / notte</div>
      </div>
      <div class="tariffa-box incremento-box">
        <div class="tariffa-num">{{ analysis.tariffe.incremento_percentuale }}</div>
        <div class="tariffa-label">Incremento stimato</div>
      </div>
    </div>
  </div>
  <div class="footer-cover">Generato con Gemini 1.5 Pro · Imagen 3 · {{ today }}</div>
</div>

{# ──────────────── VALUTAZIONE GENERALE ──────────────── #}
<div class="section">
  <div class="section-title">Valutazione generale</div>
  <hr class="divider"/>
  <p class="body-text">{{ analysis.valutazione_generale }}</p>
  {% if analysis.potenziale_str %}
  <p class="body-text">{{ analysis.potenziale_str }}</p>
  {% endif %}

  <div class="two-col">
    <div>
      <div class="col-title">Punti di forza</div>
      {% for p in analysis.punti_di_forza %}
      <div class="point green">{{ p }}</div>
      {% endfor %}
    </div>
    <div>
      <div class="col-title">Criticità da risolvere</div>
      {% for p in analysis.criticita %}
      <div class="point amber">{{ p }}</div>
      {% endfor %}
    </div>
  </div>
</div>

{# ──────────────── ROOM PAGES ──────────────── #}
{% for room in analysis.stanze %}
<div class="room-page">
  <div class="room-header">
    <div class="room-name">{{ room.nome }}</div>
    <div class="room-cost">€{{ room.costo_totale_stanza }}</div>
  </div>
  <p class="room-status">{{ room.stato_attuale }}</p>

  {# ── 3 colonne: Originale | C4 Full | C5 Smart Full ── #}
  {% set sa = room.staged_approaches or {} %}
  {% set c4_esp = room.esperimenti_staged_map.get('C4_FULL', {}) %}
  {% set c5_esp = room.esperimenti_staged_map.get('C5_SMART_FULL', {}) %}

  <div class="staging-grid">

    {# Colonna 1: Originale #}
    <div class="staging-col">
      <div class="staging-label" style="background:#555;">Originale</div>
      {% if room.original_photo_b64 %}
      <img class="staging-img"
           src="data:{{ room.original_photo_mime }};base64,{{ room.original_photo_b64 }}"
           alt="Originale"/>
      {% else %}
      <div class="staging-placeholder">Non disponibile</div>
      {% endif %}
    </div>

    {# Colonna 2: C4 Full #}
    <div class="staging-col">
      <div class="staging-label" style="background:#8B5E9C;">C4 — Full Staging</div>
      {% if sa.get('C4_FULL') %}
      <img class="staging-img"
           src="data:image/png;base64,{{ sa['C4_FULL'] }}"
           alt="C4 Full"/>
      {% else %}
      <div class="staging-placeholder">Non disponibile</div>
      {% endif %}
      {% if c4_esp.get('interventi_lista') %}
      <div class="staging-card-body">
        {% for iv in c4_esp['interventi_lista'] %}
        <div class="staging-voce">
          <span>{{ iv.voce }}</span>
          <span>€{{ iv.costo }}</span>
        </div>
        {% endfor %}
        <div class="staging-total">Budget: €{{ c4_esp.get('costo_simulato', '—') }}</div>
      </div>
      {% endif %}
    </div>

    {# Colonna 3: C5 Smart Full #}
    <div class="staging-col">
      <div class="staging-label" style="background:#2E7D5E;">C5 — Smart Full ✦</div>
      {% if sa.get('C5_SMART_FULL') %}
      <img class="staging-img"
           src="data:image/png;base64,{{ sa['C5_SMART_FULL'] }}"
           alt="C5 Smart Full"/>
      {% else %}
      <div class="staging-placeholder">Non disponibile</div>
      {% endif %}
      {% if c5_esp.get('interventi_lista') %}
      <div class="staging-card-body">
        {% for iv in c5_esp['interventi_lista'] %}
        <div class="staging-voce">
          <span>{{ iv.voce }}</span>
          <span>€{{ iv.costo }}</span>
        </div>
        {% endfor %}
        <div class="staging-total">Budget: €{{ c5_esp.get('costo_simulato', '—') }}</div>
      </div>
      {% endif %}
    </div>

  </div>

  <div class="int-section-title">Interventi</div>
  {% for iv in room.interventi %}
  <div class="intervention" style="border-left-color:{{ priority_color(iv.priorita) }}">
    <div class="int-row">
      <div>
        <span class="int-title">{{ iv.titolo }}</span>
        <span class="int-badge" style="background:{{ priority_color(iv.priorita) }}">
          {{ iv.priorita }}
        </span>
      </div>
      <span class="int-cost">€{{ iv.costo_min }}–{{ iv.costo_max }}</span>
    </div>
    <div class="int-detail">{{ iv.dettaglio }}</div>
    {% if iv.dove_comprare %}
    <div class="int-where">Dove: {{ iv.dove_comprare }}</div>
    {% endif %}
  </div>
  {% endfor %}
</div>
{% endfor %}

{# ──────────────── RIEPILOGO COSTI ──────────────── #}
<div class="costs-section">
  <div class="section-title">Riepilogo costi</div>
  <hr class="divider"/>
  {% set rc = analysis.riepilogo_costi %}
  <table class="cost-table">
    <tr><td>Tinteggiatura professionale (manodopera + materiali)</td>
        <td>€{{ rc.manodopera_tinteggiatura }}</td></tr>
    <tr><td>Materiali pittura aggiuntivi</td>
        <td>€{{ rc.materiali_pittura }}</td></tr>
    <tr><td>Arredi e complementi</td>
        <td>€{{ rc.arredi_complementi }}</td></tr>
    <tr><td>Montaggio e varie</td>
        <td>€{{ rc.montaggio_varie }}</td></tr>
    <tr class="cost-total">
        <td>Totale stimato</td><td>€{{ rc.totale }}</td></tr>
    {% if rc.budget_residuo and rc.budget_residuo > 0 %}
    <tr class="residuo">
        <td>Budget residuo disponibile</td><td>€{{ rc.budget_residuo }}</td></tr>
    {% endif %}
  </table>
  {% if rc.nota_budget %}
  <p class="nota-budget">{{ rc.nota_budget }}</p>
  {% endif %}

  <div class="section-title" style="margin-top:34px;font-size:18pt">Piano acquisti</div>
  <hr class="divider"/>
  <div class="shopping-grid">
    {% for cat in analysis.piano_acquisti %}
    <div class="shop-card">
      <div class="shop-header">
        <span class="shop-cat">{{ cat.categoria }}</span>
        <span class="shop-budget">€{{ cat.budget_stimato }}</span>
      </div>
      <div class="shop-store">{{ cat.negozi_consigliati }}</div>
      {% for item in cat.articoli %}
      <div class="shop-item">{{ item }}</div>
      {% endfor %}
    </div>
    {% endfor %}
  </div>
</div>

{# ──────────────── ANNUNCIO + ROI ──────────────── #}
<div class="listing-section">
  <div class="section-title">Annuncio e strategia STR</div>
  <hr class="divider"/>
  <div class="listing-title-box">
    <div class="listing-title-text">{{ analysis.titolo_annuncio_suggerito }}</div>
  </div>
  <div class="highlights">
    {% for h in analysis.highlights_str %}
    <span class="highlight">{{ h }}</span>
    {% endfor %}
  </div>
  {% if analysis.roi_restyling %}
  <div class="roi-box">
    <strong>ROI stimato:</strong> {{ analysis.roi_restyling }}
  </div>
  {% endif %}
</div>

</body>
</html>"""


def generate_pdf(analysis: dict, prefs: dict, photos: list,
                staged_results: list | None = None) -> bytes:
    """
    Layout per stanza: 3 colonne — Originale | C4 Full | C5 Smart Full
    """
    for i, room in enumerate(analysis.get("stanze", [])):
        idx = room.get("indice_foto", 0)
        if idx < len(photos):
            raw_bytes = compress_image(photos[idx]["content"], max_width=1400, quality=82)
            room["original_photo_b64"]  = base64.b64encode(raw_bytes).decode()
            room["original_photo_mime"] = "image/jpeg"
        else:
            room.setdefault("original_photo_b64", None)
            room.setdefault("original_photo_mime", "image/jpeg")

        room["staged_approaches"] = (staged_results[i] or {}) if staged_results and i < len(staged_results) else {}
        room["esperimenti_staged_map"] = {
            esp["logic_id"]: esp
            for esp in room.get("esperimenti_staged", [])
            if "logic_id" in esp
        }

    html_str = Template(HTML_TEMPLATE).render(
        analysis=analysis,
        prefs=prefs,
        today=date.today().strftime("%d/%m/%Y"),
        priority_color=lambda p: PRIORITY_COLOR.get(p, "#888"),
    )

    pdf_bytes = HTML(string=html_str).write_pdf(
        stylesheets=[CSS(string="@page { size: A4; margin: 0; }")]
    )
    return pdf_bytes