"""
Email Service — SendGrid
Sends the PDF report as an email attachment.
"""
import base64
import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import (
    Mail, Attachment, FileContent, FileName, FileType, Disposition,
)

SENDGRID_API_KEY = os.environ["SENDGRID_API_KEY"]
FROM_EMAIL = os.environ.get("FROM_EMAIL", "noreply@homestager.ai")
FROM_NAME = os.environ.get("FROM_NAME", "HomeStager AI")


def send_report_email(to_email: str, pdf_bytes: bytes, analysis: dict, prefs: dict):
    titolo = analysis.get("titolo_annuncio_suggerito", "Il tuo appartamento")
    incremento = analysis.get("tariffe", {}).get("incremento_percentuale", "—")
    totale = analysis.get("riepilogo_costi", {}).get("totale", 0)
    location = prefs.get("location", "")

    # Format totale as integer with dot separator (Italian style)
    totale_fmt = f"{int(totale):,}".replace(",", ".")

    html_body = f"""
    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
                max-width:600px;margin:0 auto;color:#2c2c2a">

      <div style="padding:32px 0 16px;border-bottom:2px solid #2c2c2a">
        <p style="font-size:11px;letter-spacing:2px;text-transform:uppercase;
                  color:#888;margin:0 0 10px">HomeStager AI</p>
        <h1 style="font-size:24px;margin:0 0 6px;font-weight:700">Il tuo report è pronto</h1>
        <p style="color:#666;margin:0">{titolo}</p>
      </div>

      <div style="padding:28px 0">
        <p style="margin:0 0 20px">
          Ciao! Ecco la scheda di home staging per il tuo appartamento a
          <strong>{location}</strong>. Trovi il PDF completo in allegato.
        </p>

        <table style="width:100%;border-collapse:collapse;margin:0 0 24px">
          <tr>
            <td style="background:#f5f5f3;padding:16px;border-radius:6px;
                       text-align:center;width:50%">
              <div style="font-size:11px;color:#888;text-transform:uppercase;margin-bottom:6px">
                Costo restyling stimato
              </div>
              <div style="font-size:22px;font-weight:700">€{totale_fmt}</div>
            </td>
            <td style="width:12px"></td>
            <td style="background:#2c2c2a;color:#fff;padding:16px;border-radius:6px;
                       text-align:center;width:50%">
              <div style="font-size:11px;color:#aaa;text-transform:uppercase;margin-bottom:6px">
                Incremento tariffa stimato
              </div>
              <div style="font-size:22px;font-weight:700">{incremento}</div>
            </td>
          </tr>
        </table>

        <p style="margin:0 0 12px">Nel PDF allegato trovi:</p>
        <ul style="color:#555;line-height:2;margin:0;padding-left:20px">
          <li>Foto <strong>Prima e Dopo</strong> generate da Imagen 3 per ogni stanza</li>
          <li>Piano interventi stanza per stanza con costi localizzati per {location}</li>
          <li>Lista acquisti con negozi e budget suddiviso per categoria</li>
          <li>Titolo e highlights ottimizzati per Airbnb/Booking</li>
          <li>Analisi ROI dell'investimento</li>
        </ul>
      </div>

      <div style="border-top:1px solid #eee;padding-top:16px;
                  font-size:11px;color:#bbb">
        Generato da HomeStager AI · Gemini 1.5 Pro + Imagen 3
      </div>
    </div>
    """

    message = Mail(
        from_email=(FROM_EMAIL, FROM_NAME),
        to_emails=to_email,
        subject=f"Home Staging Report — {titolo}",
        html_content=html_body,
    )

    filename_safe = location.lower().replace(" ", "_").replace("'", "")
    attachment = Attachment(
        file_content=FileContent(base64.b64encode(pdf_bytes).decode()),
        file_name=FileName(f"homestaging_{filename_safe}.pdf"),
        file_type=FileType("application/pdf"),
        disposition=Disposition("attachment"),
    )
    message.attachment = attachment

    sg = SendGridAPIClient(SENDGRID_API_KEY)
    response = sg.send(message)

    if response.status_code not in (200, 201, 202):
        raise RuntimeError(
            f"SendGrid error {response.status_code}: {response.body}"
        )
