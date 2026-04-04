FROM python:3.11-slim

# System dependencies:
#   WeasyPrint  → libpango, libcairo, libgdk-pixbuf, libffi, fonts
#   Pillow      → libjpeg, zlib, libpng  (JPEG/PNG encode-decode)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libpangocairo-1.0-0 \
    libcairo2 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    shared-mime-info \
    fonts-liberation \
    libjpeg62-turbo \
    zlib1g \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Cloud Run injetta la variabile PORT; default 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
