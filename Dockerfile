# Dockerfile — 4CM + KlastroHeron FastAPI backend
FROM python:3.11-slim

WORKDIR /app

# OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-kor \
    poppler-utils libreoffice \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Upload/document parsing deps for management advisory mode
RUN pip install --no-cache-dir \
    python-multipart \
    pypdf pdfplumber pdf2image \
    pandas openpyxl xlrd \
    python-docx python-pptx \
    pillow pytesseract

# Download sentence-transformer model at build time
# (avoids slow first-request download in production)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# App code
COPY . .

# Create angry_agents structure
RUN bash setup_new_scenarios.sh && bash setup_scenario_files.sh

# Logs and temporary upload directory
RUN mkdir -p logs /app/tmp_uploads

EXPOSE 8000

COPY entrypoint.sh .
RUN chmod +x entrypoint.sh
CMD ["bash", "entrypoint.sh"]
