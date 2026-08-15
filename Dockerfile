FROM python:3.11-slim

# System deps needed by pdfplumber/PyMuPDF/lxml/spacy at build+runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pull the spaCy model used by the extractors
RUN python -m spacy download en_core_web_sm

# Copy the rest of the project
COPY . .

# Data dirs the pipeline writes into
RUN mkdir -p data/processed data/raw

ENV PYTHONUNBUFFERED=1 \
    OLLAMA_HOST=http://ollama:11434

EXPOSE 8765

CMD ["python", "web/server.py"]