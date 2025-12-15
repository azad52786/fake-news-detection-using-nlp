# backend/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system deps for nltk
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Create NLTK data directory and download required corpora at build time (best-effort)
RUN python -m nltk.downloader -d /usr/share/nltk_data wordnet stopwords punkt

COPY . /app

ENV NLTK_DATA=/usr/share/nltk_data
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
