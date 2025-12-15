# Backend - Fake News Detection (MVP)

## Prerequisites
- Python 3.10+ (recommended)
- pip
- (Optional) Docker

## Setup (local, non-container)
1. Create virtualenv:
   python -m venv .venv
   source .venv/bin/activate   # on Windows use .venv\Scripts\activate

2. Install dependencies:
   pip install -r requirements.txt

3. Ensure NLTK corpora are present (if first run):
   python -m nltk.downloader wordnet stopwords punkt

4. Place trained artifacts into:
   backend/model_artifacts/tfidf.pkl
   backend/model_artifacts/model.pkl
   (Optional) backend/model_artifacts/metadata.json  # {"model_version": "baseline_v0.1"}

5. Run the API:
   uvicorn app:app --reload --port 8000

6. Health:
   GET http://127.0.0.1:8000/api/v1/health

7. Predict (POST):
   POST http://127.0.0.1:8000/api/v1/predict
   Headers: Content-Type: application/json
   Body:
   {
     "title": "Breaking: Example",
     "content": "This is an example news article text..."
   }

8. Example response:
   {
     "prediction_id": "uuid",
     "label": "FAKE",
     "probability": 0.934,
     "model_version": "baseline_v0.1",
     "top_tokens": ["government","fake","claim"],
     "created_at": "2025-12-13T12:00:00Z"
   }

## Docker (optional)
1. Build:
   docker build -t fake-news-backend:latest .

2. Run:
   docker run -p 8000:8000 -v $(pwd)/model_artifacts:/app/model_artifacts fake-news-backend:latest

## Notes
- The repository assumes you'll train a TF-IDF + sklearn model and save artifacts as joblib files.
- The preprocessing pipeline is deterministic and must be identical to the one used during training.
- If the model artifacts are missing, `/api/v1/health` will report `model_loaded: false` and `/api/v1/predict` will return 503.

-  python experiments/data/prepare_dataset.py
- python experiments/train_baseline.py --data-path experiments/data/data.csv --out-dir backend/model_artifacts --model-version baseline_v0.1


- pip install -r backend/requirements.txt
- python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000                 