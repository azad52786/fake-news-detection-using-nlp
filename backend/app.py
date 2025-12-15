# backend/app.py
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import uuid
import logging

from config import MODEL_VERSION
import db
from preprocessing import preprocess_for_vectorizer
from inference import ModelServer, ModelNotLoadedError

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fake-news-backend")

app = FastAPI(title="Fake News Detection API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (OK for final-year project)
    allow_credentials=True,
    allow_methods=["*"],  # allow POST, GET, OPTIONS, etc.
    allow_headers=["*"],
)

# Init DB and model server
db.init_db()
model_server = ModelServer()  # attempts to load tfidf + model at startup


class PredictRequest(BaseModel):
    title: Optional[str] = Field(None, max_length=500)
    content: str = Field(..., min_length=1, max_length=20000)


class PredictResponse(BaseModel):
    prediction_id: str
    label: str
    probability: float
    model_version: str
    top_tokens: Optional[List[str]] = None
    created_at: str


@app.get("/api/v1/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_server.loaded,
        "model_version": model_server.model_version if model_server.loaded else None,
    }


@app.get("/api/v1/history")
def history(limit: int = 20):
    try:
        items = db.fetch_history(limit=limit)
        return {"items": items}
    except Exception as e:
        logger.exception("Error fetching history")
        raise HTTPException(status_code=500, detail=f"History fetch failed: {str(e)}")


@app.post("/api/v1/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Basic validation already handled by Pydantic
    text_for_model = req.title + " " + req.content if req.title else req.content
    text_for_model = text_for_model.strip()
    if not text_for_model:
        raise HTTPException(status_code=400, detail="Empty content after trimming.")

    if not model_server.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Try again later.")

    try:
        # Preprocess (returns string normalized for vectorizer)
        prepped = preprocess_for_vectorizer(text_for_model)
        label, prob, top_tokens = model_server.predict(prepped)
    except Exception as e:
        logger.exception("Error during prediction")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    response = PredictResponse(
        prediction_id=str(uuid.uuid4()),
        label=label,
        probability=round(float(prob), 4),
        model_version=model_server.model_version or MODEL_VERSION,
        top_tokens=top_tokens,
        created_at=datetime.utcnow().isoformat() + "Z",
    )

    # Persist history (best effort)
    try:
        db.insert_prediction(
            {
                "prediction_id": response.prediction_id,
                "title": req.title,
                "content": req.content,
                "label": response.label,
                "probability": response.probability,
                "model_version": response.model_version,
                "top_tokens": response.top_tokens,
                "created_at": response.created_at,
            }
        )
    except Exception:
        logger.exception("Failed to persist prediction history")

    return response
