# backend/config.py
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Where you will place model artifacts after training
MODEL_ARTIFACTS_DIR = os.path.join(ROOT_DIR, "model_artifacts")
TFIDF_PATH = os.path.join(MODEL_ARTIFACTS_DIR, "tfidf.pkl")
MODEL_PATH = os.path.join(MODEL_ARTIFACTS_DIR, "model.pkl")
METADATA_PATH = os.path.join(MODEL_ARTIFACTS_DIR, "metadata.json")
HISTORY_DB_PATH = os.path.join(ROOT_DIR, "history.db")

# Model version default (overridden by metadata if available)
MODEL_VERSION = os.environ.get("MODEL_VERSION", "baseline_v0.1")

# Preprocessing config
MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", 20000))

# NLTK data path (optional)
NLTK_DATA_DIR = os.environ.get("NLTK_DATA_DIR", None)
