# backend/inference.py
"""
ModelServer: loads tfidf + sklearn model and provides
predict(text) -> (label, probability, top_tokens)
"""

import os
import joblib
import json
import numpy as np
from typing import Tuple, List, Optional
from config import TFIDF_PATH, MODEL_PATH, METADATA_PATH, MODEL_VERSION
import logging

logger = logging.getLogger("model-server")

# 🔧 Adjusted threshold to reduce false FAKE predictions
FAKE_THRESHOLD = 0.6


class ModelNotLoadedError(RuntimeError):
    pass


class ModelServer:
    def __init__(self):   # ✅ FIXED
        self.tfidf = None
        self.model = None
        self.model_version = None
        self.loaded = False
        self._load_artifacts()

    def _load_artifacts(self):
        try:
            if os.path.exists(TFIDF_PATH) and os.path.exists(MODEL_PATH):
                logger.info("Loading TF-IDF vectorizer from %s", TFIDF_PATH)
                self.tfidf = joblib.load(TFIDF_PATH)

                logger.info("Loading model from %s", MODEL_PATH)
                self.model = joblib.load(MODEL_PATH)

                if os.path.exists(METADATA_PATH):
                    try:
                        with open(METADATA_PATH, "r", encoding="utf-8") as fh:
                            meta = json.load(fh)
                            self.model_version = meta.get(
                                "model_version", MODEL_VERSION
                            )
                    except Exception:
                        self.model_version = MODEL_VERSION
                else:
                    self.model_version = MODEL_VERSION

                self.loaded = True
                logger.info(
                    "Model server loaded successfully. version=%s",
                    self.model_version,
                )
            else:
                logger.warning(
                    "Model artifacts not found. TFIDF: %s, Model: %s",
                    TFIDF_PATH,
                    MODEL_PATH,
                )
                self.loaded = False
        except Exception as e:
            logger.exception("Failed to load model artifacts: %s", e)
            self.loaded = False

    def predict(
        self, preprocessed_text: str
    ) -> Tuple[str, float, Optional[List[str]]]:
        """
        preprocessed_text is expected to be cleaned text
        Returns (label, probability, top_tokens)
        """

        if not self.loaded or self.tfidf is None or self.model is None:
            raise ModelNotLoadedError("Model artifacts not loaded")

        # Vectorize input
        X = self.tfidf.transform([preprocessed_text])

        # Predict probabilities
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[0]

            classes = list(self.model.classes_)
            fake_idx = classes.index("FAKE")
            real_idx = classes.index("REAL")

            fake_prob = float(probs[fake_idx])
            real_prob = float(probs[real_idx])
            
            print("fake probability", fake_prob);
            print(real_prob);

            if fake_prob >= FAKE_THRESHOLD:
                label_val = "FAKE"
                prob = fake_prob
                pred_idx = fake_idx
            else:
                label_val = "REAL"
                prob = real_prob
                pred_idx = real_idx
        else:
            label_val = str(self.model.predict(X)[0])
            prob = 1.0
            pred_idx = 0

        # Extract top contributing tokens
        top_tokens = None
        try:
            if hasattr(self.model, "coef_") and hasattr(
                self.tfidf, "get_feature_names_out"
            ):
                feature_names = self.tfidf.get_feature_names_out()
                coefs = self.model.coef_

                if coefs.ndim > 1:
                    coefs = coefs[pred_idx]

                top_idx = np.argsort(coefs)[-6:][::-1]
                top_tokens = [feature_names[i] for i in top_idx]
        except Exception:
            top_tokens = None

        return label_val, prob, top_tokens   # ✅ FIXED