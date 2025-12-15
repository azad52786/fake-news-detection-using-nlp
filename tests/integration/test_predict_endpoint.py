# tests/integration/test_predict_endpoint.py
import sys
from pathlib import Path
import importlib
import json
import pytest

# Ensure backend dir is importable
ROOT = Path(__file__).resolve().parents[2]  # project-root/tests/integration -> go up two
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Import the FastAPI app module (app.py should be in backend/)
# When backend/ is on sys.path, the module name is "app"
app_module = importlib.import_module("app")
from fastapi.testclient import TestClient

client = TestClient(app_module.app)


def test_health_endpoint_reports_model_loaded_flag_false_or_true():
    # Health endpoint should return JSON with keys 'status' and 'model_loaded'
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "model_loaded" in data


def test_predict_endpoint_with_mocked_model_server(monkeypatch):
    """
    The actual backend instantiates a ModelServer at module import time.
    We monkeypatch the model_server on the app module to simulate a loaded model,
    so the test does not depend on saved artifacts.
    """

    # Create a simple fake predict function and assign it to model_server
    class DummyModelServer:
        def __init__(self):
            self.loaded = True
            self.model_version = "test_v0"

        def predict(self, preprocessed_text):
            # Return label, probability, top_tokens
            return "FAKE", 0.9234, ["token1", "token2", "token3"]

    # Replace the module-level model_server with our dummy
    monkeypatch.setattr(app_module, "model_server", DummyModelServer())

    payload = {"title": "Test Title", "content": "This is a short test content about something."}
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["label"] in ("FAKE", "REAL")
    assert 0.0 <= float(data["probability"]) <= 1.0
    assert "model_version" in data
    assert "created_at" in data
    # top_tokens should be present (our DummyModelServer returns them)
    assert isinstance(data.get("top_tokens"), list)