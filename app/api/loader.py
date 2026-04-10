"""Load trained model pipeline at startup.

Cached at module level for zero per-request overhead.
"""

import joblib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.joblib"
METADATA_PATH = PROJECT_ROOT / "artifacts" / "model_metadata.json"

_model = None
_metadata = None

def load_model():
    global _model
    if _model is not None:
        return _model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    try:
        _model = joblib.load(MODEL_PATH)
        return _model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

def load_metadata():
    global _metadata
    if _metadata is not None:
        return _metadata
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata not found at {METADATA_PATH}")
    try:
        with open(METADATA_PATH, "r") as f:
            _metadata = json.load(f)
        return _metadata
    except Exception as e:
        raise RuntimeError(f"Failed to load metadata: {str(e)}")

def verify_artifacts():
    load_model()
    load_metadata()
