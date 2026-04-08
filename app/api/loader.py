"""
Load trained model pipeline and metadata at API startup.
"""

import joblib
import json
from pathlib import Path
from typing import Optional


# TODO: Implement model loading
# - Load pipeline from artifacts/decision_tree_v1.pkl
# - Load metadata from artifacts/model_metadata_v1.json
# - Cache in module-level variables for reuse across requests
