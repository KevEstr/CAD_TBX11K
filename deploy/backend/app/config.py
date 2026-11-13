import os
from pathlib import Path

MODELS_DIR = Path(os.path.dirname(__file__)).parent / "models"
MODEL_PATH = MODELS_DIR / "08_logistic_regression.joblib"
MANIFEST_PATH = MODELS_DIR / "08_manifest.json"
SCHEMA_PATH = MODELS_DIR / "08_feature_schema.json"
THRESHOLD_PATH = MODELS_DIR / "08_umbral_tb.json"

MAX_UPLOAD_SIZE = 20 * 1024 * 1024
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
OCCLUSION_GRID = 6
SESSION_TTL = 300
CLASS_NAMES = ["health", "sick", "tb"]
TB_THRESHOLD = 0.333768
