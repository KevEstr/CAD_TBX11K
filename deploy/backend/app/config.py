import os
from pathlib import Path

MODELS_DIR = Path(os.path.dirname(__file__)).parent / "models"
MODEL_PATH = MODELS_DIR / "08_xgboost.joblib"
MANIFEST_PATH = MODELS_DIR / "08_manifest.json"
FEAT_COLS_PATH = MODELS_DIR / "feat_cols.json"

MAX_UPLOAD_SIZE = 20 * 1024 * 1024
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
OCCLUSION_GRID = 6
SESSION_TTL = 120
CLASS_NAMES = ["health", "sick", "tb"]