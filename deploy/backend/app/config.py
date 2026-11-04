import os
from pathlib import Path

MODELS_DIR = Path(os.path.dirname(__file__)).parent / "models"
MODEL_PATH = MODELS_DIR / "v2_lr_texture.joblib"
MANIFEST_PATH = MODELS_DIR / "v2_lr_texture_manifest.json"
FEAT_COLS_PATH = MODELS_DIR / "feat_cols.json"

MAX_UPLOAD_SIZE = 20 * 1024 * 1024
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
OCCLUSION_GRID = 6
SESSION_TTL = 300
CLASS_NAMES = ["no_tb", "tb"]
TB_THRESHOLD = 0.0809
