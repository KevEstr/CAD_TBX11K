import json
from pathlib import Path

import joblib
import numpy as np

from app.config import (
    MODEL_PATH, MANIFEST_PATH, SCHEMA_PATH, THRESHOLD_PATH, TB_THRESHOLD,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from src.features import extract_features
from src.preprocessing import preprocess_bundle


class InferencePipeline:
    """Inferencia del modelo final del notebook 08 (3 clases, 1.832 features)."""

    def __init__(self):
        self.pipeline = joblib.load(str(MODEL_PATH))
        with open(str(MANIFEST_PATH), encoding="utf-8") as f:
            self.manifest = json.load(f)
        with open(str(SCHEMA_PATH), encoding="utf-8") as f:
            self.schema = json.load(f)
        self.feat_cols = list(self.schema["feature_columns"])
        self.class_names = list(self.schema["class_names"])
        self.threshold = TB_THRESHOLD
        if THRESHOLD_PATH.exists():
            with open(str(THRESHOLD_PATH), encoding="utf-8") as f:
                umbral = json.load(f)
            if umbral.get("modelo") == self.manifest.get("seleccionado_cv"):
                self.threshold = float(umbral.get("threshold_youden", TB_THRESHOLD))

    def extract(self, image_path: str) -> np.ndarray:
        row = extract_features(
            image_path, label=0, class_name="x",
            include_context=False, include_shape=False,
        )
        if row is None:
            raise RuntimeError("extract_features returned None")
        return np.array([row[k] for k in self.feat_cols], dtype=np.float64).reshape(1, -1)

    def predict(self, image_bytes: bytes, temp_dir: str) -> dict:
        original_path = Path(temp_dir) / "original.png"
        original_path.write_bytes(image_bytes)

        bundle = preprocess_bundle(str(original_path))
        if bundle is None:
            raise RuntimeError("preprocess_bundle returned None")
        img_thorax, mask_thorax = bundle["thorax"]

        vec = self.extract(str(original_path))
        proba = np.asarray(self.pipeline.predict_proba(vec))[0]
        pred_id = int(np.argmax(proba))
        idx_tb = self.class_names.index("tb")
        prob_tb = float(proba[idx_tb])
        pred_class = self.class_names[pred_id]

        return {
            "pred_class_id": pred_id,
            "pred_class": pred_class,
            "probabilities": {
                nombre: float(proba[i])
                for i, nombre in enumerate(self.class_names)
            },
            "probabilities_raw": {
                nombre: float(proba[i])
                for i, nombre in enumerate(self.class_names)
            },
            "prob_tb": prob_tb,
            "tb_positive": bool(prob_tb >= self.threshold),
            "vec": vec,
            "img_thorax": img_thorax,
            "mask_thorax": mask_thorax,
            "original_path": str(original_path),
            "threshold": self.threshold,
        }
