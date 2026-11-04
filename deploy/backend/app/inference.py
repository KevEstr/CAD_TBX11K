import json
from pathlib import Path
import joblib
import numpy as np

from app.config import MODEL_PATH, MANIFEST_PATH, FEAT_COLS_PATH, TB_THRESHOLD

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from src.features import extract_features
from src.preprocessing import preprocess_bundle
from src.modeling import prior_correction


class InferencePipeline:
    """Pipeline de inferencia v2 (LR sobre 96 features de textura).

    Replica el pipeline de entrenamiento: recorte toracico (sin ceros) + BF +
    CLAHE, extrae 96 features de textura y aplica StandardScaler -> SelectKBest
    -> LogisticRegression (binario: 0=no TB, 1=TB).

    El modelo usa ``class_weight='balanced'``, por lo que sus probabilidades
    estan en la escala balanceada (prior 0.5). Para umbral y visualizacion se
    corrigen a la prevalencia real con ``prior_correction``.
    """

    def __init__(self):
        self.pipeline = joblib.load(str(MODEL_PATH))
        with open(str(MANIFEST_PATH), encoding="utf-8") as f:
            self.manifest = json.load(f)
        with open(str(FEAT_COLS_PATH), encoding="utf-8") as f:
            self.feat_cols = json.load(f)
        self.scaler = self.pipeline.named_steps["scaler"]
        self.selector = self.pipeline.named_steps["select"]
        self.clf = self.pipeline.named_steps["clf"]
        self.threshold = float(self.manifest.get("threshold_youden", TB_THRESHOLD))
        self.prevalence = float(self.manifest.get("prevalence", 0.5))

    def extract(self, image_path: str, bundle: dict = None) -> np.ndarray:
        row = extract_features(
            image_path, label=0, class_name="x",
            include_hog=False, include_phog=False,
            include_context=False, include_shape=False,
            bundle=bundle,
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

        vec = self.extract(str(original_path), bundle=bundle)
        proba = self.pipeline.predict_proba(vec)[0]
        p_raw = float(proba[1])
        p_cal = float(prior_correction(np.array([p_raw]), self.prevalence)[0])
        pred_class_id = int(p_cal >= self.threshold)

        return {
            "pred_class_id": pred_class_id,
            "pred_class": "tb" if pred_class_id == 1 else "no_tb",
            "probabilities": {
                "no_tb": 1.0 - p_cal,
                "tb": p_cal,
            },
            "probabilities_raw": {
                "no_tb": 1.0 - p_raw,
                "tb": p_raw,
            },
            "vec_96": vec,
            "img_thorax": img_thorax,
            "mask_thorax": mask_thorax,
            "original_path": str(original_path),
            "threshold": self.threshold,
            "prevalence": self.prevalence,
        }