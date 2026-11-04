import json
import tempfile
from pathlib import Path
import joblib
import numpy as np
from app.config import MODEL_PATH, MANIFEST_PATH, FEAT_COLS_PATH

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from src.preprocessing import preprocess
from src.features import extract_features


class InferencePipeline:
    def __init__(self):
        self.pipeline = joblib.load(str(MODEL_PATH))
        with open(str(MANIFEST_PATH)) as f:
            self.manifest = json.load(f)
        with open(str(FEAT_COLS_PATH)) as f:
            self.feat_cols_3300 = json.load(f)
        self.selector = self.pipeline.named_steps["selector"]
        self.clf = self.pipeline.named_steps["clf"]
        self.selected_names = [self.feat_cols_3300[i] for i in self.selector.get_support(indices=True)]

    def predict(self, image_bytes: bytes, temp_dir: str) -> dict:
        original_path = Path(temp_dir) / "original.png"
        original_path.write_bytes(image_bytes)

        img_preprocessed = preprocess(str(original_path))
        if img_preprocessed is None:
            raise RuntimeError("preprocess returned None")

        feat_dict = extract_features(str(original_path), label=0, class_name="health")
        if feat_dict is None:
            raise RuntimeError("extract_features returned None")

        vec_3300 = np.array([feat_dict[k] for k in self.feat_cols_3300], dtype=np.float32).reshape(1, -1)
        vec_2000 = self.selector.transform(vec_3300)
        proba = self.clf.predict_proba(vec_2000)[0]
        pred_class_id = int(np.argmax(proba))

        return {
            "pred_class_id": pred_class_id,
            "pred_class": ["health", "sick", "tb"][pred_class_id],
            "probabilities": {
                "health": float(proba[0]),
                "sick": float(proba[1]),
                "tb": float(proba[2]),
            },
            "vec_2000": vec_2000,
            "img_preprocessed": img_preprocessed,
            "original_path": str(original_path),
        }