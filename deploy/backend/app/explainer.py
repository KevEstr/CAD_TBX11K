import base64
import io
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from src.features import extract_features
from src.preprocessing import preprocess_bundle


class RadiographExplainer:
    """Explicabilidad del modelo lineal de 3 clases (textura + HOG)."""

    def __init__(self, pipeline, feat_cols, selected_names=None):
        self.pipeline = pipeline
        self.scaler = pipeline.named_steps["scaler"]
        self.selector = pipeline.named_steps["selector"]
        self.clf = pipeline.named_steps["clf"]
        self.feat_cols = feat_cols
        self.idx_tb = self.clf.classes_.tolist().index(2)
        self.sel_idx = self.selector.get_support(indices=True)
        self.sel_names = [self.feat_cols[i] for i in self.sel_idx]

    def explain_level1(self, vec: np.ndarray, img_thorax: np.ndarray) -> dict:
        x_sel = self.selector.transform(vec)
        x_scaled = self.scaler.transform(x_sel)[0]
        coefs = self.clf.coef_[self.idx_tb]
        contributions = coefs * x_scaled
        order = np.argsort(np.abs(contributions))[::-1]

        top_features = []
        for j in order[:10]:
            top_features.append({
                "name": self.sel_names[j],
                "importance": round(float(abs(contributions[j])), 5),
                "direction": "aumenta TB" if contributions[j] > 0 else "reduce TB",
                "group": _grupo(self.sel_names[j]),
            })

        families = {}
        for j, contrib in enumerate(contributions):
            g = _grupo(self.sel_names[j])
            families[g] = families.get(g, 0.0) + abs(float(contrib))
        total = sum(families.values())
        families = (
            {k: round(v / total, 4) for k, v in families.items()}
            if total > 0 else {}
        )
        return {"families": families, "top_features": top_features}

    def run_occlusion_sync_polling(
        self, temp_dir: str, session_id: str, progress_dict: dict
    ) -> tuple:
        t0 = time.time()
        original_path = str(Path(temp_dir) / "original.png")
        bundle = preprocess_bundle(original_path)
        if bundle is None:
            raise RuntimeError("preprocess_bundle returned None")
        img_256, _ = bundle["thorax"]

        base_prob = self._predict_tb(original_path)
        g = 6
        cuts = np.linspace(0, 256, g + 1).astype(int)
        heatmap = [[0.0] * g for _ in range(g)]
        fill = int(np.median(img_256))

        for gy in range(g):
            for gx in range(g):
                if not Path(temp_dir).exists():
                    raise RuntimeError("Session cancelled")
                y0, y1 = int(cuts[gy]), int(cuts[gy + 1])
                x0, x1 = int(cuts[gx]), int(cuts[gx + 1])
                occ = img_256.copy()
                occ[y0:y1, x0:x1] = fill
                occ_path = Path(temp_dir) / f"occ_{gy}_{gx}.png"
                import cv2
                ok = cv2.imwrite(str(occ_path), occ)
                if not ok:
                    raise RuntimeError("No se pudo escribir la celda de oclusion")
                try:
                    prob_occ = self._predict_tb(str(occ_path))
                    heatmap[gy][gx] = round(max(base_prob - prob_occ, 0.0), 6)
                except Exception:
                    heatmap[gy][gx] = 0.0
                finally:
                    occ_path.unlink(missing_ok=True)

                progress_dict[session_id] = {
                    "current": gy * g + gx + 1,
                    "total": g * g,
                    "detail": f"Celda ({gy},{gx})",
                }

        overlay_b64 = self._render_occlusion_overlay(img_256, heatmap)
        return heatmap, overlay_b64, time.time() - t0

    def _predict_tb(self, path: str) -> float:
        row = extract_features(
            path, label=0, class_name="x",
            include_context=False, include_shape=False,
        )
        if row is None:
            return 0.0
        vec = np.array([row[k] for k in self.feat_cols], dtype=np.float64).reshape(1, -1)
        return float(self.pipeline.predict_proba(vec)[0, self.idx_tb])

    def _render_occlusion_overlay(self, img_256, heatmap):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        heat_np = np.array(heatmap, dtype=float)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img_256, cmap="gray", aspect="equal")
        ax.imshow(heat_np, extent=[0, 256, 256, 0], cmap="magma", alpha=0.55, aspect="equal")
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def _grupo(name: str) -> str:
    if name.startswith("hog_"):
        return "HOG"
    if name.startswith(("glcm_", "lbp_", "gabor_")):
        return "Textura"
    return "Otra"
