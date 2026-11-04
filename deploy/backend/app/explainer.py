import time
import base64
import io
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from src.features import extract_features, REGIONS
from src.preprocessing import preprocess_bundle


def _distribute_uniform() -> dict:
    return {r: 1.0 / len(REGIONS) for r in REGIONS}


class RadiographExplainer:
    """Explicabilidad para el modelo lineal v2 (LR sobre textura).

    - Nivel 1: contribucion de cada feature al log-odds de TB, agregada por
      region anatomica mediante pesos espaciales.
    - Nivel 3: mapa de oclusion (6x6) rapido porque solo re-extrae 96 features.
    """

    def __init__(self, pipeline, feat_cols, selected_names=None):
        self.pipeline = pipeline
        self.scaler = pipeline.named_steps["scaler"]
        self.selector = pipeline.named_steps["select"]
        self.clf = pipeline.named_steps["clf"]
        self.feat_cols = feat_cols
        self.region_weights = self._build_region_weights()

    def _build_region_weights(self) -> dict:
        weights = {}
        sel_idx = self.selector.get_support(indices=True)
        for idx, name in enumerate(self.feat_cols):
            if name[:2] in REGIONS and name[2] == "_":
                w = {r: 0.0 for r in REGIONS}
                w[name[:2]] = 1.0
            else:
                w = _distribute_uniform()
            weights[idx] = w
        self._selected_indices = sel_idx
        return weights

    # --- NIVEL 1: contribucion lineal -> 6 regiones ---
    def explain_level1(self, vec_96: np.ndarray, img_thorax: np.ndarray) -> dict:
        scaled = self.scaler.transform(vec_96)[0]
        coefs = self.clf.coef_[0]
        sel_idx = self._selected_indices

        # Contribucion local = coef * valor_escalado (en escala de log-odds)
        contributions = []
        for j, feat_idx in enumerate(sel_idx):
            contrib = coefs[j] * scaled[feat_idx]
            contributions.append((feat_idx, contrib))
        contributions.sort(key=lambda x: -abs(x[1]))

        # Mapa de regiones
        region_importance = {r: 0.0 for r in REGIONS}
        for feat_idx, contrib in contributions:
            w = self.region_weights.get(feat_idx)
            if w is None:
                continue
            for rname, rw in w.items():
                region_importance[rname] += abs(contrib) * rw

        total = sum(region_importance.values())
        regions = {}
        for r, v in region_importance.items():
            regions[r] = round(v / total, 4) if total > 0 else 0.0

        top_features = []
        for feat_idx, contrib in contributions[:10]:
            top_features.append({
                "name": self.feat_cols[feat_idx],
                "importance": round(abs(contrib), 5),
                "direction": "aumenta TB" if contrib > 0 else "reduce TB",
                "group": _grupo(self.feat_cols[feat_idx]),
            })

        families = {}
        for feat_idx, contrib in contributions:
            g = _grupo(self.feat_cols[feat_idx])
            families[g] = families.get(g, 0.0) + abs(contrib)
        tf = sum(families.values())
        families = {k: round(v / tf, 4) for k, v in families.items()} if tf > 0 else {}

        heatmap_b64 = self._render_region_heatmap(img_thorax, regions)

        return {
            "regions": regions,
            "region_heatmap_base64": heatmap_b64,
            "top_features": top_features,
            "families": families,
        }

    def _render_region_heatmap(self, img_256, regions):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img_256, cmap="gray", aspect="equal")
        ax.axis("off")
        max_v = max(regions.values()) if regions else 1.0
        for name, (rx, ry, rw, rh) in REGIONS.items():
            imp = regions.get(name, 0.0)
            norm = imp / max_v if max_v > 0 else 0.0
            color = (1.0, 1.0 - norm, 1.0 - norm)
            ax.add_patch(Rectangle((rx, ry), rw, rh, fill=True, color=color, alpha=0.4, lw=0))
            ax.text(rx + rw / 2, ry + rh / 2, f"{name}\n{imp:.0%}", ha="center", va="center",
                    fontsize=7, color="white" if norm > 0.4 else "black", fontweight="bold")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

    # --- NIVEL 3: oclusion 6x6 sobre el recorte torácico (256x256) ---
    def run_occlusion_sync_polling(self, temp_dir: str, session_id: str, progress_dict: dict) -> tuple:
        t0 = time.time()
        original_path = str(Path(temp_dir) / "original.png")

        bundle = preprocess_bundle(original_path)
        if bundle is None:
            raise RuntimeError("preprocess_bundle returned None")
        img_256, mask_256 = bundle["thorax"]
        quality = bundle["quality"]

        base_prob = self._predict_tb(original_path, bundle=bundle)
        g = 6
        cuts = np.linspace(0, 256, g + 1).astype(int)
        heatmap = [[0.0] * g for _ in range(g)]

        # Los componentes del bundle que no cambian entre celdas:
        context = bundle["context"]
        strict = bundle["strict"]
        strict_harm = bundle["strict_harmonized"]
        bbox = bundle["bbox"]

        for gy in range(g):
            for gx in range(g):
                y0, y1 = int(cuts[gy]), int(cuts[gy + 1])
                x0, x1 = int(cuts[gx]), int(cuts[gx + 1])
                occ = img_256.copy()
                occ[y0:y1, x0:x1] = 0

                fake_bundle = {
                    "thorax": (occ, mask_256),
                    "context": context,
                    "strict": strict,
                    "strict_harmonized": strict_harm,
                    "quality": quality,
                    "bbox": bbox,
                }
                try:
                    prob_occ = self._predict_tb(original_path, bundle=fake_bundle)
                    heatmap[gy][gx] = round(max(base_prob - prob_occ, 0.0), 6)
                except Exception:
                    heatmap[gy][gx] = 0.0

                progress_dict[session_id] = {
                    "current": gy * g + gx + 1, "total": g * g,
                    "detail": f"Celda ({gy},{gx})",
                }

        overlay_b64 = self._render_occlusion_overlay(img_256, heatmap)
        return heatmap, overlay_b64, time.time() - t0

    def _predict_tb(self, path: str, bundle: dict = None) -> float:
        row = extract_features(path, label=0, class_name="x",
                               include_hog=False, include_phog=False,
                               include_context=False, include_shape=False,
                               bundle=bundle)
        if row is None:
            return 0.0
        vec = np.array([row[k] for k in self.feat_cols], dtype=np.float64).reshape(1, -1)
        return float(self.pipeline.predict_proba(vec)[0, 1])

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
    if name in ("mean", "std", "contrast", "entropy"):
        return "Intensidad global"
    if name[:2] in REGIONS and name[2] == "_":
        return "Intensidad regional"
    if name.startswith("glcm_"):
        return "GLCM"
    if name.startswith("lbp_"):
        return "LBP"
    if name.startswith("gabor_"):
        return "Gabor"
    return "Otra"