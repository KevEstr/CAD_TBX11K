import time
import json
import base64
import io
import logging
import asyncio
from pathlib import Path
import cv2
import numpy as np
import shap

logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from src.preprocessing import preprocess
from src.features import extract_features
from src.features import REGIONS


def _grupo_feature(nombre: str) -> str:
    if nombre.startswith("hog_"):
        return "HOG"
    if nombre.startswith("phog_"):
        return "PHOG"
    if nombre.startswith("gabor_"):
        return "Gabor"
    if nombre.startswith("lbp_"):
        return "LBP"
    if nombre.startswith("glcm_"):
        return "GLCM"
    if nombre in ("mean", "std", "contrast", "entropy"):
        return "Intensidad global"
    if nombre[:2] in ("SI", "SD", "MI", "MD", "II", "ID"):
        return "Intensidad regional"
    return "Otra"


def _region_from_feature(nombre: str) -> str:
    if nombre.startswith("hog_") or nombre.startswith("phog_"):
        return "HOG_PHOG"
    if nombre[:2] in ("SI", "SD", "MI", "MD", "II", "ID"):
        return nombre[:2]
    return "GLOBAL"


class RadiographExplainer:
    def __init__(self, pipeline, feat_cols_3300, selected_names):
        self.pipeline = pipeline
        self.selector = pipeline.named_steps["selector"]
        self.clf = pipeline.named_steps["clf"]
        self.feat_cols_3300 = feat_cols_3300
        self.selected_names = selected_names
        self._explainer_shap = None
        self._session_store = None

        self._precompute_maps()

    def _precompute_maps(self):
        self._feature_to_group = {}
        self._feature_to_region = {}
        self._hog_block_map = {}
        for idx, name in enumerate(self.selected_names):
            group = _grupo_feature(name)
            self._feature_to_group[idx] = group
            self._feature_to_region[idx] = _region_from_feature(name)
            if group == "HOG":
                hog_idx = int(name.split("_")[1])
                block_idx = hog_idx // 36
                block_row = block_idx // 7
                block_col = block_idx % 7
                key = f"{block_row}_{block_col}"
                self._hog_block_map.setdefault(key, []).append(idx)

    def _lazy_explainer(self):
        if self._explainer_shap is None:
            self._explainer_shap = shap.TreeExplainer(self.clf)
        return self._explainer_shap

    def set_session_store(self, store):
        self._session_store = store

    def _compute_shap(self, vec_2000, predicted_class_id):
        explainer = self._lazy_explainer()
        sv = explainer.shap_values(vec_2000)
        return sv[0, :, predicted_class_id]

    # --- LEVEL 1: SHAP -> 6 anatomic regions ---
    def explain_level1(self, vec_2000, predicted_class_id, img_preprocessed):
        shap_class = self._compute_shap(vec_2000, predicted_class_id)

        region_importance = {}
        for name, (rx, ry, rw, rh) in REGIONS.items():
            region_importance[name] = 0.0
        global_sum = 0.0

        for idx in range(len(self.selected_names)):
            shap_val = abs(shap_class[idx])
            region = self._feature_to_region[idx]
            if region == "HOG_PHOG":
                for rn in REGIONS:
                    region_importance[rn] += shap_val / len(REGIONS)
                global_sum += shap_val
            elif region == "GLOBAL":
                for rn in REGIONS:
                    region_importance[rn] += shap_val / len(REGIONS)
                global_sum += shap_val
            else:
                region_importance[region] += shap_val
                global_sum += shap_val

        total = sum(region_importance.values())
        if total > 0:
            for k in region_importance:
                region_importance[k] = round(region_importance[k] / total, 4)

        top_idx = np.argsort(-np.abs(shap_class))[:10]
        top_features = []
        for tidx in top_idx:
            name = self.selected_names[tidx]
            top_features.append({
                "name": name,
                "importance": round(float(abs(shap_class[tidx])), 6),
                "group": self._feature_to_group.get(tidx, "?"),
            })

        family_imp = {}
        for idx in range(len(self.selected_names)):
            group = self._feature_to_group.get(idx, "Otra")
            family_imp[group] = family_imp.get(group, 0.0) + float(abs(shap_class[idx]))
        total_f = sum(family_imp.values())
        if total_f > 0:
            for k in family_imp:
                family_imp[k] = round(family_imp[k] / total_f, 4)

        heatmap_b64 = self._render_region_heatmap(img_preprocessed, region_importance)

        return {
            "regions": region_importance,
            "region_heatmap_base64": heatmap_b64,
            "top_features": top_features,
            "families": family_imp,
        }

    def _render_region_heatmap(self, img_256, region_importance):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img_256, cmap="gray", aspect="equal")
        ax.axis("off")

        max_imp = max(region_importance.values()) if region_importance else 1.0
        for name, (rx, ry, rw, rh) in REGIONS.items():
            imp = region_importance.get(name, 0.0)
            norm = imp / max_imp if max_imp > 0 else 0.0
            color = (1.0, 1.0 - norm, 1.0 - norm)
            ax.add_patch(Rectangle((rx, ry), rw, rh, fill=True, color=color, alpha=0.4, lw=0))
            ax.text(rx + rw / 2, ry + rh / 2, f"{name}\n{imp:.0%}", ha="center", va="center",
                    fontsize=7, color="white" if norm > 0.4 else "black", fontweight="bold")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

    # --- LEVEL 2: SHAP -> HOG 7x7 grid ---
    def explain_level2(self, vec_2000, predicted_class_id, img_preprocessed):
        shap_class = self._compute_shap(vec_2000, predicted_class_id)

        grid = [[0.0] * 7 for _ in range(7)]
        mask = [[False] * 7 for _ in range(7)]

        for key, feat_indices in self._hog_block_map.items():
            row_str, col_str = key.split("_")
            r, c = int(row_str), int(col_str)
            total_shap = sum(float(abs(shap_class[idx])) for idx in feat_indices)
            grid[r][c] = total_shap
            mask[r][c] = True

        total = sum(sum(row) for row in grid)
        if total > 0:
            for r in range(7):
                for c in range(7):
                    if mask[r][c]:
                        grid[r][c] = round(grid[r][c] / total, 4)

        overlay_b64 = self._render_hog_overlay(img_preprocessed, grid, mask)

        phog_quadrants = [0.0, 0.0, 0.0, 0.0]
        phog_counts = [0, 0, 0, 0]
        for idx in range(len(self.selected_names)):
            if self._feature_to_group.get(idx) == "PHOG":
                name = self.selected_names[idx]
                if name.startswith("phog_l0_"):
                    continue
                offset = int(name.split("_")[-1])
                quadrant = offset // 288
                if quadrant < 4:
                    phog_quadrants[quadrant] += float(abs(shap_class[idx]))
                    phog_counts[quadrant] += 1
        for q in range(4):
            if phog_counts[q] > 0:
                phog_quadrants[q] = round(phog_quadrants[q] / phog_counts[q], 6)

        return {
            "hog_heatmap_7x7": grid,
            "hog_overlay_base64": overlay_b64,
            "phog_quadrants": phog_quadrants,
        }

    def _render_hog_overlay(self, img_256, grid, mask):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        heatmap = np.zeros((7, 7))
        for r in range(7):
            for c in range(7):
                heatmap[r, c] = grid[r][c] if mask[r][c] else np.nan

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img_256, cmap="gray", aspect="equal")
        cmap = plt.cm.magma.copy()
        cmap.set_bad(color="gray")
        ax.imshow(heatmap, extent=[0, 256, 256, 0], cmap=cmap, alpha=0.5, aspect="equal")
        ax.axis("off")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

    # --- LEVEL 3: OCCLUSION 6x6 (runs in thread executor) ---
    def run_occlusion_sync(self, temp_dir: str, session_id: str, ws_manager, main_loop):
        logger.info("Occlusion START: session=%s temp_dir=%s", session_id, temp_dir)
        t0 = time.time()
        original_path = Path(temp_dir) / "original.png"
        logger.info("Reading: %s exists=%s", original_path, original_path.exists())
        img_full = cv2.imread(str(original_path), cv2.IMREAD_GRAYSCALE)
        if img_full is None:
            error_msg = "Cannot read full-res image at " + str(original_path)
            logger.error(error_msg)
            _send_ws_safe(ws_manager.send_error, session_id, error_msg, main_loop)
            return

        h, w = img_full.shape
        grid_n = 6
        cell_h, cell_w = h // grid_n, w // grid_n
        logger.info("Image: %dx%d, cells: %dx%d", h, w, cell_h, cell_w)

        img_256 = preprocess(str(original_path))
        if img_256 is None:
            raise RuntimeError("preprocess failed for base prediction")
        feat_dict = extract_features(str(original_path), label=2, class_name="tb")
        if feat_dict is None:
            raise RuntimeError("extract_features failed for base prediction")
        vec_3300 = np.array([feat_dict[k] for k in self.feat_cols_3300], dtype=np.float32).reshape(1, -1)
        vec_2000_base = self.selector.transform(vec_3300)
        prob_base = self.clf.predict_proba(vec_2000_base)[0, 2]

        heatmap = [[0.0] * grid_n for _ in range(grid_n)]
        temp_dir_obj = Path(temp_dir)

        def _send_ws(method, *args):
            try:
                asyncio.run_coroutine_threadsafe(method(*args), main_loop)
            except Exception:
                pass

        for gy in range(grid_n):
            for gx in range(grid_n):
                occ = img_full.copy()
                y0, y1 = gy * cell_h, (gy + 1) * cell_h
                x0, x1 = gx * cell_w, (gx + 1) * cell_w
                occ[y0:y1, x0:x1] = 0

                occ_path = str(temp_dir_obj / f"occ_{gy}_{gx}.png")
                cv2.imwrite(occ_path, occ)

                img_occ = preprocess(occ_path)
                if img_occ is None:
                    drop = 0.0
                else:
                    feat_dict_occ = extract_features(occ_path, label=2, class_name="tb")
                    if feat_dict_occ is None:
                        drop = 0.0
                    else:
                        vec = np.array([feat_dict_occ[k] for k in self.feat_cols_3300], dtype=np.float32).reshape(1, -1)
                        vec_sel = self.selector.transform(vec)
                        prob_occ = self.clf.predict_proba(vec_sel)[0, 2]
                        drop = max(prob_base - prob_occ, 0.0)

                heatmap[gy][gx] = round(drop, 6)
                Path(occ_path).unlink(missing_ok=True)
                _send_ws(ws_manager.send_progress, session_id, gy * grid_n + gx + 1, grid_n * grid_n, f"Celda ({gy},{gx})")

        overlay_b64 = self._render_occlusion_overlay(img_256, heatmap)
        elapsed = time.time() - t0
        logger.info("Occlusion COMPLETE: session=%s time=%.1fs", session_id, elapsed)
        return heatmap, overlay_b64, elapsed

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