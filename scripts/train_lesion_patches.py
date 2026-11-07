"""Entrena y evalúa el especialista de parches de lesión.

Controles:
- positivos alrededor de cajas TB,
- negativos A de la misma imagen, sin solape y en la misma zona,
- negativos B de imágenes health/sick,
- parches de fondo como control de fuente.

Evaluación:
- parches con folds agrupados por imagen,
- ventana deslizante en imágenes completas, mismo procedimiento para todas,
- comparación con el modelo de imagen y con el fondo.

Uso:
    python scripts/train_lesion_patches.py
    python scripts/train_lesion_patches.py --limit 20 --per-class-eval 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.lesion_patches import (  # noqa: E402
    PATCH_SIZE, PATCH_STRIDE, background_patches, patch_features,
    patch_from_box, random_patch_zone, sliding_patches, zone_of,
)
from src.preprocessing import segment_image  # noqa: E402

DATA = ROOT / "data"
MODELS = ROOT / "models"
COCO = DATA / "TBX11K" / "annotations" / "json" / "TBX11K_trainval.json"
CLASSES = ["health", "sick", "tb"]


def _metricas(y, score, pred=None) -> dict:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    if pred is None:
        pred = (score >= 0.5).astype(int)
    if len(np.unique(y)) < 2:
        return {
            "n": int(len(y)), "AUC": np.nan, "AP": np.nan,
            "Balanced Acc": np.nan, "Recall": np.nan, "Specificity": np.nan,
        }
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return {
        "n": int(len(y)),
        "AUC": float(roc_auc_score(y, score)),
        "AP": float(average_precision_score(y, score)),
        "Balanced Acc": float(balanced_accuracy_score(y, pred)),
        "Recall": tp / max(tp + fn, 1),
        "Specificity": tn / max(tn + fp, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--per-class-eval", type=int, default=40)
    args = parser.parse_args()

    meta = pd.read_csv(DATA / "full_meta.csv")
    if args.limit:
        meta = meta.groupby("class_name").head(args.limit).reset_index(drop=True)
    rng = np.random.RandomState(42)

    coco = json.loads(COCO.read_text(encoding="utf-8"))
    id_to_name = {im["id"]: Path(im["file_name"]).name for im in coco["images"]}
    basename_to_path = {Path(p).name: p for p in meta["image_path"]}
    boxes: dict[str, list[tuple[float, float, float, float]]] = {}
    for ann in coco["annotations"]:
        name = id_to_name.get(ann["image_id"])
        path = basename_to_path.get(name) if name else None
        if path is None:
            continue
        boxes.setdefault(path, []).append(tuple(float(v) for v in ann["bbox"]))

    all_zones = []
    for path, cajas in boxes.items():
        for bbox in cajas:
            all_zones.append(zone_of(bbox[1] + bbox[3] / 2, 512))

    filas = []
    for _, row in meta.iterrows():
        path = row["image_path"]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        mask, _ = segment_image(path)
        if mask is None:
            continue

        cajas = boxes.get(path, [])
        if row["class_name"] == "tb" and cajas:
            zonas_img = []
            for bbox in cajas:
                zona = zone_of(bbox[1] + bbox[3] / 2, 512)
                zonas_img.append(zona)
                for _ in range(2):
                    patch = patch_from_box(img, mask, bbox, rng=rng)
                    if patch is not None:
                        filas.append({
                            "image_path": path, "image_label": int(row["label"]),
                            "patch_label": 1, "patch_type": "positive",
                            "zone": int(zona), **patch_features(patch),
                        })
            for _ in range(4):
                zona = int(rng.choice(zonas_img)) if zonas_img else int(rng.randint(3))
                patch = random_patch_zone(img, mask, zona, cajas, rng=rng)
                if patch is not None:
                    filas.append({
                        "image_path": path, "image_label": int(row["label"]),
                        "patch_label": 0, "patch_type": "negA",
                        "zone": zona, **patch_features(patch),
                    })
        elif row["class_name"] in ("health", "sick") and rng.rand() < 0.27:
            for _ in range(2):
                zona = int(rng.choice(all_zones)) if all_zones else int(rng.randint(3))
                patch = random_patch_zone(img, mask, zona, [], rng=rng)
                if patch is not None:
                    filas.append({
                        "image_path": path, "image_label": int(row["label"]),
                        "patch_label": 0, "patch_type": "negB",
                        "zone": zona, **patch_features(patch),
                    })

        for patch in background_patches(img)[:1]:
            filas.append({
                "image_path": path, "image_label": int(row["label"]),
                "patch_label": 0, "patch_type": "background",
                "zone": -1, **patch_features(patch),
            })

    df = pd.DataFrame(filas).fillna(0)
    df.to_csv(DATA / "lesion_patches.csv", index=False)
    feature_cols = [
        c for c in df.columns
        if c not in ("image_path", "image_label", "patch_label", "patch_type", "zone")
    ]

    img_meta = meta[["image_path", "label", "class_name"]].copy()
    img_meta["fold"] = 0
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (_, va) in enumerate(skf.split(img_meta, img_meta["label"]), start=1):
        img_meta.loc[va, "fold"] = fold

    eval_idx = []
    for fold in range(1, 6):
        val = img_meta[img_meta["fold"] == fold]
        for clase in CLASSES:
            sub = val[val["class_name"] == clase]
            n = min(args.per_class_eval, len(sub))
            if n:
                eval_idx.extend(rng.choice(sub.index.to_numpy(), size=n, replace=False))
    eval_idx = set(int(i) for i in eval_idx)

    oof_lesion = np.full(len(df), np.nan)
    oof_bg = np.full(len(df), np.nan)
    image_rows = []

    for fold in range(1, 6):
        tr_paths = set(img_meta.loc[img_meta["fold"] != fold, "image_path"])
        va_paths = set(img_meta.loc[img_meta["fold"] == fold, "image_path"])

        train = df[df["image_path"].isin(tr_paths)]
        lesion = train[train["patch_type"].isin(["positive", "negA", "negB"])]
        bg = train[train["patch_type"] == "background"]
        if lesion["patch_label"].nunique() < 2:
            continue

        lesion_model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=42
            )),
        ])
        lesion_model.fit(
            lesion[feature_cols].to_numpy(dtype=np.float32),
            lesion["patch_label"].to_numpy(),
        )

        bg_model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=42
            )),
        ])
        bg_y = (bg["image_label"].to_numpy() == 2).astype(int)
        bg_model.fit(bg[feature_cols].to_numpy(dtype=np.float32), bg_y)

        val_lesion = df[
            df["image_path"].isin(va_paths)
            & df["patch_type"].isin(["positive", "negA", "negB"])
        ]
        if len(val_lesion):
            oof_lesion[val_lesion.index] = lesion_model.predict_proba(
                val_lesion[feature_cols].to_numpy(dtype=np.float32)
            )[:, 1]

        val_bg = df[
            df["image_path"].isin(va_paths) & (df["patch_type"] == "background")
        ]
        if len(val_bg):
            oof_bg[val_bg.index] = bg_model.predict_proba(
                val_bg[feature_cols].to_numpy(dtype=np.float32)
            )[:, 1]

        for idx in sorted(eval_idx):
            path = meta.loc[idx, "image_path"]
            if path not in va_paths:
                continue
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
            mask, _ = segment_image(path)
            if mask is None:
                continue

            coords, probs = [], []
            for x0, y0, patch in sliding_patches(
                img, mask, size=PATCH_SIZE, stride=PATCH_STRIDE
            ):
                f = patch_features(patch)
                probs.append(
                    lesion_model.predict_proba(
                        np.asarray([[f[c] for c in feature_cols]], dtype=np.float32)
                    )[0, 1]
                )
                coords.append((x0, y0))
            if not probs:
                continue
            probs = np.asarray(probs)
            order = np.argsort(probs)[::-1]
            top = order[:5]

            bg_scores = []
            for patch in background_patches(img):
                f = patch_features(patch)
                bg_scores.append(
                    bg_model.predict_proba(
                        np.asarray([[f[c] for c in feature_cols]], dtype=np.float32)
                    )[0, 1]
                )

            image_rows.append({
                "image_path": path,
                "label": int(meta.loc[idx, "label"]),
                "class_name": meta.loc[idx, "class_name"],
                "score_lesion": float(probs.max()),
                "score_lesion_top5": float(probs[top].mean()),
                "score_bg": float(np.max(bg_scores)) if bg_scores else np.nan,
                "top_coords": [coords[i] for i in top],
            })

    eval_df = pd.DataFrame(image_rows)

    filas_metricas = []
    lesion_df = df[df["patch_type"].isin(["positive", "negA", "negB"])].copy()
    lesion_df["score"] = oof_lesion[lesion_df.index]
    lesion_df = lesion_df[lesion_df["score"].notna()]
    for nombre, tipos in [
        ("parche_todos", ["positive", "negA", "negB"]),
        ("parche_pos_vs_A", ["positive", "negA"]),
        ("parche_pos_vs_B", ["positive", "negB"]),
    ]:
        sub = lesion_df[lesion_df["patch_type"].isin(tipos)]
        if len(sub):
            filas_metricas.append({
                "nivel": "parche", "conjunto": nombre,
                **_metricas(sub["patch_label"], sub["score"]),
            })

    bg_df = df[df["patch_type"] == "background"].copy()
    bg_df["score"] = oof_bg[bg_df.index]
    bg_df = bg_df[bg_df["score"].notna()]
    if len(bg_df):
        filas_metricas.append({
            "nivel": "fondo", "conjunto": "parche_fondo_TB",
            **_metricas((bg_df["image_label"] == 2).astype(int), bg_df["score"]),
        })

    if len(eval_df):
        tb = (eval_df["label"] == 2).astype(int)
        filas_metricas.append({
            "nivel": "imagen", "conjunto": "imagen_TB_vs_resto",
            **_metricas(tb, eval_df["score_lesion"]),
        })
        sub = eval_df[eval_df["label"].isin([1, 2])]
        if len(sub):
            filas_metricas.append({
                "nivel": "imagen", "conjunto": "imagen_TB_vs_sick",
                **_metricas((sub["label"] == 2).astype(int), sub["score_lesion"]),
            })
        if eval_df["score_bg"].notna().any():
            filas_metricas.append({
                "nivel": "imagen", "conjunto": "imagen_fondo_TB_vs_resto",
                **_metricas(tb, eval_df["score_bg"]),
            })
            if len(sub):
                filas_metricas.append({
                    "nivel": "imagen", "conjunto": "imagen_fondo_TB_vs_sick",
                    **_metricas((sub["label"] == 2).astype(int), sub["score_bg"]),
                })

    # Comparación con el modelo de imagen del notebook 08.
    oof_path = DATA / "08_oof_scores.npz"
    manifest_path = MODELS / "08_manifest.json"
    if len(eval_df) and oof_path.exists() and manifest_path.exists():
        selected = json.loads(manifest_path.read_text(encoding="utf-8"))["seleccionado_cv"]
        scores = np.load(oof_path)[selected]
        idx_eval = [int(meta.index[meta["image_path"] == p][0])
                    for p in eval_df["image_path"]]
        eval_df["score_lr"] = scores[idx_eval, 2]
        filas_metricas.append({
            "nivel": "imagen", "conjunto": f"imagen_{selected}_TB_vs_resto",
            **_metricas((eval_df["label"] == 2).astype(int), eval_df["score_lr"]),
        })
        sub = eval_df[eval_df["label"].isin([1, 2])]
        if len(sub):
            filas_metricas.append({
                "nivel": "imagen", "conjunto": f"imagen_{selected}_TB_vs_sick",
                **_metricas((sub["label"] == 2).astype(int), sub["score_lr"]),
            })

    metricas = pd.DataFrame(filas_metricas)
    metricas.to_csv(DATA / "lesion_patch_metrics.csv", index=False)
    eval_df.drop(columns=["top_coords"]).to_csv(
        DATA / "lesion_full_predictions.csv", index=False
    )

    # Localización: top-1 y top-5 contra cajas; nulo con cajas de otra imagen.
    loc_rows = []
    tb_paths = [p for p in eval_df["image_path"] if p in boxes]
    if tb_paths:
        hit1 = hit5 = 0
        null_hits = []
        for _, row in eval_df[eval_df["image_path"].isin(tb_paths)].iterrows():
            cajas = boxes[row["image_path"]]
            centros = [(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2)
                       for x, y in row["top_coords"]]
            hit5 += int(any(
                any(bx <= cx <= bx + bw and by <= cy <= by + bh
                    for bx, by, bw, bh in cajas)
                for cx, cy in centros
            ))
            hit1 += int(any(
                bx <= centros[0][0] <= bx + bw and by <= centros[0][1] <= by + bh
                for bx, by, bw, bh in cajas
            ))
            otra = tb_paths[rng.randint(len(tb_paths))]
            cajas_null = boxes[otra]
            null_hits.append(int(any(
                any(bx <= cx <= bx + bw and by <= cy <= by + bh
                    for bx, by, bw, bh in cajas_null)
                for cx, cy in centros
            )))
        n_tb = len(eval_df[eval_df["image_path"].isin(tb_paths)])
        loc_rows.append({
            "n_imagenes_tb": n_tb,
            "hit_top1": hit1 / max(n_tb, 1),
            "hit_top5": hit5 / max(n_tb, 1),
            "hit_top5_nulo": float(np.mean(null_hits)) if null_hits else np.nan,
        })
        pd.DataFrame(loc_rows).to_csv(DATA / "lesion_patch_localization.csv", index=False)

    lesion_all = df[df["patch_type"].isin(["positive", "negA", "negB"])]
    final = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        )),
    ])
    final.fit(
        lesion_all[feature_cols].to_numpy(dtype=np.float32),
        lesion_all["patch_label"].to_numpy(),
    )
    joblib.dump(final, MODELS / "lesion_patch.joblib")

    bg_all = df[df["patch_type"] == "background"]
    bg_final = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        )),
    ])
    bg_final.fit(
        bg_all[feature_cols].to_numpy(dtype=np.float32),
        (bg_all["image_label"] == 2).astype(int),
    )
    joblib.dump(bg_final, MODELS / "background_patch.joblib")

    manifest = {
        "feature_columns": feature_cols,
        "patch_size": PATCH_SIZE,
        "patch_stride": PATCH_STRIDE,
        "n_patches": int(len(df)),
        "n_positive": int((df["patch_type"] == "positive").sum()),
        "n_negA": int((df["patch_type"] == "negA").sum()),
        "n_negB": int((df["patch_type"] == "negB").sum()),
        "n_background": int((df["patch_type"] == "background").sum()),
        "metrics": metricas.round(6).to_dict(orient="records"),
    }
    (MODELS / "lesion_patch_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(metricas.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
