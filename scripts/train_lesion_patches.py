"""Entrena un especialista de parches de lesión TB vs pulmón normal.

Uso:
    python scripts/train_lesion_patches.py

Salida:
    data/lesion_patches.csv
    data/lesion_patch_metrics.csv
    models/lesion_patch.joblib
    models/lesion_patch_manifest.json
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
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.lesion_patches import patch_features, patch_from_box, random_patch
from src.preprocessing import segment_image

DATA = ROOT / "data"
MODELS = ROOT / "models"
COCO = DATA / "TBX11K" / "annotations" / "json" / "TBX11K_trainval.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    meta = pd.read_csv(DATA / "full_meta.csv")
    if args.limit:
        meta = meta.groupby("class_name").head(args.limit).reset_index(drop=True)
    coco = json.loads(COCO.read_text(encoding="utf-8"))
    id_to_name = {im["id"]: Path(im["file_name"]).name for im in coco["images"]}
    basename_to_path = {Path(p).name: p for p in meta["image_path"]}
    boxes: dict[str, list[tuple[float, float, float, float]]] = {}
    for ann in coco["annotations"]:
        name = id_to_name.get(ann["image_id"])
        if name is None:
            continue
        path = basename_to_path.get(name)
        if path is None:
            continue
        boxes.setdefault(path, []).append(tuple(float(v) for v in ann["bbox"]))

    rng = np.random.RandomState(42)
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

        if row["class_name"] == "tb" and path in boxes:
            for bbox in boxes[path]:
                for _ in range(2):
                    patch = patch_from_box(img, mask, bbox, rng=rng)
                    if patch is not None:
                        filas.append({
                            "image_path": path,
                            "image_label": int(row["label"]),
                            "patch_label": 1,
                            **patch_features(patch),
                        })
            patch = random_patch(img, mask, rng=rng)
            if patch is not None:
                filas.append({
                    "image_path": path,
                    "image_label": int(row["label"]),
                    "patch_label": 0,
                    **patch_features(patch),
                })
        elif row["class_name"] in ("health", "sick"):
            if rng.rand() < 0.22:
                patch = random_patch(img, mask, rng=rng)
                if patch is not None:
                    filas.append({
                        "image_path": path,
                        "image_label": int(row["label"]),
                        "patch_label": 0,
                        **patch_features(patch),
                    })

    df = pd.DataFrame(filas).fillna(0)
    df.to_csv(DATA / "lesion_patches.csv", index=False)

    feature_cols = [
        c for c in df.columns
        if c not in ("image_path", "image_label", "patch_label")
    ]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["patch_label"].to_numpy()
    groups = df["image_path"].to_numpy()

    cv = GroupKFold(n_splits=5)
    oof = np.zeros(len(df), dtype=float)
    for tr, va in cv.split(X, y, groups=groups):
        modelo = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=42
            )),
        ])
        modelo.fit(X[tr], y[tr])
        oof[va] = modelo.predict_proba(X[va])[:, 1]

    patch_auc = float(roc_auc_score(y, oof))
    patch_ap = float(average_precision_score(y, oof))
    patch_bal = float(balanced_accuracy_score(y, (oof >= 0.5).astype(int)))

    img_df = df[["image_path", "image_label"]].copy()
    img_df["score"] = oof
    img_max = img_df.groupby("image_path").agg(
        image_label=("image_label", "first"),
        score=("score", "max"),
    ).reset_index()
    y_img = (img_max["image_label"] == 2).astype(int)
    image_auc = float(roc_auc_score(y_img, img_max["score"]))
    image_ap = float(average_precision_score(y_img, img_max["score"]))
    image_bal = float(
        balanced_accuracy_score(y_img, (img_max["score"] >= 0.5).astype(int))
    )

    metricas = pd.DataFrame([
        {"nivel": "parche", "n": len(df), "AUC": patch_auc,
         "AP": patch_ap, "Balanced Acc": patch_bal},
        {"nivel": "imagen", "n": len(img_max), "AUC": image_auc,
         "AP": image_ap, "Balanced Acc": image_bal},
    ])
    metricas.to_csv(DATA / "lesion_patch_metrics.csv", index=False)

    final = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        )),
    ])
    final.fit(X, y)
    joblib.dump(final, MODELS / "lesion_patch.joblib")

    manifest = {
        "modelo": "lesion_patch",
        "feature_columns": feature_cols,
        "patch_size": 96,
        "patch_stride": 32,
        "n_patches": int(len(df)),
        "n_images": int(len(img_max)),
        "metricas": metricas.round(6).to_dict(orient="records"),
    }
    (MODELS / "lesion_patch_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(metricas.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
