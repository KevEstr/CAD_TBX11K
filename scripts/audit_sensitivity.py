"""Sensibilidad del modelo ya entrenado a intervenciones de adquisición.

Se entrena un clon del modelo seleccionado con el 80 % de las imágenes y se
evalúa en el 20 % restante. No se reentrena con las imágenes modificadas.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_views import _grupos  # noqa: E402
from src.features import extract_features, model_feature_columns  # noqa: E402
from src.preprocessing import segment_image  # noqa: E402

DATA = ROOT / "data"
MODELS = ROOT / "models"


def _variantes(img: np.ndarray, mask: np.ndarray) -> dict:
    out = {}
    med = int(np.median(img))
    out["fondo_neutralizado"] = np.where(mask, img, med).astype(np.uint8)
    out["pulmones_neutralizados"] = np.where(mask, med, img).astype(np.uint8)
    mean = float(img.mean())
    out["contraste_bajo"] = np.clip(
        mean + (img.astype(np.float32) - mean) * 0.85, 0, 255
    ).astype(np.uint8)
    out["contraste_alto"] = np.clip(
        mean + (img.astype(np.float32) - mean) * 1.15, 0, 255
    ).astype(np.uint8)
    out["gamma_bajo"] = np.clip(
        (img.astype(np.float32) / 255.0) ** 0.9 * 255.0, 0, 255
    ).astype(np.uint8)
    out["gamma_alto"] = np.clip(
        (img.astype(np.float32) / 255.0) ** 1.1 * 255.0, 0, 255
    ).astype(np.uint8)
    out["blur"] = cv2.GaussianBlur(img, (3, 3), 0.6)
    med = int(np.median(img))
    oc_sup = img.copy(); oc_sup[:128] = med
    oc_inf = img.copy(); oc_inf[384:] = med
    out["oclusion_superior"] = oc_sup
    out["oclusion_inferior"] = oc_inf
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=100)
    args = parser.parse_args()

    manifest = json.loads((MODELS / "08_manifest.json").read_text(encoding="utf-8"))
    schema = json.loads((MODELS / "08_feature_schema.json").read_text(encoding="utf-8"))
    nombre = manifest["seleccionado_cv"]
    modelo = joblib.load(MODELS / manifest["modelos"][nombre])
    feat_cols = list(schema["feature_columns"])
    clases = list(schema["class_names"])
    idx_tb = clases.index("tb")

    X = np.load(DATA / "X_full.npy")
    y = np.load(DATA / "y_full.npy")
    meta = pd.read_csv(DATA / "full_meta.csv")
    grupos = _grupos(meta["image_path"].tolist())

    skf = StratifiedGroupKFold(5, shuffle=True, random_state=42)
    train_idx, hold_idx = next(iter(skf.split(X, y, groups=grupos)))
    modelo = clone(modelo).fit(X[train_idx], y[train_idx])

    def score_tb(vec):
        if hasattr(modelo, "predict_proba"):
            return float(modelo.predict_proba(vec)[0, idx_tb])
        scores = np.asarray(modelo.decision_function(vec))[0]
        return float(scores[idx_tb])

    rng = np.random.RandomState(42)
    sample = []
    for clase in np.unique(y[hold_idx]):
        sub = hold_idx[y[hold_idx] == clase]
        n = min(args.per_class, len(sub))
        sample.extend(rng.choice(sub, size=n, replace=False))
    sample = np.array(sample)

    filas = []
    for k, idx in enumerate(sample, start=1):
        path = meta.loc[idx, "image_path"]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        mask, _ = segment_image(path)
        if mask is None:
            continue
        base = score_tb(X[[idx]])
        for nombre_v, arr in _variantes(img, mask).items():
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
            cv2.imwrite(tmp_path, arr)
            try:
                f = extract_features(
                    tmp_path, int(y[idx]), meta.loc[idx, "class_name"],
                    include_context=False, include_shape=False,
                )
                if f is None:
                    continue
                vec = np.asarray([[f[c] for c in feat_cols]], dtype=np.float32)
                prob = score_tb(vec)
                filas.append({
                    "image_path": path, "label": int(y[idx]),
                    "variante": nombre_v, "base": base, "prob": prob,
                    "delta": prob - base,
                })
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        if k % 50 == 0:
            print(f"Sensibilidad {k}/{len(sample)}")

    df = pd.DataFrame(filas)
    df.to_csv(DATA / "sensibilidad_intervenciones.csv", index=False)

    resumen = []
    for variante, sub in df.groupby("variante"):
        vals = sub["delta"].to_numpy()
        paths = sub["image_path"].tolist()
        g = _grupos(paths)
        unicos = np.unique(g)
        por_grupo = [np.where(g == u)[0] for u in unicos]
        medias = []
        for _ in range(1000):
            ii = rng.choice(len(unicos), len(unicos), replace=True)
            idx = np.concatenate([por_grupo[i] for i in ii])
            medias.append(float(np.mean(vals[idx])))
        resumen.append({
            "variante": variante, "n": len(sub),
            "delta_medio": float(np.mean(vals)),
            "delta_mediano": float(np.median(vals)),
            "delta_low": float(np.percentile(medias, 2.5)),
            "delta_high": float(np.percentile(medias, 97.5)),
            "caida_media": float(np.mean(sub["prob"] < sub["base"])),
        })
    pd.DataFrame(resumen).to_csv(DATA / "sensibilidad_resumen.csv", index=False)
    print(pd.DataFrame(resumen).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
