"""Evalúa textura estricta y armonizada, con y sin residualización de contexto.

Requiere:
    data/features_tbx11k_v2.csv
    data/features_strict.csv

Salida:
    data/v2/strict_deconfound_summary.csv
    reports/v2/strict_deconfound.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.modeling import (  # noqa: E402
    ContextResidualizer,
    binary_metrics,
    threshold_for_sensitivity,
    youden_threshold,
)

DATA = ROOT / "data"
OUT = DATA / "v2"
REPORTS = ROOT / "reports" / "v2"
OUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


def texture_cols(df, prefix):
    cols = [
        c for c in df.columns
        if c.startswith(prefix)
        and not c.startswith(prefix + "shape_")
        and not c.startswith(prefix + "thumb_")
    ]
    return cols


def local_cols(df, prefix):
    return [
        c for c in texture_cols(df, prefix)
        if any(c.startswith(prefix + p) for p in ("glcm_", "lbp_", "gabor_"))
    ]


def make_model(model_name, n_features, k=48):
    clf = (
        LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=42)
        if model_name == "lr"
        else SVC(probability=True, C=1.0, class_weight="balanced", random_state=42)
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=min(k, n_features))),
        ("clf", clf),
    ])


def make_resid(model_name, n_texture, k=48):
    return Pipeline([
        ("resid", ContextResidualizer(n_texture=n_texture, alpha=10.0)),
        ("model", make_model(model_name, n_texture, k)),
    ])


def main() -> None:
    df = pd.read_csv(DATA / "features_tbx11k_v2.csv")
    strict = pd.read_csv(DATA / "features_strict.csv")
    strict = strict.drop(columns=[c for c in ("class_name", "label") if c in strict.columns])
    df = df.merge(strict, on="image_path", how="left")
    print("Merged:", df.shape, "nulos strict:",
          int(df[[c for c in df.columns if c.startswith(("strict_", "stricth_"))]].isnull().sum().sum()))

    y = (df["class_name"] == "tb").astype(int).to_numpy()
    cv = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(df, y))

    context = [c for c in df.columns if c.startswith("ctx_")]
    sets = {
        "texture_all_v2": (texture_cols(df, ""), None),  # reference handled below
        "strict_all": (texture_cols(df, "strict_"), None),
        "strict_local": (local_cols(df, "strict_"), None),
        "stricth_all": (texture_cols(df, "stricth_"), None),
        "stricth_local": (local_cols(df, "stricth_"), None),
        "strict_resid_context": (texture_cols(df, "strict_"), context),
        "stricth_resid_context": (texture_cols(df, "stricth_"), context),
    }
    # Referencia v2: columnas sin prefijo de las familias de textura.
    v2_texture = [
        c for c in df.columns
        if c in {"mean", "std", "contrast", "entropy"}
        or c.startswith(("glcm_", "lbp_", "gabor_"))
        or any(c.startswith(r + "_") for r in ["SI", "SD", "MI", "MD", "II", "ID"])
    ]
    sets["texture_all_v2"] = (v2_texture, None)

    rows = []
    for name, (cols, nuisance) in sets.items():
        cols = [c for c in cols if c in df.columns]
        X = df[cols + (nuisance or [])].to_numpy(dtype=np.float32)
        for model_name in ("lr", "svm"):
            print(f">>> {name} | {model_name} | X={X.shape}")
            oof = np.zeros(len(y))
            for tr, te in cv:
                model = (make_model(model_name, X.shape[1])
                         if nuisance is None
                         else make_resid(model_name, len(cols)))
                model.fit(X[tr], y[tr])
                oof[te] = model.predict_proba(X[te])[:, 1]
            tj = youden_threshold(y, oof)
            m = binary_metrics(y, oof, tj)
            m90 = binary_metrics(y, oof, threshold_for_sensitivity(y, oof, 0.90))
            rows.append({
                "set": name, "model": model_name, "n_features": len(cols),
                "balacc_youden": m["balanced_accuracy"],
                "sens_youden": m["sensitivity_tb"],
                "spec_youden": m["specificity"],
                "ap": average_precision_score(y, oof),
                "auc": roc_auc_score(y, oof),
                "sens90": m90["sensitivity_tb"],
                "spec90": m90["specificity"],
            })

    out = pd.DataFrame(rows).sort_values(["set", "model"])
    out.to_csv(OUT / "strict_deconfound_summary.csv", index=False)
    (OUT / "strict_deconfound_summary.json").write_text(
        json.dumps(out.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(out.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(11, 5))
    labels = out["set"] + " | " + out["model"]
    ax.barh(labels, out["balacc_youden"], color="#2166AC")
    ax.axvline(0.5, color="k", ls="--", lw=0.8)
    ax.set_xlabel("Balanced accuracy (Youden)")
    ax.set_title("Textura estricta/armonizada vs contexto")
    plt.tight_layout()
    plt.savefig(REPORTS / "strict_deconfound.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
