"""Corrección del atajo de contexto: textura local y residualización.

Compara, con las mismas particiones de 5 folds:

- textura completa (intensidad + GLCM + LBP + Gabor)
- textura local (GLCM + LBP + Gabor, sin intensidad global/regional)
- contexto solo
- forma solo
- textura residualizada contra contexto
- textura residualizada contra contexto + forma
- textura local residualizada contra contexto

La residualización se ajusta **dentro de cada fold** con
``ContextResidualizer``. Si la textura conserva rendimiento, su señal no es
explicable por la fuente/adquisición.

Salida:
    data/v2/deconfound_summary.csv
    data/v2/oof_deconfound__*.csv
    reports/v2/deconfound.png
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
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import get_feature_groups  # noqa: E402
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


def build_sets(df: pd.DataFrame):
    groups = get_feature_groups(df)
    texture = groups["texture"]
    local = [c for c in texture if c.startswith(("glcm_", "lbp_", "gabor_"))]
    context = groups["context"]
    shape = groups["shape"]

    return {
        "texture_all": (texture, None),
        "texture_local": (local, None),
        "context": (context, None),
        "shape": (shape, None),
        "texture_resid_context": (texture, context),
        "texture_resid_context_shape": (texture, context + shape),
        "texture_local_resid_context": (local, context),
    }


def make_model(model_name: str, n_features: int, k: int):
    clf = (
        LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                           random_state=42)
        if model_name == "lr"
        else SVC(probability=True, C=1.0, class_weight="balanced", random_state=42)
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=min(k, n_features))),
        ("clf", clf),
    ])


def make_residual_model(model_name: str, n_texture: int, n_total: int, k: int):
    base = make_model(model_name, n_texture, k)
    return Pipeline([
        ("resid", ContextResidualizer(n_texture=n_texture, alpha=10.0)),
        ("model", base),
    ])


def main() -> None:
    df = pd.read_csv(DATA / "features_tbx11k_v2.csv")
    y = (df["class_name"] == "tb").astype(int).to_numpy()
    cv = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(df, y))
    sets = build_sets(df)

    rows, oof_store = [], {}
    for name, (cols, nuisance) in sets.items():
        X = df[cols + (nuisance or [])].to_numpy(dtype=np.float32)
        n_texture = len(cols)
        for model_name in ("lr", "svm"):
            print(f">>> {name} | {model_name}")
            oof = np.zeros(len(y), dtype=float)
            for tr, te in cv:
                if nuisance is None:
                    model = make_model(model_name, X.shape[1], 48)
                else:
                    model = make_residual_model(model_name, n_texture, X.shape[1], 48)
                model.fit(X[tr], y[tr])
                oof[te] = model.predict_proba(X[te])[:, 1]

            tj = youden_threshold(y, oof)
            t90 = threshold_for_sensitivity(y, oof, 0.90)
            mj = binary_metrics(y, oof, tj)
            m90 = binary_metrics(y, oof, t90)
            rows.append({
                "set": name,
                "model": model_name,
                "balacc_youden": mj["balanced_accuracy"],
                "sens_youden": mj["sensitivity_tb"],
                "spec_youden": mj["specificity"],
                "ap": average_precision_score(y, oof),
                "auc": roc_auc_score(y, oof),
                "balacc_0.5": balanced_accuracy_score(y, (oof >= 0.5).astype(int)),
                "sens90": m90["sensitivity_tb"],
                "spec90": m90["specificity"],
                "balacc_sens90": m90["balanced_accuracy"],
            })
            oof_store[f"{name}__{model_name}"] = oof
            pd.DataFrame({"y_true": y, "proba": oof}).to_csv(
                OUT / f"oof_deconfound__{name}__{model_name}.csv", index=False
            )

    out = pd.DataFrame(rows).sort_values(["set", "model"])
    out.to_csv(OUT / "deconfound_summary.csv", index=False)
    (OUT / "deconfound_summary.json").write_text(
        json.dumps(out.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(out.round(3).to_string(index=False))

    # Figura: comparación directa.
    fig, ax = plt.subplots(figsize=(11, 5))
    labels = out["set"] + " | " + out["model"]
    ax.barh(labels, out["balacc_youden"], color="#2166AC")
    ax.axvline(0.5, color="k", ls="--", lw=0.8)
    ax.set_xlabel("Balanced accuracy (umbral Youden)")
    ax.set_title("Textura vs contexto: antes y después de residualizar")
    plt.tight_layout()
    plt.savefig(REPORTS / "deconfound.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
