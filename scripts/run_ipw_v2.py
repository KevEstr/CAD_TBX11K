"""Corrección del sesgo de contexto por ponderación por propensión inversa (IPW).

Idea: se estima, dentro de cada fold de entrenamiento, la probabilidad de ser
TB a partir del contexto/fuente. Esa probabilidad se usa para ponderar las
muestras de forma que las distribuciones de contexto de TB y no-TB queden
equilibradas. El modelo de textura se entrena con esos pesos y se evalúa en el
fold externo sin ponderar.

Si el rendimiento se mantiene, la textura aporta señal más allá del contexto.
Si cae, el atajo era el contexto.

Salida:
    data/v2/ipw_summary.csv
    reports/v2/ipw.png
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

from src.modeling import binary_metrics, threshold_for_sensitivity, youden_threshold  # noqa: E402

DATA = ROOT / "data"
OUT = DATA / "v2"
REPORTS = ROOT / "reports" / "v2"
OUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


def make_texture_model(model_name, n_features, k=48):
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


def fit_propensity(Xc_tr, y_tr):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=1.0,
                                   class_weight="balanced", random_state=42)),
    ])
    model.fit(Xc_tr, y_tr)
    return model


def main() -> None:
    df = pd.read_csv(DATA / "features_tbx11k_v2.csv")
    strict = pd.read_csv(DATA / "features_strict.csv")
    strict = strict.drop(columns=[c for c in ("class_name", "label") if c in strict.columns])
    df = df.merge(strict, on="image_path", how="left")

    y = (df["class_name"] == "tb").astype(int).to_numpy()
    context = [c for c in df.columns if c.startswith("ctx_")]
    Xc = df[context].to_numpy(dtype=np.float32)

    strict_all = [c for c in df.columns
                  if c.startswith("strict_") and not c.startswith(("strict_shape_", "strict_thumb_"))]
    v2_texture = [
        c for c in df.columns
        if c in {"mean", "std", "contrast", "entropy"}
        or c.startswith(("glcm_", "lbp_", "gabor_"))
        or any(c.startswith(r + "_") for r in ["SI", "SD", "MI", "MD", "II", "ID"])
    ]

    cv = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(df, y))
    rows, oof_store = [], {}
    for set_name, cols in [("texture_v2", v2_texture), ("texture_strict", strict_all)]:
        X = df[cols].to_numpy(dtype=np.float32)
        for model_name in ("lr", "svm"):
            for weighting in ("none", "ipw"):
                print(f">>> {set_name} | {model_name} | {weighting}")
                oof = np.zeros(len(y))
                for tr, te in cv:
                    model = make_texture_model(model_name, X.shape[1])
                    if weighting == "none":
                        model.fit(X[tr], y[tr])
                    else:
                        prop = fit_propensity(Xc[tr], y[tr])
                        e = np.clip(prop.predict_proba(Xc[tr])[:, 1], 0.05, 0.95)
                        w = y[tr] / e + (1 - y[tr]) / (1 - e)
                        w = np.clip(w, 0.1, 10.0)
                        w = w / w.mean()
                        model.fit(X[tr], y[tr], clf__sample_weight=w)
                    oof[te] = model.predict_proba(X[te])[:, 1]

                tj = youden_threshold(y, oof)
                m = binary_metrics(y, oof, tj)
                m90 = binary_metrics(y, oof, threshold_for_sensitivity(y, oof, 0.90))
                rows.append({
                    "set": set_name, "model": model_name, "weighting": weighting,
                    "balacc_youden": m["balanced_accuracy"],
                    "sens_youden": m["sensitivity_tb"],
                    "spec_youden": m["specificity"],
                    "ap": average_precision_score(y, oof),
                    "auc": roc_auc_score(y, oof),
                    "sens90": m90["sensitivity_tb"],
                    "spec90": m90["specificity"],
                })
                oof_store[f"{set_name}__{model_name}__{weighting}"] = oof
                pd.DataFrame({"y_true": y, "proba": oof}).to_csv(
                    OUT / f"oof_ipw__{set_name}__{model_name}__{weighting}.csv", index=False
                )

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "ipw_summary.csv", index=False)
    (OUT / "ipw_summary.json").write_text(
        json.dumps(out.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(out.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 4))
    labels = out["set"] + " | " + out["model"] + " | " + out["weighting"]
    ax.barh(labels, out["balacc_youden"], color="#2166AC")
    ax.axvline(0.5, color="k", ls="--", lw=0.8)
    ax.set_xlabel("Balanced accuracy (Youden)")
    ax.set_title("Efecto de IPW sobre el sesgo de contexto")
    plt.tight_layout()
    plt.savefig(REPORTS / "ipw.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
