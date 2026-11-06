"""Compara estrategias de desbalanceo con LR sobre textura pulmonar.

Uso:
    python scripts/run_imbalance_v2.py

Salida:
    data/v2/imbalance_summary.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.modeling import columns_for_set  # noqa: E402

DATA = ROOT / "data"
OUT = DATA / "v2"
OUT.mkdir(parents=True, exist_ok=True)


def make_pipeline(strategy: str):
    steps = [
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=48)),
    ]
    if strategy == "smote":
        steps.append(("sampler", SMOTE(random_state=42, k_neighbors=5)))
    elif strategy == "under":
        steps.append(("sampler", RandomUnderSampler(random_state=42)))
    elif strategy == "over":
        steps.append(("sampler", RandomOverSampler(random_state=42)))
    cw = "balanced" if strategy == "balanced" else None
    steps.append(("clf", LogisticRegression(max_iter=2000, C=1.0,
                                            class_weight=cw, random_state=42)))
    return ImbPipeline(steps)


def main() -> None:
    df = pd.read_csv(DATA / "features_tbx11k_v2.csv")
    cols = columns_for_set(df, "texture")
    X = df[cols].to_numpy(dtype=np.float32)
    y = (df["class_name"] == "tb").astype(int).to_numpy()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "balanced_accuracy": "balanced_accuracy",
        "recall_tb": "recall",
        "precision_tb": "precision",
        "ap": "average_precision",
        "roc_auc": "roc_auc",
    }
    rows = []
    for strategy in ["none", "balanced", "smote", "under", "over"]:
        print(">>>", strategy)
        res = cross_validate(make_pipeline(strategy), X, y, cv=cv,
                             scoring=scoring, n_jobs=-1)
        row = {"strategy": strategy}
        for k in scoring:
            row[f"{k}_mean"] = float(np.mean(res[f"test_{k}"]))
            row[f"{k}_std"] = float(np.std(res[f"test_{k}"]))
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("balanced_accuracy_mean", ascending=False)
    out.to_csv(OUT / "imbalance_summary.csv", index=False)
    print(out.to_string(index=False))
    (OUT / "imbalance_summary.json").write_text(
        json.dumps(out.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
