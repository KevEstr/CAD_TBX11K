"""Evalúa el modelo final en un dataset externo.

Uso:
    python scripts/evaluate_external.py --manifest ruta/manifest.csv

El manifest debe tener ``image_path`` y, si se quieren métricas, ``label`` con
0=health, 1=sick, 2=tb. Las imágenes externas pasan por la misma segmentación
PSPNet, normalización y extracción de features que el entrenamiento.

Salida:
    data/external_eval.csv
    data/external_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.predict import cargar_contexto, predecir_imagen  # noqa: E402

DATA = ROOT / "data"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", default=str(DATA / "external_eval.csv"))
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    if "image_path" not in manifest.columns:
        raise ValueError("El manifest debe tener una columna 'image_path'.")

    modelo, nombre, feat_cols, class_names, umbral = cargar_contexto(args.model)
    filas = []
    for _, row in manifest.iterrows():
        try:
            resultado = predecir_imagen(
                str(row["image_path"]), modelo, feat_cols, class_names, umbral
            )
        except Exception as exc:  # noqa: BLE001
            resultado = {
                "image_path": str(row["image_path"]),
                "error": str(exc),
            }
        if "label" in manifest.columns:
            resultado["label"] = int(row["label"])
        filas.append(resultado)

    df = pd.DataFrame(filas)
    df.insert(0, "modelo", nombre)
    df.to_csv(args.out, index=False)

    resumen = {"modelo": nombre, "n_imagenes": int(len(df))}
    validos = df[df.get("error").isna()] if "error" in df.columns else df
    if "label" in validos.columns and len(validos):
        from sklearn.metrics import (
            balanced_accuracy_score, confusion_matrix, recall_score,
        )

        y = validos["label"].to_numpy()
        pred = validos["clase_indice"].to_numpy()
        resumen.update({
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "recall_por_clase": {
                class_names[i]: float(v) for i, v in enumerate(
                    recall_score(y, pred, labels=list(range(len(class_names))),
                                 average=None, zero_division=0)
                )
            },
            "confusion_matrix": confusion_matrix(y, pred).tolist(),
        })
        if "prob_tb" in validos.columns:
            from sklearn.metrics import average_precision_score, roc_auc_score

            y_tb = (y == 2).astype(int)
            resumen["auc_tb"] = float(roc_auc_score(y_tb, validos["prob_tb"]))
            resumen["ap_tb"] = float(
                average_precision_score(y_tb, validos["prob_tb"])
            )

    (DATA / "external_eval.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(resumen, ensure_ascii=False, indent=2))
    print(f"Predicciones guardadas en {args.out}")


if __name__ == "__main__":
    main()
