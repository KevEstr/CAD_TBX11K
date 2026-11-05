"""Predice una imagen o un lote con un modelo final del notebook 08.

Uso:
    python scripts/predict.py --image ruta/a/radiografia.png
    python scripts/predict.py --image ruta.png --model "XGBoost"
    python scripts/predict.py --csv data/imagenes.csv --out data/predicciones.csv

El CSV de lote debe tener una columna ``image_path``. Si además tiene
``label`` (0=health, 1=sick, 2=tb), se calculan métricas globales.

El modelo se elige del manifiesto ``models/08_manifest.json``. Si no se indica
``--model``, se usa el seleccionado por validación cruzada.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import extract_features, model_feature_columns  # noqa: E402
from src.preprocessing import preprocess, preprocess_bundle, segment_image  # noqa: E402

DATA = ROOT / "data"
MODELS = ROOT / "models"
CLASS_NAMES = ["health", "sick", "tb"]


def cargar_contexto(modelo_nombre: str | None):
    manifest_path = MODELS / "08_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Falta models/08_manifest.json. Ejecuta primero el notebook 08."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_path = MODELS / "08_feature_schema.json"
    schema = (
        json.loads(schema_path.read_text(encoding="utf-8"))
        if schema_path.exists()
        else None
    )
    class_names = schema["class_names"] if schema else CLASS_NAMES
    nombre = modelo_nombre or manifest["seleccionado_cv"]
    if nombre not in manifest["modelos"]:
        raise ValueError(
            f"Modelo desconocido: {nombre}. Disponibles: {sorted(manifest['modelos'])}"
        )
    feat_cols = (
        list(schema["feature_columns"])
        if schema
        else model_feature_columns(pd.read_csv(DATA / "features_tbx11k.csv", nrows=0))
    )
    modelo = joblib.load(MODELS / manifest["modelos"][nombre])
    n_modelo = getattr(modelo, "n_features_in_", None)
    if n_modelo is not None and n_modelo != len(feat_cols):
        raise ValueError(
            f"El modelo espera {n_modelo} features y el esquema tiene "
            f"{len(feat_cols)}. Reentrena 05–08 o revisa el esquema."
        )
    umbral_path = MODELS / "08_umbral_tb.json"
    umbral = (
        json.loads(umbral_path.read_text(encoding="utf-8"))
        if umbral_path.exists()
        else None
    )
    if umbral and umbral.get("modelo") != nombre:
        umbral = None
    return modelo, nombre, feat_cols, class_names, umbral


def predecir_imagen(
    image_path: str,
    modelo,
    feat_cols: list[str],
    class_names: list[str],
    umbral: dict | None = None,
) -> dict:
    fila = extract_features(image_path, 0, "health")
    if fila is None:
        raise ValueError(f"No se pudo procesar la imagen: {image_path}")
    faltantes = [c for c in feat_cols if c not in fila]
    if faltantes:
        raise ValueError(f"Faltan features en la extracción: {faltantes[:10]}")
    X = np.asarray([[fila[c] for c in feat_cols]], dtype=np.float32)

    pred = int(modelo.predict(X)[0])
    resultado = {
        "image_path": image_path,
        "prediccion": class_names[pred],
        "clase_indice": pred,
        "mask_area_fraction": float(fila["mask_area_fraction"]),
        "mask_lr_balance": float(fila["mask_lr_balance"]),
        "mask_contrast": float(fila["mask_contrast"]),
        "mask_fallback": int(fila["mask_fallback"]),
    }
    if fila["mask_fallback"]:
        resultado["advertencia"] = (
            "La segmentación usó el respaldo torácico (fallback); "
            "la predicción es menos fiable."
        )
    if hasattr(modelo, "predict_proba"):
        proba = np.asarray(modelo.predict_proba(X))[0]
        for i, nombre in enumerate(class_names):
            resultado[f"prob_{nombre}"] = float(proba[i])
        if umbral and "tb" in class_names:
            p_tb = float(proba[class_names.index("tb")])
            resultado["tb_probabilidad"] = p_tb
            resultado["tb_positivo_youden"] = bool(
                p_tb >= umbral["threshold_youden"]
            )
            resultado["tb_positivo_sens90"] = bool(
                p_tb >= umbral["threshold_sens90"]
            )
    elif hasattr(modelo, "decision_function"):
        scores = np.asarray(modelo.decision_function(X))[0]
        for i, nombre in enumerate(class_names):
            resultado[f"score_{nombre}"] = float(scores[i])
    return resultado


def _score_tb(modelo, X: np.ndarray, class_names: list[str]) -> float:
    idx = class_names.index("tb") if "tb" in class_names else len(class_names) - 1
    if hasattr(modelo, "predict_proba"):
        return float(np.asarray(modelo.predict_proba(X))[0][idx])
    scores = np.asarray(modelo.decision_function(X))[0]
    return float(scores[idx])


def generar_heatmap(
    image_path: str,
    modelo,
    feat_cols: list[str],
    class_names: list[str],
    out_path: str,
    grid: int = 8,
) -> None:
    """Mapa de oclusión: celdas que más reducen la probabilidad de TB."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    h, w = img.shape
    cell_h, cell_w = h // grid, w // grid
    fill = int(np.median(img))

    def _score(arr: np.ndarray) -> float:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            cv2.imwrite(tmp_path, arr)
            fila = extract_features(tmp_path, 0, "health")
            if fila is None:
                return 0.0
            X = np.asarray([[fila[c] for c in feat_cols]], dtype=np.float32)
            return _score_tb(modelo, X, class_names)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    base = _score(img)
    heat = np.zeros((grid, grid), dtype=np.float32)
    for gy in range(grid):
        for gx in range(grid):
            occluded = img.copy()
            y0, x0 = gy * cell_h, gx * cell_w
            occluded[y0:y0 + cell_h, x0:x0 + cell_w] = fill
            heat[gy, gx] = max(0.0, base - _score(occluded))

    if heat.max() > 0:
        heat = heat / heat.max()
    display = preprocess(image_path, modo="full")
    if display is None:
        return
    heat256 = cv2.resize(heat, (display.shape[1], display.shape[0]),
                         interpolation=cv2.INTER_CUBIC)
    heat_u8 = np.clip(heat256 * 255, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    base_bgr = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(base_bgr, 0.55, color, 0.45, 0)
    cv2.imwrite(out_path, overlay)
    print(f"Mapa de oclusión guardado en {out_path} (base TB={base:.4f})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="Ruta de una radiografía.")
    parser.add_argument("--csv", default=None, help="CSV con columna image_path.")
    parser.add_argument("--model", default=None, help="Nombre del modelo del manifiesto.")
    parser.add_argument("--out", default=None, help="CSV/JSON de salida.")
    parser.add_argument("--save-mask", default=None, help="PNG de la máscara 512×512.")
    parser.add_argument(
        "--save-preprocessed", default=None, help="PNG del recorte torácico 256×256."
    )
    parser.add_argument(
        "--heatmap", default=None, help="PNG del mapa de oclusión para la clase TB."
    )
    args = parser.parse_args()

    if bool(args.image) == bool(args.csv):
        raise ValueError("Indica exactamente uno: --image o --csv.")

    modelo, nombre_modelo, feat_cols, class_names, umbral = cargar_contexto(args.model)

    if args.image:
        image = Path(args.image)
        if not image.exists():
            raise FileNotFoundError(f"No existe la imagen: {image}")
        resultado = predecir_imagen(
            str(image), modelo, feat_cols, class_names, umbral
        )
        resultado["modelo"] = nombre_modelo

        if args.save_mask:
            mask512, _ = segment_image(str(image))
            if mask512 is not None:
                cv2.imwrite(args.save_mask, (mask512.astype(np.uint8) * 255))
                resultado["mascara_guardada"] = args.save_mask
        if args.save_preprocessed:
            bundle = preprocess_bundle(str(image))
            if bundle is not None:
                cv2.imwrite(args.save_preprocessed, bundle["thorax"][0])
                resultado["preprocesada_guardada"] = args.save_preprocessed
        if args.heatmap:
            generar_heatmap(str(image), modelo, feat_cols, class_names, args.heatmap)
            resultado["heatmap_guardado"] = args.heatmap

        print(json.dumps(resultado, ensure_ascii=False, indent=2))
        if args.out:
            Path(args.out).write_text(
                json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return

    datos = pd.read_csv(args.csv)
    if "image_path" not in datos.columns:
        raise ValueError("El CSV debe tener una columna 'image_path'.")
    filas = [
        predecir_imagen(str(p), modelo, feat_cols, class_names, umbral)
        for p in datos["image_path"]
    ]
    salida = pd.DataFrame(filas)
    salida.insert(0, "modelo", nombre_modelo)

    if "label" in datos.columns:
        from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score

        y_true = datos["label"].to_numpy()
        y_pred = salida["clase_indice"].to_numpy()
        print("Balanced accuracy:", round(balanced_accuracy_score(y_true, y_pred), 4))
        print("Recall por clase:",
              {class_names[i]: round(v, 4) for i, v in enumerate(
                  recall_score(y_true, y_pred, labels=list(range(len(class_names))),
                               average=None, zero_division=0)
              )})
        print("Matriz de confusión:")
        print(confusion_matrix(y_true, y_pred))

    if args.out:
        salida.to_csv(args.out, index=False)
        print(f"Predicciones guardadas en {args.out}")
    else:
        print(salida.to_string(index=False))


if __name__ == "__main__":
    main()
