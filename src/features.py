"""
cad_tbx11k.features
====================
Extracción de características del pipeline CAD-TBX11K.

El diseño separa explícitamente **tejido pulmonar**, **forma de la máscara** y
**contexto/fuente**, para que el análisis pueda demostrar qué parte del
rendimiento se debe a cada uno:

+----------------------+--------------------------------------------------+
| Grupo                | Qué captura                                      |
+======================+==================================================+
| ``texture``          | intensidad, GLCM, LBP y Gabor calculados dentro  |
|                      | del campo pulmonar (96 features)                 |
+----------------------+--------------------------------------------------+
| ``hog``              | gradientes orientados sobre el recorte torácico  |
|                      | (1764 features, mapeables a posición espacial)   |
+----------------------+--------------------------------------------------+
| ``shape``            | geometría de la máscara (10) + miniatura 8×8     |
|                      | de la máscara (64) — **solo control de sesgo**   |
+----------------------+--------------------------------------------------+
| ``context``          | intensidad/textura del fondo + miniatura 16×16   |
|                      | del fondo (352) — **control de fuente**          |
+----------------------+--------------------------------------------------+
| ``quality``          | métricas de la segmentación (área, balance,      |
|                      | contraste, fallback)                             |
+----------------------+--------------------------------------------------+

Las características de textura se calculan con **soporte de máscara**: la
intensidad, GLCM, LBP y Gabor solo usan píxeles pulmonares. El exterior se
rellena con la mediana pulmonar (no con 0) para no crear bordes artificiales.
HOG se calcula sobre el recorte torácico completo, sin máscara, para que no
codifique el contorno de la máscara.

El contraste de intensidad usa ``p95 - p5`` en lugar de ``max - min`` para no
depender de píxeles saturados por CLAHE (el antiguo ``ID_contrast`` ≈ 245 era
un artefacto de saturación).

Referencias
-----------
- Haralick et al. (1973). Textural features for image classification.
- Ojala et al. (2002). Multiresolution gray-scale and rotation invariant LBP.
- Dalal & Triggs (2005). Histograms of oriented gradients for human detection.
- Bosch et al. (2007). Representing shape with a spatial pyramid kernel.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from skimage.filters import gabor as skimage_gabor
from tqdm import tqdm

from .preprocessing import IMG_SIZE, preprocess_bundle

# ---------------------------------------------------------------------------
# Regiones anatómicas (6 zonas sobre la imagen de 256×256)
# ---------------------------------------------------------------------------

REGIONS: dict[str, tuple[int, int, int, int]] = {
    "SI": (0, 0, 128, 86),
    "SD": (128, 0, 128, 86),
    "MI": (0, 86, 128, 85),
    "MD": (128, 86, 128, 85),
    "II": (0, 171, 128, 85),
    "ID": (128, 171, 128, 85),
}

# ---------------------------------------------------------------------------
# Parámetros
# ---------------------------------------------------------------------------

GLCM_PROPS = ["contrast", "correlation", "energy", "homogeneity", "dissimilarity"]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

LBP_P = 24
LBP_R = 3.0
LBP_BINS = LBP_P + 2

HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (32, 32)
HOG_CELLS_PER_BLOCK = (2, 2)

GABOR_FREQUENCIES = [0.1, 0.2, 0.3, 0.4]
GABOR_THETAS = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

PHOG_ORIENTATIONS = 8
PHOG_CELLS_PER_BLOCK = (2, 2)
PHOG_L0_PPC = (64, 64)
PHOG_L1_PPC = (32, 32)

CONTEXT_THUMB = (16, 16)
SHAPE_THUMB = (8, 8)

META_COLS = {"image_path", "class_name", "label"}
QUALITY_COLS = {
    "mask_area_fraction",
    "mask_lr_balance",
    "mask_contrast",
    "mask_fallback",
}


def model_feature_columns(features_df: pd.DataFrame) -> list[str]:
    """Columnas usadas por el modelo (excluye metadatos y calidad de máscara)."""
    return [c for c in features_df.columns if c not in META_COLS | QUALITY_COLS]


# ---------------------------------------------------------------------------
# Utilidades de máscara
# ---------------------------------------------------------------------------

def _safe_mask(mask: Optional[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    return mask.astype(bool)


def _erode(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    eroded = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
    return eroded if eroded.any() else mask


def _filled(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Rellena el exterior de la máscara con la mediana interior (no con 0)."""
    if mask.all() or not mask.any():
        return img
    med = int(np.median(img[mask]))
    return np.where(mask, img, med).astype(np.uint8)


def _values(img: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    m = _safe_mask(mask, img.shape)
    vals = img[m]
    return vals if vals.size else img.ravel()


def _robust_contrast(vals: np.ndarray) -> float:
    if vals.size < 2:
        return 0.0
    return float(np.percentile(vals, 95) - np.percentile(vals, 5))


def _entropy_from_values(vals: np.ndarray, bins: int = 256) -> float:
    h, _ = np.histogram(vals, bins=bins, range=(0, 256), density=False)
    h = h.astype(float) / (h.sum() + 1e-9)
    return float(scipy_entropy(h + 1e-9))


# ---------------------------------------------------------------------------
# Intensidad
# ---------------------------------------------------------------------------

def feat_intensity_global(img: np.ndarray, mask: Optional[np.ndarray] = None) -> dict:
    """Media, desviación, contraste robusto (p95-p5) y entropía.

    Si ``mask`` se proporciona, todas las métricas se calculan solo dentro de
    la máscara.
    """
    vals = _values(img, mask)
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "contrast": _robust_contrast(vals),
        "entropy": _entropy_from_values(vals),
    }


def feat_intensity_regional(img: np.ndarray, mask: Optional[np.ndarray] = None) -> dict:
    """Las mismas métricas en las seis regiones anatómicas.

    Cada región se intersecta con la máscara pulmonar. Si la intersección tiene
    menos de 20 píxeles, la región se resume con los píxeles disponibles; si
    está vacía, se devuelve 0.
    """
    feats: dict = {}
    for name, (x, y, w, h) in REGIONS.items():
        roi = img[y:y + h, x:x + w]
        if mask is not None:
            roi_mask = mask[y:y + h, x:x + w]
            vals = roi[roi_mask] if roi_mask.any() else np.array([], dtype=roi.dtype)
        else:
            vals = roi.ravel()
        if vals.size == 0:
            feats[f"{name}_mean"] = 0.0
            feats[f"{name}_std"] = 0.0
            feats[f"{name}_contrast"] = 0.0
            feats[f"{name}_entropy"] = 0.0
            continue
        feats[f"{name}_mean"] = float(np.mean(vals))
        feats[f"{name}_std"] = float(np.std(vals))
        feats[f"{name}_contrast"] = _robust_contrast(vals)
        feats[f"{name}_entropy"] = _entropy_from_values(vals)
    return feats


# ---------------------------------------------------------------------------
# GLCM / LBP / Gabor
# ---------------------------------------------------------------------------

def feat_glcm(img: np.ndarray, mask: Optional[np.ndarray] = None) -> dict:
    """GLCM (64 niveles, 4 ángulos) con soporte de máscara.

    El exterior de la máscara se rellena con la mediana pulmonar y el cálculo
    se restringe a la caja de la máscara para no incluir fondo.
    """
    m = _safe_mask(mask, img.shape)
    work = _filled(img, m)
    if m.any():
        ys, xs = np.where(m)
        y0, y1 = max(0, ys.min() - 2), min(img.shape[0], ys.max() + 3)
        x0, x1 = max(0, xs.min() - 2), min(img.shape[1], xs.max() + 3)
        work = work[y0:y1, x0:x1]
    img64 = (work // 4).astype(np.uint8)
    glcm = graycomatrix(
        img64, distances=[1], angles=GLCM_ANGLES, levels=64,
        symmetric=True, normed=True,
    )
    feats: dict = {}
    for prop in GLCM_PROPS:
        vals = graycoprops(glcm, prop).flatten()
        feats[f"glcm_{prop}_mean"] = float(np.mean(vals))
        feats[f"glcm_{prop}_std"] = float(np.std(vals))
    return feats


def feat_lbp(img: np.ndarray, mask: Optional[np.ndarray] = None) -> dict:
    """Histograma LBP uniforme calculado solo sobre píxeles de la máscara.

    Se erosiona la máscara para evitar respuestas de borde en el contorno.
    """
    m = _safe_mask(mask, img.shape)
    work = _filled(img, m)
    lbp_img = local_binary_pattern(work, P=LBP_P, R=LBP_R, method="uniform")
    inner = _erode(m, radius=3)
    vals = lbp_img[inner] if inner.any() else lbp_img.ravel()
    hist, _ = np.histogram(vals, bins=LBP_BINS, range=(0, LBP_BINS), density=True)
    return {f"lbp_{i}": float(v) for i, v in enumerate(hist)}


def feat_gabor(img: np.ndarray, mask: Optional[np.ndarray] = None) -> dict:
    """Banco Gabor 4×4; media y desviación de la magnitud dentro de la máscara.

    Se implementa con ``cv2.getGaborKernel`` + ``cv2.filter2D`` (float32), que
    es ~10× más rápido que el filtrado FFT de scikit-image para 16 filtros.
    """
    m = _safe_mask(mask, img.shape)
    work = _filled(img, m).astype(np.float32)
    inner = _erode(m, radius=4)
    feats: dict = {}
    for freq in GABOR_FREQUENCIES:
        sigma = 0.56 / max(freq, 1e-6)
        ksize = int(2 * round(3 * sigma) + 1)
        ksize = min(max(ksize, 9), 127)
        for theta in GABOR_THETAS:
            kernel = cv2.getGaborKernel(
                (ksize, ksize), sigma, theta, 1.0 / freq, 0.5, ktype=cv2.CV_32F
            )
            real = cv2.filter2D(work, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
            # Respuesta real + respuesta en cuadratura (Hilbert aproximado).
            kernel_q = cv2.getGaborKernel(
                (ksize, ksize), sigma, theta, 1.0 / freq, 0.5,
                ktype=cv2.CV_32F, psi=np.pi / 2,
            )
            imag = cv2.filter2D(work, cv2.CV_32F, kernel_q, borderType=cv2.BORDER_REFLECT)
            mag = cv2.magnitude(real, imag)
            vals = mag[inner] if inner.any() else mag.ravel()
            label = f"gabor_f{freq:.1f}_t{int(np.degrees(theta))}"
            feats[f"{label}_mean"] = float(np.mean(vals))
            feats[f"{label}_std"] = float(np.std(vals))
    return feats


# ---------------------------------------------------------------------------
# HOG / PHOG
# ---------------------------------------------------------------------------

def feat_hog(img: np.ndarray) -> dict:
    """HOG 1764 features sobre el recorte torácico completo (sin máscara)."""
    fd = hog(
        img,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return {f"hog_{i}": float(v) for i, v in enumerate(fd)}


def _hog_region(region: np.ndarray, ppc: tuple[int, int]) -> np.ndarray:
    return hog(
        region,
        orientations=PHOG_ORIENTATIONS,
        pixels_per_cell=ppc,
        cells_per_block=PHOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    )


def feat_phog(img: np.ndarray) -> dict:
    """PHOG en dos niveles (1440 features). Se conserva para comparación."""
    h, w = img.shape
    feats: dict = {}
    l0 = _hog_region(img, PHOG_L0_PPC)
    for i, v in enumerate(l0):
        feats[f"phog_l0_{i}"] = float(v)
    idx = 0
    for row in range(2):
        for col in range(2):
            roi = img[row * h // 2:(row + 1) * h // 2,
                      col * w // 2:(col + 1) * w // 2]
            l1 = _hog_region(roi, PHOG_L1_PPC)
            for v in l1:
                feats[f"phog_l1_{idx}"] = float(v)
                idx += 1
    return feats


# ---------------------------------------------------------------------------
# Forma de la máscara (control de sesgo)
# ---------------------------------------------------------------------------

def feat_shape(mask: np.ndarray) -> dict:
    """Geometría de la máscara pulmonar, para el baseline de forma."""
    m = mask.astype(np.uint8)
    h, w = m.shape
    area = float(m.sum())
    ys, xs = np.where(m)
    if area == 0 or len(xs) == 0:
        base = {k: 0.0 for k in [
            "area_fraction", "bbox_w", "bbox_h", "aspect", "extent",
            "solidity", "perimeter_norm", "lr_balance", "centroid_x", "centroid_y",
        ]}
        base.update({f"shape_thumb_{i}": 0.0 for i in range(SHAPE_THUMB[0] * SHAPE_THUMB[1])})
        return base

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perim = float(sum(cv2.arcLength(c, True) for c in contours))
    hull_area = float(sum(cv2.contourArea(cv2.convexHull(c)) for c in contours)) or 1.0
    left = float(m[:, : w // 2].sum())
    right = float(m[:, w // 2:].sum())

    feats = {
        "area_fraction": area / (h * w),
        "bbox_w": bw / w,
        "bbox_h": bh / h,
        "aspect": bw / bh,
        "extent": area / (bw * bh),
        "solidity": min(area / hull_area, 1.0),
        "perimeter_norm": perim / (2 * (h + w)),
        "lr_balance": min(left, right) / max(max(left, right), 1.0),
        "centroid_x": float(xs.mean()) / w,
        "centroid_y": float(ys.mean()) / h,
    }
    thumb = cv2.resize(m * 255, SHAPE_THUMB, interpolation=cv2.INTER_AREA).astype(float) / 255.0
    feats.update({f"shape_thumb_{i}": float(v) for i, v in enumerate(thumb.ravel())})
    return feats


# ---------------------------------------------------------------------------
# Contexto / fuente (control de sesgo)
# ---------------------------------------------------------------------------

def feat_context(img: np.ndarray, context_mask: np.ndarray) -> dict:
    """Características del fondo: intensidad/textura + miniatura 16×16.

    La miniatura captura la firma de adquisición (esquinas, colimación,
    marcadores) que en TBX11K puede separar las clases sin ver el pulmón.
    """
    m = context_mask.astype(bool)
    feats: dict = {}
    for k, v in feat_intensity_global(img, m).items():
        feats[f"ctx_{k}"] = v
    for k, v in feat_intensity_regional(img, m).items():
        feats[f"ctx_{k}"] = v
    for k, v in feat_glcm(img, m).items():
        feats[f"ctx_{k}"] = v
    for k, v in feat_lbp(img, m).items():
        feats[f"ctx_{k}"] = v
    for k, v in feat_gabor(img, m).items():
        feats[f"ctx_{k}"] = v

    # Miniatura del fondo: se anula el campo torácico para no mezclar pulmón.
    bg = img.copy()
    bg[~m] = 0
    thumb = cv2.resize(bg, CONTEXT_THUMB, interpolation=cv2.INTER_AREA)
    feats.update({
        f"ctx_thumb_{i}": float(v) for i, v in enumerate(thumb.ravel())
    })
    return feats


# ---------------------------------------------------------------------------
# Extracción por imagen
# ---------------------------------------------------------------------------

def extract_features(
    image_path: str,
    label: int,
    class_name: str,
    include_hog: bool = True,
    include_phog: bool = False,
    include_context: bool = True,
    include_shape: bool = True,
) -> Optional[dict]:
    """Extrae todas las familias de características para una imagen.

    La textura usa la máscara pulmonar como soporte. HOG se calcula sobre la
    vista ``lung`` (exterior relleno con la mediana pulmonar) para no introducir
    bordes del fondo ni la geometría del recorte. El contexto se extrae de la
    imagen completa con la máscara de fondo.
    """
    bundle = preprocess_bundle(image_path)
    if bundle is None:
        return None

    img, mask = bundle["thorax"]
    quality = bundle["quality"]

    row: dict = {"image_path": image_path, "class_name": class_name, "label": label}
    row.update(feat_intensity_global(img, mask))
    row.update(feat_intensity_regional(img, mask))
    row.update(feat_glcm(img, mask))
    row.update(feat_lbp(img, mask))
    row.update(feat_gabor(img, mask))
    lung_img = bundle["lung"][0]
    if include_hog:
        row.update(feat_hog(lung_img))
    if include_phog:
        row.update(feat_phog(lung_img))
    if include_shape:
        row.update(feat_shape(mask))

    row.update({
        "mask_area_fraction": float(quality.area_fraction),
        "mask_lr_balance": float(quality.lr_balance),
        "mask_contrast": float(quality.contrast),
        "mask_fallback": int(quality.fallback),
    })

    if include_context:
        ctx_img, ctx_mask = bundle["context"]
        row.update(feat_context(ctx_img, ctx_mask))
    return row


def extract_all_features(
    df_base: pd.DataFrame,
    sample: bool = False,
    sample_size: int = 50,
    random_state: int = 42,
    **kwargs,
) -> tuple[pd.DataFrame, list[str]]:
    """Extrae características para todas las imágenes (o una muestra por clase)."""
    if sample:
        df_proc = pd.concat([
            df_base[df_base["class_name"] == clase].sample(
                min(sample_size, len(df_base[df_base["class_name"] == clase])),
                random_state=random_state,
            )
            for clase in df_base["class_name"].unique()
        ]).reset_index(drop=True)
        print(f"Modo muestra: {len(df_proc)} imágenes ({sample_size} por clase).")
    else:
        df_proc = df_base.copy()
        print(f"Procesando dataset completo: {len(df_proc)} imágenes.")

    rows: list[dict] = []
    errors: list[str] = []
    for _, row in tqdm(df_proc.iterrows(), total=len(df_proc), desc="Extrayendo características"):
        result = extract_features(
            row["image_path"], row["label"], row["class_name"], **kwargs
        )
        if result is not None:
            rows.append(result)
        else:
            errors.append(row["image_path"])

    features_df = pd.DataFrame(rows)
    print(f"\nProcesadas : {len(features_df)}")
    print(f"Errores    : {len(errors)}")
    print(f"Shape      : {features_df.shape}")
    return features_df, errors


# ---------------------------------------------------------------------------
# Grupos de características
# ---------------------------------------------------------------------------

def get_feature_groups(features_df: pd.DataFrame) -> dict[str, list[str]]:
    """Clasifica las columnas en familias para ablaciones y selección."""
    feat_cols = [c for c in features_df.columns if c not in META_COLS]

    texture = [
        c for c in feat_cols
        if c in {"mean", "std", "contrast", "entropy"}
        or any(c.startswith(r + "_") for r in REGIONS)
        or c.startswith(("glcm_", "lbp_", "gabor_"))
    ]
    hog_cols = [c for c in feat_cols if c.startswith("hog_")]
    phog_cols = [c for c in feat_cols if c.startswith("phog_")]
    shape = [c for c in feat_cols if c.startswith("shape_")]
    context = [c for c in feat_cols if c.startswith("ctx_")]
    quality = [c for c in feat_cols if c.startswith("mask_")]

    return {
        "texture": texture,
        "hog": hog_cols,
        "phog": phog_cols,
        "shape": shape,
        "context": context,
        "quality": quality,
    }


def print_feature_summary(features_df: pd.DataFrame) -> None:
    """Imprime filas, clases, nulos, infinitos y features por grupo."""
    feat_cols = [c for c in features_df.columns if c not in META_COLS]
    grupos = get_feature_groups(features_df)

    print(f"{'Filas':.<30} {len(features_df)}")
    print(f"{'Features':.<30} {len(feat_cols)}")
    print()
    print("Distribución de clases:")
    print(features_df["class_name"].value_counts().to_string())
    print()
    print(f"{'Valores nulos':.<30} {features_df[feat_cols].isnull().sum().sum()}")
    print(f"{'Valores infinitos':.<30} {np.isinf(features_df[feat_cols].values).sum()}")
    print()
    print("Features por grupo:")
    for nombre, cols in grupos.items():
        print(f"  {nombre:.<28} {len(cols):>5}")
    print(f"  {'TOTAL':.<28} {sum(len(v) for v in grupos.values()):>5}")


def verify_extractors(df_base: pd.DataFrame) -> None:
    """Verifica dimensiones de cada extractor sobre una imagen por clase."""
    expected = {
        "global": 4,
        "regional": 24,
        "GLCM": 10,
        "LBP": 26,
        "Gabor": 32,
        "HOG": 1764,
        "PHOG": 1440,
        "shape": 10 + SHAPE_THUMB[0] * SHAPE_THUMB[1],
        "context": 28 + 10 + 26 + 32 + CONTEXT_THUMB[0] * CONTEXT_THUMB[1],
    }

    row = None
    for class_name in ("health", "sick", "tb"):
        sample = df_base[df_base["class_name"] == class_name]
        if sample.empty:
            continue
        path = sample.iloc[0]["image_path"]
        row = extract_features(path, 0, class_name, include_hog=True,
                               include_phog=True, include_context=True)
        if row is not None:
            break
    assert row is not None, "No se pudo extraer características de ninguna imagen."

    grupos = {
        "global": feat_intensity_global,
        "regional": feat_intensity_regional,
        "GLCM": feat_glcm,
        "LBP": feat_lbp,
        "Gabor": feat_gabor,
        "HOG": feat_hog,
        "PHOG": feat_phog,
    }
    bundle = preprocess_bundle(path)
    img, mask = bundle["thorax"]
    for name, fn in grupos.items():
        n = len(fn(img, mask) if name in ("global", "regional", "GLCM", "LBP", "Gabor") else fn(img))
        print(f"{name:>10}: {n}")
        assert n == expected[name], f"{name}: esperado {expected[name]}, obtenido {n}"

    assert len(feat_shape(mask)) == expected["shape"]
    ctx_img, ctx_mask = bundle["context"]
    assert len(feat_context(ctx_img, ctx_mask)) == expected["context"]
    print("Verificación completada sin errores.")
