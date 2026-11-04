"""
cad_tbx11k.preprocessing
=========================
Preprocesamiento de radiografías torácicas del pipeline CAD-TBX11K.

Cadena principal (modo ``'thorax'``):

    imagen original → escala de grises → segmentación pulmonar a 512×512
    → recorte torácico (sin poner ceros) → resize 256×256
    → Bilateral Filter → CLAHE

La segmentación se delega en :mod:`src.lung_segmentation`, que implementa un
campo pulmonar anatómico validado con las cajas de lesión de TBX11K. El recorte
torácico conserva la textura original (incluidas las lesiones) y elimina el
fondo, los marcadores y las esquinas de la adquisición, que son la principal
fuente de atajo en TBX11K.

Modos disponibles
-----------------
``'thorax'``
    Recorte torácico + BF + CLAHE. **Modo principal.** No enmascara con ceros.
``'lung'``
    Como ``'thorax'``, pero el exterior de la máscara pulmonar se rellena con la
    mediana de intensidad pulmonar (no con 0) para evitar bordes artificiales.
``'context'``
    Imagen completa + máscara del **fondo** (complemento del campo torácico).
    Se usa para el control de sesgo por contexto/fuente.
``'full'``
    Imagen completa + BF + CLAHE, sin máscara.
``'mask'``
    Recorte con el exterior a 0. Solo por compatibilidad; **no** es el modo del
    pipeline porque introduce la forma de la máscara como atajo.

Referencias
-----------
- OpenCV bilateral filter : https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html
- OpenCV CLAHE            : https://docs.opencv.org/4.x/d6/db6/classcv_1_1CLAHE.html
- OpenCV resize           : https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .lung_segmentation import (
    MaskQuality,
    SEG_SIZE,
    segment_lungs as _segment_lungs_512,
    strict_lung_mask,
    thorax_bbox,
)

IMG_SIZE: Tuple[int, int] = (256, 256)
"""Tamaño objetivo de la imagen preprocesada (ancho × alto)."""

BF_D: int = 9
"""Diámetro del vecindario del Bilateral Filter."""

BF_SIGMA_COLOR: float = 80.0
"""Sigma en el espacio de intensidad del Bilateral Filter."""

BF_SIGMA_SPACE: float = 80.0
"""Sigma en el espacio de coordenadas del Bilateral Filter."""

CLAHE_CLIP_LIMIT: float = 2.0
"""Límite de recorte de CLAHE (valor habitual 2–4; 40 saturaba el contraste)."""

CLAHE_TILE_GRID: Tuple[int, int] = (8, 8)
"""Cuadrícula de teselas de CLAHE."""

MODES = ("thorax", "lung", "context", "full", "mask", "strict", "strict_harmonized")


# ---------------------------------------------------------------------------
# Pasos individuales
# ---------------------------------------------------------------------------

def load_grayscale(path: str) -> Optional[np.ndarray]:
    """Carga una imagen en escala de grises (uint8) o ``None`` si falla."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[ERROR] No se pudo leer la imagen: {path}")
    return img


def resize_image(img: np.ndarray, size: Tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """Redimensiona con interpolación de área (recomendada para reducción)."""
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def apply_bilateral_filter(
    img: np.ndarray,
    d: int = BF_D,
    sigma_color: float = BF_SIGMA_COLOR,
    sigma_space: float = BF_SIGMA_SPACE,
) -> np.ndarray:
    """Filtro bilateral: reduce ruido preservando bordes tisulares."""
    return cv2.bilateralFilter(img, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def apply_clahe(
    img: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid: Tuple[int, int] = CLAHE_TILE_GRID,
) -> np.ndarray:
    """CLAHE con límite de recorte bajo para no saturar opacidades."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(img)


def segment_lungs(img: np.ndarray) -> np.ndarray:
    """Compatibilidad: segmenta a 512 y devuelve la máscara al tamaño de entrada."""
    h, w = img.shape[:2]
    img512 = cv2.resize(img, SEG_SIZE, interpolation=cv2.INTER_AREA)
    mask, _ = _segment_lungs_512(img512)
    if (w, h) != SEG_SIZE:
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def segment_image(path: str) -> tuple[Optional[np.ndarray], Optional[MaskQuality]]:
    """Carga y segmenta una imagen a 512×512."""
    img = load_grayscale(path)
    if img is None:
        return None, None
    img512 = cv2.resize(img, SEG_SIZE, interpolation=cv2.INTER_AREA)
    mask, quality = _segment_lungs_512(img512)
    return mask, quality


def preprocess_with_mask(
    path: str,
    modo: str = "thorax",
) -> Optional[tuple[np.ndarray, Optional[np.ndarray], Optional[MaskQuality]]]:
    """Preprocesa una imagen y devuelve ``(imagen, máscara, calidad)``.

    La máscara está alineada con la imagen de 256×256 devuelta. En el modo
    ``'context'`` la máscara marca el **fondo** (complemento del campo torácico).
    """
    if modo not in MODES:
        raise ValueError(f"Modo de preprocesamiento desconocido: {modo}")

    if modo in ("strict", "strict_harmonized"):
        bundle = preprocess_bundle(path)
        if bundle is None:
            return None
        img, mask = bundle[modo]
        return img, mask, bundle["quality"]

    img = load_grayscale(path)
    if img is None:
        return None

    img512 = cv2.resize(img, SEG_SIZE, interpolation=cv2.INTER_AREA)
    mask512, quality = _segment_lungs_512(img512)

    if modo == "full":
        out = resize_image(img512)
        out = apply_bilateral_filter(out)
        out = apply_clahe(out)
        return out, None, quality

    if modo == "context":
        # Imagen completa; la máscara es el fondo (fuera del campo torácico).
        out = resize_image(img512)
        out = apply_bilateral_filter(out)
        out = apply_clahe(out)
        ctx_mask = ~mask512
        ctx_mask = cv2.resize(
            ctx_mask.astype(np.uint8), IMG_SIZE, interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        return out, ctx_mask, quality

    bbox = thorax_bbox(mask512)
    if bbox is None:
        bbox = (int(0.06 * 512), int(0.04 * 512), int(0.94 * 512), int(0.96 * 512))

    x0, y0, x1, y1 = bbox
    crop = img512[y0:y1, x0:x1]
    mask_crop = mask512[y0:y1, x0:x1]

    out = cv2.resize(crop, IMG_SIZE, interpolation=cv2.INTER_AREA)
    mask256 = cv2.resize(
        mask_crop.astype(np.uint8), IMG_SIZE, interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    out = apply_bilateral_filter(out)
    out = apply_clahe(out)

    if modo == "mask":
        out = np.where(mask256, out, 0).astype(np.uint8)
    elif modo == "lung":
        # Relleno suave: el exterior toma la mediana pulmonar, no 0.
        if mask256.any() and (~mask256).any():
            mediana = int(np.median(out[mask256]))
            out = np.where(mask256, out, mediana).astype(np.uint8)
    return out, mask256, quality


def preprocess(path: str, modo: str = "thorax") -> Optional[np.ndarray]:
    """Preprocesa una imagen y devuelve solo la imagen de 256×256."""
    result = preprocess_with_mask(path, modo=modo)
    if result is None:
        return None
    img, _, _ = result
    return img


def preprocess_bundle(path: str) -> Optional[dict]:
    """Segmenta una vez y devuelve las vistas ``thorax`` y ``context``.

    Evita repetir la segmentación cuando se extraen las características
    pulmonares y el control de fondo en la misma pasada.

    Retorna
    -------
    dict con claves ``'thorax'``, ``'context'``, ``'quality'`` y ``'bbox'``.
    """
    img = load_grayscale(path)
    if img is None:
        return None

    img512 = cv2.resize(img, SEG_SIZE, interpolation=cv2.INTER_AREA)
    mask512, quality = _segment_lungs_512(img512)
    bbox = thorax_bbox(mask512)
    if bbox is None:
        bbox = (int(0.06 * 512), int(0.04 * 512), int(0.94 * 512), int(0.96 * 512))

    # Vista torácica: recorte sin ceros.
    x0, y0, x1, y1 = bbox
    crop = img512[y0:y1, x0:x1]
    mask_crop = mask512[y0:y1, x0:x1]
    thorax = cv2.resize(crop, IMG_SIZE, interpolation=cv2.INTER_AREA)
    mask_thorax = cv2.resize(
        mask_crop.astype(np.uint8), IMG_SIZE, interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    thorax = apply_clahe(apply_bilateral_filter(thorax))

    # Vista de contexto: imagen completa y máscara de fondo.
    context = apply_clahe(apply_bilateral_filter(resize_image(img512)))
    mask_context = cv2.resize(
        (~mask512).astype(np.uint8), IMG_SIZE, interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    # Vista estricta: máscara erosionada, recorte ajustado y exterior relleno
    # con la mediana pulmonar. Elimina hombros, abdomen y esquinas.
    strict512 = strict_lung_mask(mask512)
    strict_bbox = thorax_bbox(
        strict512, margin_frac=0.01, min_width_frac=0.55, min_height_frac=0.55
    )
    if strict_bbox is None:
        strict_bbox = bbox
    sx0, sy0, sx1, sy1 = strict_bbox
    scrop = img512[sy0:sy1, sx0:sx1]
    smask_crop = strict512[sy0:sy1, sx0:sx1]
    strict = cv2.resize(scrop, IMG_SIZE, interpolation=cv2.INTER_AREA)
    mask_strict = cv2.resize(
        smask_crop.astype(np.uint8), IMG_SIZE, interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    strict = apply_clahe(apply_bilateral_filter(strict))
    if mask_strict.any() and (~mask_strict).any():
        med = int(np.median(strict[mask_strict]))
        strict = np.where(mask_strict, strict, med).astype(np.uint8)

    # Vista estricta armonizada: normalización por percentiles dentro del
    # pulmón, sin CLAHE, para reducir diferencias de adquisición.
    harm = strict.copy()
    if mask_strict.any():
        vals = harm[mask_strict]
        lo, hi = np.percentile(vals, 2), np.percentile(vals, 98)
        if hi > lo:
            harm = np.clip((harm.astype(float) - lo) / (hi - lo) * 255.0, 0, 255)
            harm = harm.astype(np.uint8)
        med = int(np.median(harm[mask_strict]))
        harm = np.where(mask_strict, harm, med).astype(np.uint8)

    return {
        "thorax": (thorax, mask_thorax),
        "context": (context, mask_context),
        "strict": (strict, mask_strict),
        "strict_harmonized": (harm, mask_strict),
        "quality": quality,
        "bbox": bbox,
    }


def preprocess_steps(path: str) -> Optional[dict[str, np.ndarray]]:
    """Estados intermedios para inspección visual (modo ``'full'``)."""
    img = load_grayscale(path)
    if img is None:
        return None
    resized = resize_image(img)
    bilateral = apply_bilateral_filter(resized)
    clahe_img = apply_clahe(bilateral)
    return {
        "original": resized,
        "bilateral": bilateral,
        "clahe": clahe_img,
    }


# ---------------------------------------------------------------------------
# Compatibilidad con la API anterior
# ---------------------------------------------------------------------------

def crop_bbox(
    mask: np.ndarray,
    margin: int = 8,
    min_width_frac: float = 0.75,
    min_height_frac: float = 0.80,
) -> Optional[Tuple[int, int, int, int]]:
    """Compatibilidad: caja torácica a partir de una máscara de cualquier tamaño."""
    h, w = mask.shape
    box = thorax_bbox(mask, margin_frac=0.0, min_width_frac=min_width_frac,
                      min_height_frac=min_height_frac)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return (max(0, x0 - margin), max(0, y0 - margin),
            min(w, x1 + margin), min(h, y1 + margin))
