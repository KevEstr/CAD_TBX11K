"""Segmentación pulmonar con un modelo preentrenado de tórax.

Se usa PSPNet de ``torchxrayvision`` (clases ``Left Lung`` y ``Right Lung``).
La máscara se umbraliza, se limpia y se dilata un margen pequeño para incluir
lesiones periféricas. Si el modelo no detecta pulmones, se devuelve un campo
torácico central marcado con ``fallback=True`` para no romper la extracción.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

SEG_SIZE: tuple[int, int] = (512, 512)

LUNG_THRESHOLD: float = 0.50
LUNG_MARGIN_PX: int = 12
MIN_COMPONENT_FRACTION: float = 0.005
MIN_AREA_FRACTION: float = 0.04
MAX_AREA_FRACTION: float = 0.60
MIN_LR_BALANCE: float = 0.20
STRICT_ERODE_PX: int = 12

_MODEL = None


@dataclass
class MaskQuality:
    """Métricas objetivas de una máscara pulmonar."""

    area_fraction: float
    area_left: int
    area_right: int
    lr_balance: float
    mean_inside: float
    mean_outside: float
    contrast: float
    fallback: bool
    reason: str = ""


def _disk(radius: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    return kernel


def _get_model():
    """Carga PSPNet una sola vez."""
    global _MODEL
    if _MODEL is None:
        try:
            import torch
            import torchxrayvision as xrv
        except ImportError as exc:
            raise ImportError(
                "La segmentación requiere torch, torchvision y torchxrayvision. "
                "Instala con: pip install torch torchvision torchxrayvision"
            ) from exc
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        model = xrv.baseline_models.chestx_det.PSPNet()
        model.eval()
        _MODEL = model
    return _MODEL


def _fallback_mask(img: np.ndarray, reason: str) -> tuple[np.ndarray, MaskQuality]:
    """Campo torácico central para casos en que el modelo no detecta pulmón."""
    h, w = img.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    eje_x = (w // 3, 2 * w // 3)
    eje_y = h // 2
    radio = (int(0.20 * w), int(0.34 * h))
    for cx in eje_x:
        cv2.ellipse(mask, (cx, eje_y), radio, 0, 0, 360, 1, -1)
    mask = binary_fill_holes(mask.astype(bool))
    q = mask_quality(img, mask, fallback=True, reason=reason)
    return mask, q


def mask_quality(
    img: np.ndarray,
    mask: np.ndarray,
    body: Optional[np.ndarray] = None,
    fallback: bool = False,
    reason: str = "",
) -> MaskQuality:
    """Calcula área, balance y contraste de una máscara."""
    h, w = mask.shape
    if body is None:
        body = np.zeros_like(mask, dtype=bool)
        ys, xs = np.where(mask)
        if len(xs):
            y0 = max(0, int(ys.min() - 0.05 * h))
            y1 = min(h, int(ys.max() + 0.05 * h))
            x0 = max(0, int(xs.min() - 0.05 * w))
            x1 = min(w, int(xs.max() + 0.05 * w))
            body[y0:y1, x0:x1] = True
        else:
            body[:] = True

    area_mask = int(mask.sum())
    area_left = int(mask[:, : w // 2].sum())
    area_right = int(mask[:, w // 2:].sum())
    balance = min(area_left, area_right) / max(max(area_left, area_right), 1)

    dentro = img[mask]
    fuera = img[body & ~mask]
    mean_in = float(dentro.mean()) if dentro.size else float("nan")
    mean_out = float(fuera.mean()) if fuera.size else float("nan")
    return MaskQuality(
        area_fraction=area_mask / (h * w),
        area_left=area_left,
        area_right=area_right,
        lr_balance=float(balance),
        mean_inside=mean_in,
        mean_outside=mean_out,
        contrast=float(mean_out - mean_in),
        fallback=bool(fallback),
        reason=reason,
    )


def segment_lungs(img: np.ndarray) -> tuple[np.ndarray, MaskQuality]:
    """Segmenta pulmones con PSPNet y devuelve máscara y calidad."""
    if img.ndim != 2:
        raise ValueError("segment_lungs espera una imagen en escala de grises.")
    if img.shape != SEG_SIZE:
        img = cv2.resize(img, SEG_SIZE, interpolation=cv2.INTER_AREA)

    import torch
    import torchxrayvision as xrv

    model = _get_model()
    normalizada = xrv.datasets.normalize(img, 255)
    tensor = torch.from_numpy(normalizada).unsqueeze(0).unsqueeze(0).float()

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0]

    idx_left = model.targets.index("Left Lung")
    idx_right = model.targets.index("Right Lung")
    prob = torch.maximum(probs[idx_left], probs[idx_right]).cpu().numpy()
    mask = (prob >= LUNG_THRESHOLD).astype(np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _disk(2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _disk(5))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = MIN_COMPONENT_FRACTION * SEG_SIZE[0] * SEG_SIZE[1]
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if not keep:
        return _fallback_mask(img, "PSPNet no detectó pulmones")

    mask = np.isin(labels, keep).astype(np.uint8)
    mask = binary_fill_holes(mask).astype(np.uint8)
    if LUNG_MARGIN_PX > 0:
        mask = cv2.dilate(mask, _disk(LUNG_MARGIN_PX)).astype(bool)
    mask = mask.astype(bool)

    q = mask_quality(img, mask)
    if not (MIN_AREA_FRACTION <= q.area_fraction <= MAX_AREA_FRACTION):
        return _fallback_mask(
            img, f"área fuera de rango: {q.area_fraction:.2f}"
        )
    if q.lr_balance < MIN_LR_BALANCE:
        return _fallback_mask(img, f"balance: {q.lr_balance:.2f}")
    return mask, q


def strict_lung_mask(mask: np.ndarray, erode_px: int = STRICT_ERODE_PX) -> np.ndarray:
    """Máscara pulmonar erosionada, sin pared torácica."""
    m = cv2.erode(mask.astype(np.uint8), _disk(erode_px)).astype(bool)
    if not m.any():
        return mask.astype(bool)
    n, labels, stats, cent = cv2.connectedComponentsWithStats(
        m.astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return m
    w = mask.shape[1]
    candidatos = []
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < 0.02 * mask.sum():
            continue
        cx, _ = cent[i]
        candidatos.append((area, i, cx))
    izq = [c for c in candidatos if c[2] < w / 2]
    der = [c for c in candidatos if c[2] >= w / 2]
    keep = []
    if izq:
        keep.append(max(izq, key=lambda x: x[0])[1])
    if der:
        keep.append(max(der, key=lambda x: x[0])[1])
    if not keep:
        return m
    out = np.isin(labels, keep)
    out = binary_fill_holes(out)
    out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_OPEN, _disk(2)).astype(bool)
    return out


def thorax_bbox(
    mask: np.ndarray,
    margin_frac: float = 0.04,
    min_width_frac: float = 0.70,
    min_height_frac: float = 0.72,
) -> Optional[tuple[int, int, int, int]]:
    """Caja torácica que contiene la máscara pulmonar."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1

    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    bw = max(x1 - x0, int(min_width_frac * w))
    bh = max(y1 - y0, int(min_height_frac * h))

    x0 = int(round(cx - bw / 2))
    x1 = int(round(cx + bw / 2))
    y0 = int(round(cy - bh / 2))
    y1 = int(round(cy + bh / 2))

    mx = int(round(margin_frac * w))
    my = int(round(margin_frac * h))
    x0, y0 = max(0, x0 - mx), max(0, y0 - my)
    x1, y1 = min(w, x1 + mx), min(h, y1 + my)

    if x1 - x0 < 64 or y1 - y0 < 64:
        return None
    return x0, y0, x1, y1
