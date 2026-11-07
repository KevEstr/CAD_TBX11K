"""Parches de lesión y de pulmón normal para el especialista TB."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .features import feat_gabor, feat_glcm, feat_lbp

PATCH_SIZE = 96
PATCH_STRIDE = 32


def _fill_outside(patch: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.any() and (~mask).any():
        med = int(np.median(patch[mask]))
        return np.where(mask, patch, med).astype(np.uint8)
    return patch


def _crop(
    img: np.ndarray,
    mask: np.ndarray,
    x0: int,
    y0: int,
    size: int,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    h, w = img.shape
    x1, y1 = x0 + size, y0 + size
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    patch = img[y0:y1, x0:x1]
    pmask = mask[y0:y1, x0:x1]
    if pmask.mean() < 0.35:
        return None
    return _fill_outside(patch, pmask), pmask


def patch_from_box(
    img: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[float, float, float, float],
    size: int = PATCH_SIZE,
    margin: float = 0.20,
    rng: Optional[np.random.RandomState] = None,
) -> Optional[np.ndarray]:
    x, y, bw, bh = bbox
    cx, cy = x + bw / 2.0, y + bh / 2.0
    side = max(bw, bh) * (1.0 + margin)
    side = max(side, size)
    if rng is not None:
        cx += rng.uniform(-0.08, 0.08) * bw
        cy += rng.uniform(-0.08, 0.08) * bh
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    recorte = _crop(img, mask, x0, y0, int(round(side)))
    if recorte is None:
        return None
    patch, _ = recorte
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)


def random_patch(
    img: np.ndarray,
    mask: np.ndarray,
    size: int = PATCH_SIZE,
    rng: Optional[np.random.RandomState] = None,
    attempts: int = 40,
) -> Optional[np.ndarray]:
    rng = rng or np.random.RandomState(0)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    for _ in range(attempts):
        idx = rng.randint(len(xs))
        cx, cy = int(xs[idx]), int(ys[idx])
        x0 = cx - size // 2
        y0 = cy - size // 2
        recorte = _crop(img, mask, x0, y0, size)
        if recorte is not None:
            return recorte[0]
    return None


def patch_features(patch: np.ndarray) -> dict:
    out = {}
    out.update(feat_glcm(patch))
    out.update(feat_lbp(patch))
    out.update(feat_gabor(patch))
    return out


def sliding_patches(
    img: np.ndarray,
    mask: np.ndarray,
    size: int = PATCH_SIZE,
    stride: int = PATCH_STRIDE,
):
    h, w = img.shape
    for y0 in range(0, h - size + 1, stride):
        for x0 in range(0, w - size + 1, stride):
            pmask = mask[y0:y0 + size, x0:x0 + size]
            if pmask.mean() < 0.60:
                continue
            patch = img[y0:y0 + size, x0:x0 + size]
            patch = _fill_outside(patch, pmask)
            yield x0, y0, patch
