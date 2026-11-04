"""
cad_tbx11k.lung_segmentation
============================
SegmentaciÃ³n pulmonar anatÃ³mica para radiografÃ­as de tÃ³rax (TBX11K).

El objetivo de este mÃ³dulo es producir una mÃ¡scara de **campo pulmonar** que:

1. excluya el fondo, los marcadores y el abdomen;
2. **incluya las opacidades y lesiones** (que son mÃ¡s claras que el aire
   pulmonar normal y que la segmentaciÃ³n por Otsu invertido deja fuera);
3. no introduzca la forma de la mÃ¡scara como atajo en las caracterÃ­sticas.

La estrategia es anatÃ³mica y multi-etapa:

- mÃ¡scara corporal por componentes conectadas al borde;
- realce de contraste local para estimar quÃ© regiones son "mÃ¡s oscuras que
  su entorno" (pulmones) sin depender de un Ãºnico umbral global;
- umbral adaptativo con inclusiÃ³n explÃ­cita de tejido brillante (lesiones);
- separaciÃ³n izquierda/derecha y eliminaciÃ³n del mediastino;
- cierre morfolÃ³gico y envolvente convexa por pulmÃ³n para absorber lesiones
  perifÃ©ricas;
- control de calidad con mÃ©tricas objetivas (Ã¡rea, simetrÃ­a, contraste).

La validaciÃ³n cuantitativa se hace con las cajas de lesiÃ³n anotadas de TBX11K
(``annotations/json/TBX11K_trainval.json``): una mÃ¡scara Ãºtil debe contener
una fracciÃ³n alta del Ã¡rea de las cajas de TB.

Uso
---
>>> from src.lung_segmentation import segment_lungs, mask_quality
>>> mask, info = segment_lungs(img_512)
>>> info['area_fraction'], info['lr_balance'], info['fallback']

Referencias
-----------
- Otsu (1979). A threshold selection method from gray-level histograms.
- OpenCV morphological operations:
  https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

# ---------------------------------------------------------------------------
# ParÃ¡metros
# ---------------------------------------------------------------------------

SEG_SIZE: tuple[int, int] = (512, 512)
"""ResoluciÃ³n de trabajo para segmentar (se reduce despuÃ©s a 256)."""

BODY_BG_DELTA: int = 12
"""Margen de intensidad sobre el nivel de fondo del borde para separar cuerpo."""

BODY_MIN_FRACTION: float = 0.25
"""FracciÃ³n mÃ­nima del Ã¡rea de la imagen que debe ocupar el cuerpo."""

LOCAL_SIGMA: float = 24.0
"""Sigma del filtro gaussiano usado para el contraste local (escala torÃ¡cica)."""

CAND_SCORE_PERCENTILE: float = 50.0
"""Percentil del score de oscuridad local usado como umbral inicial."""

LUNG_MARGIN_PX: int = 20
"""Margen de dilataciÃ³n (px a 512) alrededor del campo pulmonar.

Incluye lesiones perifÃ©ricas y opacidades de borde sin llegar al fondo. La
mÃ¡scara resultante es un *campo pulmonar con margen*, no una segmentaciÃ³n
clÃ­nica de parÃ©nquima; se valida con las cajas de lesiÃ³n de TBX11K.
"""

MIN_LUNG_FRACTION: float = 0.06
"""Ãrea mÃ­nima de cada pulmÃ³n respecto al cuerpo."""

MAX_LUNG_FRACTION: float = 0.48
"""Ãrea mÃ¡xima de cada pulmÃ³n respecto al cuerpo (evita cuerpo entero)."""

MIN_LR_BALANCE: float = 0.35
"""RelaciÃ³n mÃ­nima entre el pulmÃ³n pequeÃ±o y el grande."""

FALLBACK_AREA_RANGE: tuple[float, float] = (0.10, 0.75)
"""Rango aceptable de Ã¡rea pulmonar total respecto al cuerpo."""


@dataclass
class MaskQuality:
    """MÃ©tricas objetivas de una mÃ¡scara pulmonar."""

    area_fraction: float
    area_left: float
    area_right: float
    lr_balance: float
    mean_inside: float
    mean_outside: float
    contrast: float
    fallback: bool
    reason: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Utilidades morfolÃ³gicas
# ---------------------------------------------------------------------------

def _disk(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Devuelve la componente conectada de mayor Ã¡rea."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return np.zeros_like(mask, dtype=bool)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == idx


def _components_lateral(
    mask: np.ndarray,
    min_area: float,
    max_area: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Conserva la mejor componente a cada lado del centro.

    Se exige que el centroide estÃ© claramente lateral y que el Ã¡rea estÃ© en un
    rango anatÃ³mico. Devuelve dos mÃ¡scaras booleanas (izquierda, derecha).
    """
    h, w = mask.shape
    n, labels, stats, cent = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    candidatos = []
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        cx, _ = cent[i]
        # Evita el mediastino central.
        if abs(cx - w / 2.0) < 0.08 * w:
            continue
        candidatos.append((area, i, cx))

    izq = [c for c in candidatos if c[2] < w / 2.0]
    der = [c for c in candidatos if c[2] >= w / 2.0]

    left = np.zeros_like(mask, dtype=bool)
    right = np.zeros_like(mask, dtype=bool)
    if izq:
        left = labels == max(izq, key=lambda x: x[0])[1]
    if der:
        right = labels == max(der, key=lambda x: x[0])[1]
    return left, right


def _convex_hull(mask: np.ndarray) -> np.ndarray:
    """Envolvente convexa de una mÃ¡scara booleana."""
    puntos = cv2.findNonZero(mask.astype(np.uint8))
    if puntos is None:
        return np.zeros_like(mask, dtype=bool)
    hull = cv2.convexHull(puntos)
    salida = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillConvexPoly(salida, hull, 1)
    return salida.astype(bool)


# ---------------------------------------------------------------------------
# SegmentaciÃ³n
# ---------------------------------------------------------------------------

def body_mask(img: np.ndarray) -> np.ndarray:
    """MÃ¡scara del cuerpo (tÃ³rax + abdomen), excluyendo el fondo negro.

    El fondo se define como las componentes de intensidad baja conectadas al
    borde de la imagen. AsÃ­ se eliminan tambiÃ©n marcos y etiquetas del borde
    sin eliminar regiones oscuras internas (pulmones).
    """
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    h, w = blur.shape

    borde = np.concatenate([blur[0, :], blur[-1, :], blur[:, 0], blur[:, -1]])
    nivel_fondo = float(np.median(borde))
    p95 = float(np.percentile(blur, 95))
    umbral = min(nivel_fondo + BODY_BG_DELTA, 0.45 * p95)

    fondo_candidato = (blur <= umbral).astype(np.uint8)
    n, labels, _, _ = cv2.connectedComponentsWithStats(fondo_candidato, connectivity=8)
    if n <= 1:
        return np.ones_like(blur, dtype=bool)

    etiquetas_borde = set(labels[0, :]) | set(labels[-1, :])
    etiquetas_borde |= set(labels[:, 0]) | set(labels[:, -1])
    etiquetas_borde.discard(0)

    fondo = np.isin(labels, list(etiquetas_borde))
    cuerpo = ~fondo
    cuerpo = _largest_component(cuerpo)
    cuerpo = binary_fill_holes(cuerpo)
    cuerpo = cv2.morphologyEx(
        cuerpo.astype(np.uint8), cv2.MORPH_CLOSE, _disk(6)
    ).astype(bool)

    if cuerpo.sum() < BODY_MIN_FRACTION * h * w:
        return np.ones_like(blur, dtype=bool)
    return cuerpo


def thorax_envelope(img: np.ndarray, body: np.ndarray) -> np.ndarray:
    """Campo torÃ¡cico generoso: cuerpo sin mediastino central ni abdomen.

    No pretende ser una mÃ¡scara de parÃ©nquima fino. Es una regiÃ³n anatÃ³mica
    segura que contiene ambos pulmones y las lesiones, y que excluye fondo,
    brazos y abdomen. Se usa como prior de la segmentaciÃ³n fina y como
    respaldo marcado cuando la segmentaciÃ³n fina no supera el control de
    calidad.
    """
    h, w = img.shape
    ys, xs = np.where(body)
    if len(xs) == 0:
        # Ãšltimo recurso: elipses anatÃ³micas.
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (w // 3, h // 2), (w // 5, int(0.36 * h)), 0, 0, 360, 1, -1)
        cv2.ellipse(mask, (2 * w // 3, h // 2), (w // 5, int(0.36 * h)), 0, 0, 360, 1, -1)
        return mask.astype(bool)

    y0b, y1b = int(ys.min()), int(ys.max())
    x0b, x1b = int(xs.min()), int(xs.max())
    alto = max(y1b - y0b, 1)

    # Ancho torÃ¡cico robusto: mediana de anchos de fila en la zona media.
    anchos = []
    for y in range(y0b + int(0.25 * alto), y0b + int(0.70 * alto)):
        fila = np.where(body[y])[0]
        if len(fila) >= 2:
            anchos.append(fila[-1] - fila[0])
    ancho_torso = int(np.median(anchos)) if anchos else x1b - x0b
    ancho_torso = int(np.clip(ancho_torso, 0.50 * w, 0.94 * w))

    # Centro del tÃ³rax: mediana de los centros de fila (robusta a brazos).
    centros = []
    for y in range(y0b + int(0.25 * alto), y0b + int(0.70 * alto)):
        fila = np.where(body[y])[0]
        if len(fila) >= 2:
            centros.append((fila[0] + fila[-1]) / 2.0)
    cx = int(np.median(centros)) if centros else (x0b + x1b) // 2

    x0 = max(0, cx - ancho_torso // 2)
    x1 = min(w, cx + ancho_torso // 2)
    y0 = max(0, y0b + int(0.03 * alto))
    y1 = min(h, y0b + int(0.88 * alto))

    env = np.zeros((h, w), dtype=bool)
    env[y0:y1, x0:x1] = True
    env &= body

    # Banda mediastÃ­nica central: se excluye el corazÃ³n/mediastino, pero se
    # deja un margen para no cortar el pulmÃ³n medial.
    half = max(3, int(0.055 * ancho_torso))
    env[:, max(0, cx - half): min(w, cx + half)] = False
    return binary_fill_holes(env)


def _fallback_mask(img: np.ndarray, body: np.ndarray, reason: str) -> tuple[np.ndarray, MaskQuality]:
    """Respaldo anatÃ³mico: campo torÃ¡cico sin fondo ni mediastino central.

    A diferencia de dos elipses rÃ­gidas, este respaldo sigue el tamaÃ±o y la
    posiciÃ³n reales del tÃ³rax, por lo que incluye lesiones perifÃ©ricas y
    apicales. Queda marcado con ``fallback=True`` para que el anÃ¡lisis lo
    trate como una segmentaciÃ³n gruesa.
    """
    mask = thorax_envelope(img, body)
    q = mask_quality(img, mask, body, fallback=True, reason=reason)
    return mask, q


def mask_quality(
    img: np.ndarray,
    mask: np.ndarray,
    body: Optional[np.ndarray] = None,
    fallback: bool = False,
    reason: str = "",
) -> MaskQuality:
    """Calcula mÃ©tricas de calidad de una mÃ¡scara pulmonar."""
    if body is None:
        body = np.ones_like(mask, dtype=bool)
    area_body = max(int(body.sum()), 1)
    area_mask = int(mask.sum())
    h, w = mask.shape
    area_left = int(mask[:, : w // 2].sum())
    area_right = int(mask[:, w // 2:].sum())
    balance = min(area_left, area_right) / max(max(area_left, area_right), 1)

    dentro = img[mask]
    fuera = img[body & ~mask]
    mean_in = float(dentro.mean()) if dentro.size else float("nan")
    mean_out = float(fuera.mean()) if fuera.size else float("nan")
    return MaskQuality(
        area_fraction=area_mask / area_body,
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
    """Segmenta el campo pulmonar en una radiografÃ­a 512Ã—512.

    ParÃ¡metros
    ----------
    img : np.ndarray
        Imagen en escala de grises (uint8), idealmente 512Ã—512.

    Retorna
    -------
    mask : np.ndarray
        MÃ¡scara booleana del campo pulmonar (incluye lesiones).
    quality : MaskQuality
        MÃ©tricas objetivas y bandera de respaldo.
    """
    if img.ndim != 2:
        raise ValueError("segment_lungs espera una imagen en escala de grises.")
    if img.shape != SEG_SIZE:
        img = cv2.resize(img, SEG_SIZE, interpolation=cv2.INTER_AREA)

    h, w = img.shape
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    body = body_mask(img)
    envelope = thorax_envelope(img, body)

    # 1. Score de oscuridad local: positivo donde la imagen es mÃ¡s oscura que
    #    su entorno amplio (pulmones). Robusto a iluminaciÃ³n global.
    fondo_local = cv2.GaussianBlur(blur.astype(np.float32), (0, 0), LOCAL_SIGMA)
    score = fondo_local - blur.astype(np.float32)
    score_body = score[envelope]
    if score_body.size == 0:
        return _fallback_mask(img, body, "campo torÃ¡cico vacÃ­o")

    umbral_score = float(np.percentile(score_body, CAND_SCORE_PERCENTILE))
    cand = envelope & (score > max(umbral_score, 1.0))

    # 2. Relleno de huecos pequeÃ±os (vasos, lesiones claras) antes de separar.
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE, _disk(9)).astype(bool)
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, _disk(4)).astype(bool)

    # 3. Separar pulmones y descartar mediastino/abdomen.
    area_body = max(int(body.sum()), 1)
    left, right = _components_lateral(
        cand,
        min_area=MIN_LUNG_FRACTION * area_body,
        max_area=MAX_LUNG_FRACTION * area_body,
    )

    # Si un lado falla, se intenta con el candidato mÃ¡s laxo.
    if not left.any() or not right.any():
        cand2 = envelope & (score > float(np.percentile(score_body, 38.0)))
        cand2 = cv2.morphologyEx(cand2.astype(np.uint8), cv2.MORPH_CLOSE, _disk(15)).astype(bool)
        left2, right2 = _components_lateral(
            cand2,
            min_area=0.4 * MIN_LUNG_FRACTION * area_body,
            max_area=MAX_LUNG_FRACTION * area_body,
        )
        if not left.any() and left2.any():
            left = left2
        if not right.any() and right2.any():
            right = right2

    if not left.any() or not right.any():
        return _fallback_mask(img, body, "no se detectaron dos pulmones")

    # 4. Envolvente convexa por pulmÃ³n: absorbe lesiones perifÃ©ricas y
    #    opacidades que no superan el umbral de oscuridad.
    left = _convex_hull(left)
    right = _convex_hull(right)

    # 5. Margen perilesional y restricciÃ³n al campo torÃ¡cico (ya excluye
    #    fondo, brazos y mediastino).
    margen = _disk(LUNG_MARGIN_PX)
    left = cv2.dilate(left.astype(np.uint8), margen).astype(bool) & envelope
    right = cv2.dilate(right.astype(np.uint8), margen).astype(bool) & envelope

    mask = left | right
    mask = binary_fill_holes(mask)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, _disk(5)).astype(bool)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, _disk(3)).astype(bool)

    q = mask_quality(img, mask, body)
    total_ok = FALLBACK_AREA_RANGE[0] <= q.area_fraction <= FALLBACK_AREA_RANGE[1]
    if not total_ok or q.lr_balance < MIN_LR_BALANCE or q.contrast <= 0:
        motivo = []
        if not total_ok:
            motivo.append(f"Ã¡rea={q.area_fraction:.2f}")
        if q.lr_balance < MIN_LR_BALANCE:
            motivo.append(f"balance={q.lr_balance:.2f}")
        if q.contrast <= 0:
            motivo.append("contraste<=0")
        return _fallback_mask(img, body, ",".join(motivo))

    return mask, q


STRICT_ERODE_PX: int = 12
"""Erosión (px a 512) para obtener un campo pulmonar estricto.

La máscara principal es generosa para incluir lesiones. Para el análisis de
textura sin pared torácica se erode el borde: se pierde algo de periferia,
pero se elimina la señal de pared/mediastino.
"""


def strict_lung_mask(mask: np.ndarray, erode_px: int = STRICT_ERODE_PX) -> np.ndarray:
    """Máscara pulmonar estricta: erosión + componentes principales.

    No se usa para validar cobertura de lesiones (para eso está la máscara
    generosa). Se usa para extraer textura de parénquima y comprobar si la
    señal sobrevive al eliminar la pared torácica.
    """
    m = cv2.erode(mask.astype(np.uint8), _disk(erode_px)).astype(bool)
    if not m.any():
        return mask.astype(bool)
    # Conservar las dos componentes laterales principales.
    n, labels, stats, cent = cv2.connectedComponentsWithStats(
        m.astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return m
    h, w = mask.shape
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
    """Caja torÃ¡cica que contiene la mÃ¡scara pulmonar, sin recortar lesiones.

    Se expande a un tamaÃ±o mÃ­nimo para conservar pared torÃ¡cica y diafragma.
    """
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

