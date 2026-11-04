"""
cad_tbx11k.preprocessing
=========================
Módulo de preprocesamiento de radiografías torácicas del pipeline CAD-TBX11K.

Define la cadena de preprocesamiento estándar aplicada a cada imagen antes
de cualquier extracción de características:

    imagen original → escala de grises → resize 256×256 → Bilateral Filter → CLAHE

Cada paso tiene una función independiente para facilitar la inspección
y comparación de efectos intermedios.

Documentación de referencia
---------------------------
- OpenCV bilateral filter : https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed
- OpenCV CLAHE            : https://docs.opencv.org/4.x/d6/db6/classcv_1_1CLAHE.html
- OpenCV resize           : https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d
"""

from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

# ---------------------------------------------------------------------------
# Parámetros por defecto del pipeline
# ---------------------------------------------------------------------------

IMG_SIZE: Tuple[int, int] = (256, 256)
"""Tamaño objetivo al que se redimensionan todas las imágenes (ancho × alto)."""

BF_D: int = 9
"""Diámetro del vecindario utilizado por el Bilateral Filter."""

BF_SIGMA_COLOR: float = 80.0
"""Sigma en el espacio de color del Bilateral Filter.
Valores altos mezclan colores más distantes → mayor suavizado de color."""

BF_SIGMA_SPACE: float = 80.0
"""Sigma en el espacio de coordenadas del Bilateral Filter.
Valores altos mezclan píxeles más lejanos → mayor suavizado espacial."""

CLAHE_CLIP_LIMIT: float = 40.0
"""Límite de recorte del histograma para CLAHE.
Valores altos permiten mayor realce de contraste pero pueden amplificar ruido."""

CLAHE_TILE_GRID: Tuple[int, int] = (8, 8)
"""Tamaño de la cuadrícula de teselas para CLAHE.
Cada tesela recibe su propia ecualización de histograma."""

SEG_MIN_AREA: int = 500
"""Área mínima (px) de una componente para considerarla un pulmón."""

SEG_MAX_CY: float = 200.0
"""Centroide vertical máximo para aceptar una componente como pulmón."""


# ---------------------------------------------------------------------------
# Pasos individuales del pipeline
# ---------------------------------------------------------------------------

def load_grayscale(path: str) -> Optional[np.ndarray]:
    """
    Carga una imagen desde disco y la convierte a escala de grises.

    Parámetros
    ----------
    path : str
        Ruta al archivo de imagen.

    Retorna
    -------
    np.ndarray o None
        Imagen en escala de grises (dtype uint8) o ``None`` si la carga falla.

    Referencias
    -----------
    - cv2.imread: https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[ERROR] No se pudo leer la imagen: {path}")
    return img


def resize_image(img: np.ndarray, size: Tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """
    Redimensiona la imagen al tamaño objetivo utilizando interpolación
    de área (``INTER_AREA``), recomendada para reducción de escala porque
    minimiza el efecto de moiré.

    Parámetros
    ----------
    img : np.ndarray
        Imagen en escala de grises.
    size : tuple de int
        Tamaño destino ``(ancho, alto)``.

    Retorna
    -------
    np.ndarray
        Imagen redimensionada.

    Referencias
    -----------
    - cv2.resize: https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d
    """
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def apply_bilateral_filter(
    img: np.ndarray,
    d: int = BF_D,
    sigma_color: float = BF_SIGMA_COLOR,
    sigma_space: float = BF_SIGMA_SPACE,
) -> np.ndarray:
    """
    Aplica el Filtro Bilateral para reducir ruido preservando bordes.

    El Filtro Bilateral pondera los píxeles vecinos por similitud espacial
    **y** por similitud de intensidad, de modo que los bordes (transiciones
    abruptas de intensidad) no se suavizan.  Esto es especialmente útil en
    radiografías, donde las fronteras entre tejidos son clínicamente relevantes.

    Parámetros
    ----------
    img : np.ndarray
        Imagen en escala de grises (uint8).
    d : int
        Diámetro del vecindario de cada píxel.
    sigma_color : float
        Sigma en el espacio de intensidad.
    sigma_space : float
        Sigma en el espacio de coordenadas.

    Retorna
    -------
    np.ndarray
        Imagen filtrada.

    Referencias
    -----------
    - Bilateral Filter en OpenCV: https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed
    - Tomasi & Manduchi (1998): "Bilateral filtering for gray and color images."
    """
    return cv2.bilateralFilter(img, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def apply_clahe(
    img: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid: Tuple[int, int] = CLAHE_TILE_GRID,
) -> np.ndarray:
    """
    Aplica CLAHE (*Contrast Limited Adaptive Histogram Equalization*)
    para realzar el contraste local de la imagen.

    A diferencia de la ecualización global, CLAHE subdivide la imagen en
    teselas (``tile_grid``) y ecualiza el histograma en cada una, luego
    interpola en los bordes para evitar discontinuidades.  El parámetro
    ``clip_limit`` limita la amplificación del contraste para controlar
    el ruido en zonas homogéneas.

    Parámetros
    ----------
    img : np.ndarray
        Imagen en escala de grises (uint8), idealmente ya filtrada.
    clip_limit : float
        Umbral de recorte del histograma por tesela.
    tile_grid : tuple de int
        Número de teselas en ``(x, y)``.

    Retorna
    -------
    np.ndarray
        Imagen con contraste realzado.

    Referencias
    -----------
    - CLAHE en OpenCV: https://docs.opencv.org/4.x/d6/db6/classcv_1_1CLAHE.html
    - Zuiderveld (1994): "Contrast Limited Adaptive Histogram Equalization."
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(img)


def segment_lungs(img: np.ndarray) -> np.ndarray:
    """
    Segmentación heurística del campo pulmonar sobre una imagen 256×256.

    Pasos:
    1. Umbral de Otsu invertido para aislar regiones oscuras (pulmones).
    2. Eliminación de componentes que tocan el borde (fondo/fuera del cuerpo).
    3. Selección de la componente lateral izquierda y derecha más grandes.
    4. Relleno de huecos y dilatación suave.

    Es una segmentación **heurística**, no una segmentación clínica validada.
    Se usa para excluir el fondo antes de extraer características.

    Parámetros
    ----------
    img : np.ndarray
        Imagen en escala de grises, ya redimensionada a :data:`IMG_SIZE`.

    Retorna
    -------
    np.ndarray
        Máscara booleana con ``True`` en el campo pulmonar aproximado.

    Referencias
    -----------
    - Otsu (1979): "A threshold selection method from gray-level histograms."
    - OpenCV connected components: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html
    """
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    n, labels, stats, cent = cv2.connectedComponentsWithStats(th, 8)
    h, w = img.shape
    candidatos = []

    for i in range(1, n):
        x, y0, w0, h0, area = stats[i]
        if area < SEG_MIN_AREA:
            continue
        if x == 0 or y0 == 0 or x + w0 == w or y0 + h0 == h:
            continue
        cx, cy = cent[i]
        if cy > SEG_MAX_CY:
            continue
        candidatos.append((area, i, cx, cy))

    izquierda = [c for c in candidatos if c[2] < w / 2]
    derecha = [c for c in candidatos if c[2] >= w / 2]
    keep = []
    if izquierda:
        keep.append(max(izquierda, key=lambda x: x[0])[1])
    if derecha:
        keep.append(max(derecha, key=lambda x: x[0])[1])

    if not keep:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (w // 2, h // 2), (w // 3, h // 3), 0, 0, 360, 1, -1)
        return mask.astype(bool)

    mask = np.isin(labels, keep)
    mask = binary_fill_holes(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    return mask.astype(bool)


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------

def preprocess(path: str, segmentar: bool = True) -> Optional[np.ndarray]:
    """
    Ejecuta la cadena completa de preprocesamiento sobre una imagen.

    Pasos:
    1. Carga en escala de grises.
    2. Resize a :data:`IMG_SIZE` con interpolación de área.
    3. Segmentación heurística del campo pulmonar y enmascarado del fondo.
    4. Bilateral Filter (suavizado preservando bordes).
    5. CLAHE (realce de contraste local).

    El fondo se reemplaza por 0 para que la extracción de características
    no dependa del contexto fuera del pulmón.

    Parámetros
    ----------
    path : str
        Ruta al archivo de imagen original.
    segmentar : bool
        Si es ``True`` (por defecto), enmascara el fondo con la segmentación
        pulmonar. Si es ``False``, mantiene la imagen completa (útil para
        análisis de sesgo de contexto).

    Retorna
    -------
    np.ndarray o None
        Imagen preprocesada (uint8, shape ``IMG_SIZE``) o ``None`` si
        la imagen no puede cargarse.
    """
    img = load_grayscale(path)
    if img is None:
        return None
    img = resize_image(img)

    mask = segment_lungs(img) if segmentar else None

    img = apply_bilateral_filter(img)
    img = apply_clahe(img)

    if segmentar:
        img = np.where(mask, img, 0).astype(np.uint8)
    return img


def preprocess_steps(path: str) -> Optional[dict[str, np.ndarray]]:
    """
    Ejecuta el pipeline paso a paso y retorna cada estado intermedio.

    Útil para la comparación visual en los notebooks de análisis.

    Parámetros
    ----------
    path : str
        Ruta al archivo de imagen original.

    Retorna
    -------
    dict o None
        Diccionario con claves ``'original'``, ``'bilateral'`` y
        ``'clahe'``, o ``None`` si la imagen no puede cargarse.
    """
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
