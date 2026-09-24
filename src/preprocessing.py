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


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------

def preprocess(path: str) -> Optional[np.ndarray]:
    """
    Ejecuta la cadena completa de preprocesamiento sobre una imagen.

    Pasos:
    1. Carga en escala de grises.
    2. Resize a :data:`IMG_SIZE` con interpolación de área.
    3. Bilateral Filter (suavizado preservando bordes).
    4. CLAHE (realce de contraste local).

    Parámetros
    ----------
    path : str
        Ruta al archivo de imagen original.

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
    img = apply_bilateral_filter(img)
    img = apply_clahe(img)
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
