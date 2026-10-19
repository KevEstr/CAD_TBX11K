"""
cad_tbx11k.features
====================
Módulo de extracción de características del pipeline CAD-TBX11K.

Convierte cada radiografía preprocesada en un vector numérico de 4199 características
agrupadas en doce grupos:

+----------------------+-------------------------------------------+----------+
| Grupo                | Qué captura                               | Features |
+======================+===========================================+==========+
| Intensidad global    | brillo, variabilidad, rango dinámico,     |        4 |
|                      | desorden del histograma                   |          |
+----------------------+-------------------------------------------+----------+
| Intensidad regional  | las mismas 4 métricas en 6 zonas          |       24 |
|                      | anatómicas (SI, SD, MI, MD, II, ID)       |          |
+----------------------+-------------------------------------------+----------+
| GLCM                 | relación entre píxeles vecinos —          |       10 |
|                      | textura del tejido pulmonar               |          |
+----------------------+-------------------------------------------+----------+
| LBP                  | micro-patrones locales —                  |       26 |
|                      | bordes finos y transiciones               |          |
+----------------------+-------------------------------------------+----------+
| HOG                  | gradientes orientados —                   |     1764 |
|                      | forma y estructura                        |          |
+----------------------+-------------------------------------------+----------+
| Gabor                | filtros de Gabor —                        |       32 |
|                      | características de textura                |          |
+----------------------+-------------------------------------------+----------+
| PHOG                 | HOG extendido con parámetros              |      1440|
|                      | adicionales                               |          |
+----------------------+-------------------------------------------+----------+
| Estadística avanzada | distribución, asimetría                   |    77    |
|  global/regional     | y dispersión de intensidades              |          |
+----------------------+-------------------------------------------+----------+
| GLCM multiescala     |                                           |          |
| /regional            | textura a distintas escalas y regiones    |      270 |
|                      |                                           |          |
+----------------------+-------------------------------------------+----------+
| LBP multiescala      |   microtexturas a diferentes escalas      |       28 |
|                      |                                           |          |
+----------------------+-------------------------------------------+----------+
| Asimetría bilateral  |   diferencias entre ambos hemitórax       |        12|
|                      |                                           |          |
+----------------------+-------------------------------------------+----------+
| GIST                 |Estructura y contexto global de la imagen  |     512  |
|                      |                                           |          |
+----------------------+-------------------------------------------+----------+
| **Total**            |                                           | **4199** |
+----------------------+-------------------------------------------+----------+

Documentación de referencia
---------------------------
- scipy.stats.entropy : https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html
- skimage GLCM        : https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.graycomatrix
- skimage LBP         : https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.local_binary_pattern
- skimage HOG         : https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hog
- OpenCV calcHist     : https://docs.opencv.org/4.x/d6/dc7/group__imgproc__hist.html
"""

from typing import Optional

import cv2
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy, skew, kurtosis
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from skimage.filters import gabor as skimage_gabor
from tqdm import tqdm

from .preprocessing import preprocess

# ---------------------------------------------------------------------------
# Definición de regiones anatómicas
# ---------------------------------------------------------------------------

REGIONS: dict[str, tuple[int, int, int, int]] = {
    "SI": (0,   0,   128, 86),
    "SD": (128, 0,   128, 86),
    "MI": (0,   86,  128, 85),
    "MD": (128, 86,  128, 85),
    "II": (0,   171, 128, 85),
    "ID": (128, 171, 128, 85),
}
"""
División en seis regiones anatómicas sobre la imagen 256×256.

Cada entrada es ``(x, y, ancho, alto)`` en píxeles.

Distribución::

    ┌─────────────┬─────────────┐
    │  SI         │  SD         │  ← zona superior
    ├─────────────┼─────────────┤
    │  MI         │  MD         │  ← zona media
    ├─────────────┼─────────────┤
    │  II         │  ID         │  ← zona inferior
    └─────────────┴─────────────┘
         col 0–127    col 128–255

El análisis en ``04_Analisis_Regional.ipynb`` justifica esta segmentación
mostrando que las regiones superiores (SI, SD) presentan diferencias de
intensidad estadísticamente relevantes entre las clases ``health`` y ``tb``.
"""

# ---------------------------------------------------------------------------
# Parámetros de los extractores
# ---------------------------------------------------------------------------

GLCM_PROPS: list[str] = ["contrast", "correlation", "energy", "homogeneity", "dissimilarity"]
"""Propiedades GLCM calculadas por cada combinación ángulo/distancia."""

GLCM_ANGLES: list[float] = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
"""Cuatro ángulos para invarianza a la orientación (0°, 45°, 90°, 135°)."""

LBP_P: int = 24
"""Número de puntos de muestreo en el círculo LBP."""

LBP_R: float = 3.0
"""Radio del círculo de muestreo LBP en píxeles."""

LBP_BINS: int = LBP_P + 2
"""Número de bins del histograma LBP (``P + 2`` para el método ``uniform``)."""

HOG_ORIENTATIONS: int = 9
"""Número de orientaciones del gradiente para HOG."""

HOG_PIXELS_PER_CELL: tuple[int, int] = (32, 32)
"""Tamaño de cada celda HOG en píxeles.

Se usan celdas de 32×32 (en lugar del estándar 16×16) para obtener
1764 features en lugar de ~8100, manteniendo un balance adecuado
entre detalle y dimensionalidad dado el tamaño del dataset (8400 muestras)."""

HOG_CELLS_PER_BLOCK: tuple[int, int] = (2, 2)
"""Número de celdas por bloque para la normalización HOG."""

# Gabor
GABOR_FREQUENCIES: list[float] = [0.1, 0.2, 0.3, 0.4]
"""Frecuencias de los filtros Gabor en ciclos/píxel.

Cubren desde estructuras gruesas (~10 px, p. ej. cavidades)
hasta estructuras finas (~2.5 px, p. ej. bordes de infiltrados)."""

GABOR_THETAS: list[float] = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
"""Orientaciones de los filtros Gabor (0°, 45°, 90°, 135°).

Las mismas cuatro orientaciones que se usan en GLCM,
lo que facilita la comparación directa entre ambos descriptores."""

# PHOG
PHOG_ORIENTATIONS: int = 8
"""Número de orientaciones del gradiente para PHOG."""

PHOG_CELLS_PER_BLOCK: tuple[int, int] = (2, 2)
"""Celdas por bloque para normalización en PHOG."""

PHOG_L0_PPC: tuple[int, int] = (64, 64)
"""pixels_per_cell en nivel 0 de PHOG (imagen 256×256 completa).

Con ppc=64: 4×4 celdas → 3×3 bloques → 3×3×2×2×8 = 288 features."""

PHOG_L1_PPC: tuple[int, int] = (32, 32)
"""pixels_per_cell en nivel 1 de PHOG (cuadrantes 128×128).

Con ppc=32 sobre 128×128: 4×4 celdas → 3×3 bloques → 3×3×2×2×8 = 288 features.
4 cuadrantes × 288 = 1152 features."""

# Features adicionales
ADVANCED_PERCENTILES: tuple[int, ...] = (10, 25, 50, 75, 90)

GLCM_DISTANCES_MS: list[int] = [1, 2, 4, 8]

# La escala (24, 3) ya está representada por feat_lbp(); se agregan solo
# escalas complementarias para evitar duplicar exactamente las mismas 26 features.
LBP_EXTRA_SCALES: list[tuple[int, float]] = [(8, 1.0), (16, 2.0)]


GIST_FREQUENCIES: list[float] = [0.05, 0.1, 0.2, 0.4]
GIST_ORIENTATIONS: list[float] = [k * np.pi / 8 for k in range(8)]
GIST_GRID: tuple[int, int] = (4, 4)


# ---------------------------------------------------------------------------
# Extractor 1: Intensidad global
# ---------------------------------------------------------------------------

def feat_intensity_global(img: np.ndarray) -> dict:
    """
    Calcula cuatro métricas de intensidad sobre la imagen completa.

    - **mean**     : brillo promedio global.
    - **std**      : variabilidad de brillo — tejidos más heterogéneos
                     (p. ej. infiltrados) producen mayor desviación.
    - **contrast** : diferencia entre el píxel más brillante y el más oscuro.
    - **entropy**  : desorden del histograma de intensidad — lesiones TB
                     aumentan la heterogeneidad y, con ello, la entropía.

    Parámetros
    ----------
    img : np.ndarray
        Imagen en escala de grises preprocesada (uint8).

    Retorna
    -------
    dict
        Cuatro entradas: ``mean``, ``std``, ``contrast``, ``entropy``.

    Referencias
    -----------
    - scipy.stats.entropy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html
    - cv2.calcHist: https://docs.opencv.org/4.x/d6/dc7/group__imgproc__hist.html
    """
    h = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
    h_norm = h / (h.sum() + 1e-9)
    return {
        "mean":     float(np.mean(img)),
        "std":      float(np.std(img)),
        # int() explícito para evitar overflow en la resta de uint8
        "contrast": float(int(img.max()) - int(img.min())),
        "entropy":  float(scipy_entropy(h_norm + 1e-9)),
    }


# ---------------------------------------------------------------------------
# Extractor 2: Intensidad regional
# ---------------------------------------------------------------------------

def feat_intensity_regional(img: np.ndarray) -> dict:
    """
    Calcula las mismas cuatro métricas de :func:`feat_intensity_global`
    en cada una de las seis regiones anatómicas definidas en :data:`REGIONS`.

    El análisis regional captura la distribución espacial de la intensidad,
    que se diluye cuando se calculan métricas únicamente a nivel global.
    En TB, las regiones superiores (SI, SD) suelen presentar mayor intensidad
    y heterogeneidad que en imágenes sanas.

    Parámetros
    ----------
    img : np.ndarray
        Imagen en escala de grises preprocesada (uint8, shape 256×256).

    Retorna
    -------
    dict
        24 entradas con el patrón ``{REGION}_{metrica}``
        (p. ej. ``SI_mean``, ``SD_entropy``).
    """
    feats: dict = {}
    for name, (x, y, w, h) in REGIONS.items():
        roi = img[y:y + h, x:x + w]
        hr = cv2.calcHist([roi], [0], None, [256], [0, 256]).flatten()
        hr_norm = hr / (hr.sum() + 1e-9)
        feats[f"{name}_mean"]     = float(np.mean(roi))
        feats[f"{name}_std"]      = float(np.std(roi))
        feats[f"{name}_contrast"] = float(int(roi.max()) - int(roi.min()))
        feats[f"{name}_entropy"]  = float(scipy_entropy(hr_norm + 1e-9))
    return feats


# ---------------------------------------------------------------------------
# Extractor 3: GLCM
# ---------------------------------------------------------------------------

def feat_glcm(img: np.ndarray) -> dict:
    """
    Calcula la matriz de co-ocurrencia de niveles de gris (GLCM) y extrae
    cinco propiedades de textura, cada una resumida como media y desviación
    estándar sobre los cuatro ángulos.

    **Propiedades:**
    - ``contrast``       : variación local de intensidad entre pares de píxeles.
    - ``correlation``    : dependencia lineal entre píxeles vecinos.
    - ``energy``         : uniformidad — imágenes homogéneas tienen mayor energía.
    - ``homogeneity``    : cercanía de los elementos de la GLCM a la diagonal.
    - ``dissimilarity``  : similar al contraste pero con peso lineal.

    Se usan 64 niveles de gris (``img // 4``) por eficiencia y robustez al ruido;
    con 256 niveles la GLCM sería dispersa y el cálculo costoso.

    La media sobre los cuatro ángulos produce un descriptor invariante a la
    orientación de las estructuras anatómicas.  La desviación estándar sobre
    ángulos captura la anisotropía de la textura.

    Parámetros
    ----------
    img : np.ndarray
        Imagen en escala de grises preprocesada (uint8, shape 256×256).

    Retorna
    -------
    dict
        10 entradas con el patrón ``glcm_{propiedad}_{mean|std}``.

    Referencias
    -----------
    - graycomatrix: https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.graycomatrix
    - graycoprops: https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.graycoprops
    - Haralick et al. (1973): "Textural features for image classification."
    """
    img64 = (img // 4).astype(np.uint8)
    glcm = graycomatrix(
        img64,
        distances=[1],
        angles=GLCM_ANGLES,
        levels=64,
        symmetric=True,
        normed=True,
    )
    feats: dict = {}
    for prop in GLCM_PROPS:
        vals = graycoprops(glcm, prop).flatten()
        feats[f"glcm_{prop}_mean"] = float(np.mean(vals))
        feats[f"glcm_{prop}_std"]  = float(np.std(vals))
    return feats


# ---------------------------------------------------------------------------
# Extractor 4: LBP
# ---------------------------------------------------------------------------

def feat_lbp(img: np.ndarray) -> dict:
    """
    Calcula el histograma de Local Binary Patterns (LBP) con el método
    ``uniform``.

    LBP compara cada píxel con sus vecinos equidistantes en un círculo de
    radio :data:`LBP_R` y codifica las comparaciones como un número binario.
    El método ``uniform`` retiene solo los patrones con ≤ 2 transiciones
    0→1, que corresponden a estructuras locales significativas (bordes,
    esquinas, puntos).  Los patrones ruidosos se agrupan en un solo bin.

    Con :data:`LBP_P` = 24 vecinos a radio 3 se obtienen 26 bins (P + 2).
    El histograma se normaliza (``density=True``) para ser independiente
    del tamaño de la imagen.

    Parámetros
    ----------
    img : np.ndarray
        Imagen en escala de grises preprocesada (uint8, shape 256×256).

    Retorna
    -------
    dict
        26 entradas con el patrón ``lbp_{i}`` (i ∈ 0..25).

    Referencias
    -----------
    - local_binary_pattern: https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.local_binary_pattern
    - Ojala et al. (2002): "Multiresolution gray-scale and rotation invariant
      texture classification with local binary patterns."
    """
    lbp_img = local_binary_pattern(img, P=LBP_P, R=LBP_R, method="uniform")
    hist, _ = np.histogram(lbp_img.ravel(), bins=LBP_BINS, range=(0, LBP_BINS), density=True)
    return {f"lbp_{i}": float(v) for i, v in enumerate(hist)}


# ---------------------------------------------------------------------------
# Extractor 5: HOG
# ---------------------------------------------------------------------------

def feat_hog(img: np.ndarray) -> dict:
    """
    Calcula el descriptor HOG (*Histogram of Oriented Gradients*).

    HOG divide la imagen en celdas de :data:`HOG_PIXELS_PER_CELL` píxeles,
    computa un histograma de orientaciones de gradiente en cada celda y
    normaliza en bloques de :data:`HOG_CELLS_PER_BLOCK` celdas con L2-Hys
    para robustez ante cambios de iluminación y contraste.

    Se usa ``pixels_per_cell=(32, 32)`` en lugar del estándar ``(16, 16)``
    para controlar la dimensionalidad:

    - Con 16×16: ~8100 features (problemático con ~8400 muestras disponibles).
    - Con 32×32: 1764 features (dimensionalidad manejable sin perder detalle estructural relevante).

    Parámetros
    ----------
    img : np.ndarray
        Imagen en escala de grises preprocesada (uint8, shape 256×256).

    Retorna
    -------
    dict
        1764 entradas con el patrón ``hog_{i}``.

    Referencias
    -----------
    - skimage HOG: https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hog
    - Dalal & Triggs (2005): "Histograms of oriented gradients for human detection."
    """
    fd = hog(
        img,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return {f"hog_{i}": float(v) for i, v in enumerate(fd)}


# ---------------------------------------------------------------------------
# Extractor 6: Gabor
# ---------------------------------------------------------------------------

def feat_gabor(img: np.ndarray) -> dict:
    """
    Aplica un banco de filtros Gabor y resume cada respuesta con media y
    desviación estándar de la magnitud.

    Un filtro Gabor combina una sinusoide de frecuencia ``f`` orientada a
    ``θ`` con una envolvente gaussiana.  La magnitud de la respuesta
    ``sqrt(real² + imag²)`` mide cuánta energía tiene la imagen en esa
    frecuencia y orientación local.

    **Banco utilizado:**

    - 4 frecuencias: 0.1, 0.2, 0.3, 0.4 ciclos/píxel
      (estructuras de ~10 px hasta ~2.5 px).
    - 4 orientaciones: 0°, 45°, 90°, 135°
      (mismas que GLCM para comparación directa).

    Por cada uno de los 4×4 = 16 filtros se calculan mean y std →
    **32 features en total**.

    Parámetros
    ----------
    img : np.ndarray
        Imagen en escala de grises preprocesada (uint8, shape 256×256).

    Retorna
    -------
    dict
        32 entradas con el patrón ``gabor_f{freq}_t{grados}_{mean|std}``.

    Referencias
    -----------
    - skimage gabor: https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.gabor
    - Chauhan et al.: "Role of Gist and PHOG features in computer-aided
      diagnosis of tuberculosis without segmentation."
    """
    img_float = img.astype(float)
    feats: dict = {}
    for freq in GABOR_FREQUENCIES:
        for theta in GABOR_THETAS:
            real, imag = skimage_gabor(img_float, frequency=freq, theta=theta)
            magnitude  = np.sqrt(real ** 2 + imag ** 2)
            label = f"gabor_f{freq:.1f}_t{int(np.degrees(theta))}"
            feats[f"{label}_mean"] = float(np.mean(magnitude))
            feats[f"{label}_std"]  = float(np.std(magnitude))
    return feats


# ---------------------------------------------------------------------------
# Extractor 7: PHOG
# ---------------------------------------------------------------------------

def _hog_region(region: np.ndarray, ppc: tuple[int, int]) -> np.ndarray:
    """Calcula HOG sobre una región con los parámetros de PHOG."""
    return hog(
        region,
        orientations=PHOG_ORIENTATIONS,
        pixels_per_cell=ppc,
        cells_per_block=PHOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    )


def feat_phog(img: np.ndarray) -> dict:
    """
    Calcula el descriptor Pyramid HOG (PHOG) en dos niveles de resolución
    espacial.

    PHOG extiende HOG añadiendo información sobre *dónde* aparecen las
    orientaciones de gradiente en la imagen.  Esto es relevante en
    radiografías de tórax porque la distribución espacial de las lesiones
    (p. ej. predominio en lóbulos superiores en TB activa) aporta
    información diagnóstica que un HOG global no captura.

    **Pirámide utilizada** (sobre imagen 256×256):

    .. code-block:: text

        Nivel 0 — imagen completa (256×256), ppc = (64, 64)
        ┌─────────────────────────────────────┐
        │  HOG global → 288 features          │
        └─────────────────────────────────────┘

        Nivel 1 — 4 cuadrantes (128×128 c/u), ppc = (32, 32)
        ┌──────────────────┬──────────────────┐
        │  HOG → 288       │  HOG → 288       │
        ├──────────────────┼──────────────────┤
        │  HOG → 288       │  HOG → 288       │
        └──────────────────┴──────────────────┘
        4 × 288 = 1152 features

    **Total: 288 + 1152 = 1440 features.**

    El nivel 2 (16 cuadrantes de 64×64) no es viable porque HOG requiere
    al menos 128×128 con los parámetros elegidos.

    Parámetros
    ----------
    img : np.ndarray
        Imagen en escala de grises preprocesada (uint8, shape 256×256).

    Retorna
    -------
    dict
        1440 entradas con el patrón ``phog_l{nivel}_{i}``.

    Referencias
    -----------
    - Chauhan et al.: "Role of Gist and PHOG features in computer-aided
      diagnosis of tuberculosis without segmentation."
    - Anna Bosch et al. (2007): "Representing shape with a spatial pyramid
      kernel."
    """
    h, w = img.shape
    feats: dict = {}

    # Nivel 0 — imagen completa
    l0 = _hog_region(img, PHOG_L0_PPC)
    for i, v in enumerate(l0):
        feats[f"phog_l0_{i}"] = float(v)

    # Nivel 1 — 4 cuadrantes de 128×128
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
# Extractor 8: Estadísticas de intensidad avanzadas
# ---------------------------------------------------------------------------

def _advanced_stats(region: np.ndarray, prefix: str) -> dict:
    """
    Resume una distribución de intensidad mediante estadísticos robustos y
    momentos de orden superior.

    Complementa ``mean`` y ``std`` con mediana, varianza, asimetría, curtosis,
    percentiles, rango intercuartílico (IQR), desviación absoluta mediana (MAD)
    y coeficiente de variación. Estas medidas describen cambios en la forma del
    histograma que pueden quedar ocultos en los cuatro estadísticos básicos.

    Parámetros
    ----------
    region : np.ndarray
        Imagen o región de interés en escala de grises.
    prefix : str
        Prefijo utilizado en los nombres de las características.

    Retorna
    -------
    dict
        11 características con el patrón ``{prefix}_{metrica}``.

    Referencias
    -----------
    - scipy.stats.skew:
      https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html
    - scipy.stats.kurtosis:
      https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosis.html
    - Jaeger et al. (2013): "Automatic screening for tuberculosis in chest
      radiographs: a survey."
      https://pmc.ncbi.nlm.nih.gov/articles/PMC3636475/
    """
    x = region.astype(np.float64).ravel()
    p10, p25, p50, p75, p90 = np.percentile(x, ADVANCED_PERCENTILES)
    mean = float(np.mean(x))
    std = float(np.std(x))
    return {
        f"{prefix}_median": float(p50),
        f"{prefix}_variance": float(np.var(x)),
        f"{prefix}_skewness": float(skew(x, bias=False)),
        f"{prefix}_kurtosis": float(kurtosis(x, fisher=True, bias=False)),
        f"{prefix}_p10": float(p10),
        f"{prefix}_p25": float(p25),
        f"{prefix}_p75": float(p75),
        f"{prefix}_p90": float(p90),
        f"{prefix}_iqr": float(p75 - p25),
        f"{prefix}_mad": float(np.median(np.abs(x - p50))),
        f"{prefix}_cv": float(std / (abs(mean) + 1e-9)),
    }

def feat_intensity_advanced(img: np.ndarray) -> dict:
    """
    Extrae estadísticas de intensidad avanzadas globales y regionales.

    Calcula 11 medidas sobre la imagen completa y sobre las seis regiones
    anatómicas de :data:`REGIONS`. El objetivo es representar diferencias en
    la forma, dispersión y colas del histograma de intensidad que no quedan
    descritas únicamente por media, desviación, contraste y entropía.

    Parámetros
    ----------
    img : np.ndarray
        Imagen preprocesada (uint8, shape 256×256).

    Retorna
    -------
    dict
        77 características: 11 globales + 11 × 6 regiones.

    Referencias
    -----------
    - Jaeger et al. (2013): "Automatic screening for tuberculosis in chest
      radiographs: a survey."
      https://pmc.ncbi.nlm.nih.gov/articles/PMC3636475/
    - SciPy Statistics:
      https://docs.scipy.org/doc/scipy/reference/stats.html
    """
    feats = _advanced_stats(img, "adv_global")
    for name, (x, y, w, h) in REGIONS.items():
        feats.update(_advanced_stats(img[y:y+h, x:x+w], f"adv_{name}"))
    return feats

# ---------------------------------------------------------------------------
# Extractor 9: GLCM multiescala y regional
# ---------------------------------------------------------------------------

def _glcm_at_distances(region: np.ndarray, distances: list[int], prefix: str) -> dict:
    """Calcula propiedades GLCM por distancia y resume los cuatro ángulos."""
    img64 = (region // 4).astype(np.uint8)
    matrix = graycomatrix(
        img64, distances=distances, angles=GLCM_ANGLES, levels=64,
        symmetric=True, normed=True
    )
    feats = {}
    for prop in GLCM_PROPS:
        values = graycoprops(matrix, prop)
        for i, distance in enumerate(distances):
            vals = values[i, :]
            key = f"{prefix}_d{distance}_{prop}"
            feats[f"{key}_mean"] = float(np.mean(vals))
            feats[f"{key}_std"] = float(np.std(vals))
    return feats


def feat_glcm_multiscale_regional(img: np.ndarray) -> dict:
    """
    Extrae características GLCM multiescala y regionales.

    El GLCM original del proyecto utiliza distancia 1 sobre la imagen completa.
    Esta extensión añade distancias 2, 4 y 8 a nivel global y calcula las cuatro
    distancias 1, 2, 4 y 8 dentro de cada una de las seis regiones anatómicas.
    Así se incorpora simultáneamente escala espacial y localización.

    La distancia 1 global no se repite porque ya está representada por
    :func:`feat_glcm`.

    Parámetros
    ----------
    img : np.ndarray
        Imagen preprocesada (uint8, shape 256×256).

    Retorna
    -------
    dict
        270 características: 30 globales adicionales + 40 × 6 regiones.

    Referencias
    -----------
    - skimage.feature.graycomatrix:
      https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.graycomatrix
    - Chauhan et al. (2014): "Role of Gist and PHOG Features in
      Computer-Aided Diagnosis of Tuberculosis without Segmentation."
      https://pmc.ncbi.nlm.nih.gov/articles/PMC4229306/
    """
    feats = _glcm_at_distances(img, [2, 4, 8], "glcm_ms_global")
    for name, (x, y, w, h) in REGIONS.items():
        feats.update(_glcm_at_distances(
            img[y:y+h, x:x+w], GLCM_DISTANCES_MS, f"glcm_ms_{name}"
        ))
    return feats

# ---------------------------------------------------------------------------
# Extractor 10: LBP multiescala
# ---------------------------------------------------------------------------

def feat_lbp_multiscale(img: np.ndarray) -> dict:
    """
    Añade Local Binary Patterns en escalas complementarias.

    El extractor original ya representa P=24, R=3. Para evitar duplicación,
    esta función agrega únicamente (P,R)=(8,1) y (16,2), capturando
    microtexturas más finas y de escala intermedia. Se mantiene el método
    ``uniform`` y el histograma normalizado.

    Parámetros
    ----------
    img : np.ndarray
        Imagen preprocesada (uint8, shape 256×256).

    Retorna
    -------
    dict
        28 características: 10 bins para (8,1) y 18 para (16,2).

    Referencias
    -----------
    - Ojala et al. (2002): "Multiresolution Gray-Scale and Rotation Invariant
      Texture Classification with Local Binary Patterns."
    - Jaeger et al. (2013):
      https://pmc.ncbi.nlm.nih.gov/articles/PMC3636475/
    - skimage.feature.local_binary_pattern:
      https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.local_binary_pattern
    """
    feats = {}
    for p, r in LBP_EXTRA_SCALES:
        lbp_img = local_binary_pattern(img, P=p, R=r, method="uniform")
        bins = p + 2
        hist, _ = np.histogram(
            lbp_img.ravel(), bins=bins, range=(0, bins), density=True
        )
        for i, value in enumerate(hist):
            feats[f"lbp_ms_p{p}_r{int(r)}_{i}"] = float(value)
    return feats

# ---------------------------------------------------------------------------
# Extractor 11: Asimetría bilateral
# ---------------------------------------------------------------------------

def feat_bilateral_asymmetry(img: np.ndarray) -> dict:
    """
    Cuantifica diferencias relativas entre regiones izquierda y derecha.

    Compara SI-SD, MI-MD e II-ID para media, desviación, contraste y entropía.
    La asimetría se expresa como ``|L-R| / (|L|+|R|+eps)``, convirtiendo la
    comparación entre ambos hemitórax en variables explícitas y normalizadas.

    Parámetros
    ----------
    img : np.ndarray
        Imagen preprocesada (uint8, shape 256×256).

    Retorna
    -------
    dict
        12 características: 3 pares anatómicos × 4 métricas.

    Referencias
    -----------
    - Jaeger et al. (2013): revisión de CAD para TB y análisis de campos
      pulmonares. https://pmc.ncbi.nlm.nih.gov/articles/PMC3636475/
    """
    regional = feat_intensity_regional(img)
    pairs = {
        "superior": ("SI", "SD"),
        "middle": ("MI", "MD"),
        "inferior": ("II", "ID"),
    }
    feats = {}
    for pair_name, (left, right) in pairs.items():
        for metric in ("mean", "std", "contrast", "entropy"):
            l = regional[f"{left}_{metric}"]
            r = regional[f"{right}_{metric}"]
            feats[f"asym_{pair_name}_{metric}"] = float(
                abs(l - r) / (abs(l) + abs(r) + 1e-9)
            )
    return feats

# ---------------------------------------------------------------------------
# Extractor 12: GIST espacial multiescala
# ---------------------------------------------------------------------------

def feat_gist(img: np.ndarray) -> dict:
    """
    Extrae una representación GIST basada en energía Gabor espacial.

    Se filtra la radiografía a cuatro frecuencias y ocho orientaciones. La
    magnitud de cada respuesta se agrega sobre una cuadrícula 4×4, produciendo
    una descripción compacta del contexto global, las frecuencias dominantes,
    sus orientaciones y su localización espacial.

    Esta implementación sigue el principio de GIST utilizado en literatura de
    CAD de TB, pero está implementada con ``skimage.filters.gabor`` para
    mantener coherencia con las dependencias del proyecto.

    Parámetros
    ----------
    img : np.ndarray
        Imagen preprocesada (uint8, shape 256×256).

    Retorna
    -------
    dict
        512 características: 4 frecuencias × 8 orientaciones × 16 bloques.

    Referencias
    -----------
    - Chauhan et al. (2014): "Role of Gist and PHOG Features in
      Computer-Aided Diagnosis of Tuberculosis without Segmentation."
      https://pmc.ncbi.nlm.nih.gov/articles/PMC4229306/
    - skimage.filters.gabor:
      https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.gabor
    """
    image = img.astype(np.float64) / 255.0
    h, w = img.shape
    rows, cols = GIST_GRID
    feats = {}

    for freq in GIST_FREQUENCIES:
        for theta in GIST_ORIENTATIONS:
            real, imag = skimage_gabor(image, frequency=freq, theta=theta)
            magnitude = np.sqrt(real ** 2 + imag ** 2)
            block_idx = 0
            for row in range(rows):
                y0, y1 = row * h // rows, (row + 1) * h // rows
                for col in range(cols):
                    x0, x1 = col * w // cols, (col + 1) * w // cols
                    angle10 = int(round(np.degrees(theta) * 10))
                    feats[f"gist_f{freq:.2f}_t{angle10}_b{block_idx}"] = float(
                        np.mean(magnitude[y0:y1, x0:x1])
                    )
                    block_idx += 1
    return feats

# ---------------------------------------------------------------------------
# Función principal de extracción por imagen
# ---------------------------------------------------------------------------

def extract_features(image_path: str, label: int, class_name: str) -> Optional[dict]:
    """
    Ejecuta el pipeline completo de preprocesamiento y extracción de
    características para una sola imagen.

    Parámetros
    ----------
    image_path : str
        Ruta al archivo de imagen original.
    label : int
        Etiqueta numérica de la clase.
    class_name : str
        Nombre de la clase (``'health'``, ``'sick'`` o ``'tb'``).

    Retorna
    -------
    dict o None
        Diccionario con metadatos y las 4248 características extraídas,
        o ``None`` si la imagen no puede cargarse o procesarse.
    """
    img = preprocess(image_path)
    if img is None:
        return None

    row: dict = {"image_path": image_path, "class_name": class_name, "label": label}
    row.update(feat_intensity_global(img))
    row.update(feat_intensity_regional(img))
    row.update(feat_glcm(img))
    row.update(feat_lbp(img))
    row.update(feat_hog(img))
    row.update(feat_gabor(img))
    row.update(feat_phog(img))
    row.update(feat_intensity_advanced(img))
    row.update(feat_glcm_multiscale_regional(img))
    row.update(feat_lbp_multiscale(img))
    row.update(feat_bilateral_asymmetry(img))
    row.update(feat_gist(img))
    return row


# ---------------------------------------------------------------------------
# Extracción masiva sobre el DataFrame base
# ---------------------------------------------------------------------------
def extract_all_features(
    df_base: pd.DataFrame,
    sample: bool = False,
    sample_size: int = 50,
    random_state: int = 42,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Procesa todas las imágenes del DataFrame base y construye la matriz
    de características.

    Parámetros
    ----------
    df_base : pd.DataFrame
        DataFrame con columnas ``image_path``, ``label`` y ``class_name``.
    sample : bool
        Si es ``True``, procesa solo ``sample_size`` imágenes por clase.
        Útil para pruebas rápidas durante el desarrollo.
    sample_size : int
        Número máximo de imágenes por clase en modo muestra.
    random_state : int
        Semilla para reproducibilidad del muestreo.

    Retorna
    -------
    features_df : pd.DataFrame
        DataFrame con una fila por imagen y las 4248 + 3 columnas (metadatos + features).
    errors : list de str
        Lista de rutas de imágenes que no pudieron procesarse.

    Referencias
    -----------
    - tqdm: https://tqdm.github.io/
    """
    if sample:
        df_proc = pd.concat([
            df_base[df_base["class_name"] == clase].sample(
                min(sample_size, len(df_base[df_base["class_name"] == clase])),
                random_state=random_state
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
        result = extract_features(row["image_path"], row["label"], row["class_name"])
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
# Verificación de integridad
# ---------------------------------------------------------------------------

def verify_extractors(df_base: pd.DataFrame) -> None:
    """
    Verifica el número de características producido por cada extractor.

    Ejecuta todos los extractores sobre una imagen de muestra de cada clase y
    comprueba tanto las dimensiones individuales como el total esperado.

    Parámetros
    ----------
    df_base : pd.DataFrame
        DataFrame base con al menos una imagen por clase.

    Raises
    ------
    AssertionError
        Si algún extractor no produce la cantidad esperada.
    """
    from .data import LABEL_MAP

    expected = {
        "global": 4, "regional": 24, "GLCM": 10, "LBP": 26,
        "HOG": 1764, "Gabor": 32, "PHOG": 1440,
        "Advanced": 77, "GLCM_MS": 270, "LBP_MS": 28,
        "Asymmetry": 12, "GIST": 512,
        "TOTAL": 4199,
    }

    for class_name in LABEL_MAP:
        sample = df_base[df_base["class_name"] == class_name]
        if sample.empty:
            continue
        img = preprocess(sample.iloc[0]["image_path"])
        assert img is not None
        assert img.shape == (256, 256)

        groups = {
            "global": feat_intensity_global(img),
            "regional": feat_intensity_regional(img),
            "GLCM": feat_glcm(img),
            "LBP": feat_lbp(img),
            "HOG": feat_hog(img),
            "Gabor": feat_gabor(img),
            "PHOG": feat_phog(img),
            "Advanced": feat_intensity_advanced(img),
            "GLCM_MS": feat_glcm_multiscale_regional(img),
            "LBP_MS": feat_lbp_multiscale(img),
            "Asymmetry": feat_bilateral_asymmetry(img),
            "GIST": feat_gist(img),
        }
        total = sum(len(v) for v in groups.values())
        print(class_name, " | ".join(f"{k}={len(v)}" for k, v in groups.items()), f"TOTAL={total}")
        for key, values in groups.items():
            assert len(values) == expected[key], f"{key}: esperado {expected[key]}, obtenido {len(values)}"
        assert total == expected["TOTAL"], f"TOTAL: esperado {expected['TOTAL']}, obtenido {total}"

    print(f"Verificación completada. Total esperado: {expected['TOTAL']} features.")


# ---------------------------------------------------------------------------
# Resumen de grupos de características
# ---------------------------------------------------------------------------

def get_feature_groups(features_df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Clasifica las columnas del DataFrame de características en grupos
    según el extractor que las generó.

    Parámetros
    ----------
    features_df : pd.DataFrame
        DataFrame con la matriz de características.

    Retorna
    -------
    dict
        Diccionario con claves ``'Intensidad global'``, ``'Intensidad regional'``,
        ``'GLCM'``, ``'LBP'`` y ``'HOG'``, y listas de nombres de columna.
    """
    meta_cols = {"image_path", "class_name", "label"}
    feat_cols = [c for c in features_df.columns if c not in meta_cols]

    return {
        "Intensidad global": [c for c in feat_cols if c in {"mean", "std", "contrast", "entropy"}],
        "Intensidad regional": [c for c in feat_cols if any(c.startswith(r + "_") for r in REGIONS)],
        "GLCM": [c for c in feat_cols if c.startswith("glcm_") and not c.startswith("glcm_ms_")],
        "LBP": [c for c in feat_cols if c.startswith("lbp_") and not c.startswith("lbp_ms_")],
        "HOG": [c for c in feat_cols if c.startswith("hog_")],
        "Gabor": [c for c in feat_cols if c.startswith("gabor_")],
        "PHOG": [c for c in feat_cols if c.startswith("phog_")],
        "Estadística avanzada": [c for c in feat_cols if c.startswith("adv_")],
        "GLCM multiescala/regional": [c for c in feat_cols if c.startswith("glcm_ms_")],
        "LBP multiescala": [c for c in feat_cols if c.startswith("lbp_ms_")],
        "Asimetría bilateral": [c for c in feat_cols if c.startswith("asym_")],
        "GIST": [c for c in feat_cols if c.startswith("gist_")],
    }

def print_feature_summary(features_df: pd.DataFrame) -> None:
    """
    Imprime un resumen de la distribución de características por grupo
    y verifica la integridad del DataFrame (nulos e infinitos).

    Parámetros
    ----------
    features_df : pd.DataFrame
        DataFrame con la matriz de características completa.
    """
    meta_cols = ["image_path", "class_name", "label"]
    feat_cols = [c for c in features_df.columns if c not in meta_cols]
    grupos = get_feature_groups(features_df)

    print(f"{'Filas':.<30} {len(features_df)}")
    print(f"{'Features':.<30} {len(feat_cols)}")
    print()
    print("Distribución de clases:")
    print(features_df["class_name"].value_counts().to_string())
    print()

    nulls = features_df[feat_cols].isnull().sum().sum()
    infs  = np.isinf(features_df[feat_cols].values).sum()
    print(f"{'Valores nulos':.<30} {nulls}")
    print(f"{'Valores infinitos':.<30} {infs}")
    print()
    print("Features por grupo:")
    for nombre, cols in grupos.items():
        print(f"  {nombre:.<28} {len(cols):>5}")
    print(f"  {'TOTAL':.<28} {sum(len(v) for v in grupos.values()):>5}")
