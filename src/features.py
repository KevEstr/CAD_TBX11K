"""
cad_tbx11k.features
====================
Módulo de extracción de características del pipeline CAD-TBX11K.

Convierte cada radiografía preprocesada en un vector numérico de 1828 características
agrupadas en cinco grupos:

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
| **Total**            |                                           | **1828** |
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
from scipy.stats import entropy as scipy_entropy
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
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
        Diccionario con los metadatos de la imagen y las 1828 características,
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
        DataFrame con una fila por imagen y las 1828 + 3 columnas (metadatos + features).
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
    Verifica que cada extractor produce el número correcto de características
    sobre una imagen de muestra de cada clase.

    Parámetros
    ----------
    df_base : pd.DataFrame
        DataFrame base con al menos una imagen por clase.

    Raises
    ------
    AssertionError
        Si algún extractor no produce la cantidad esperada de características.
    """
    from .data import LABEL_MAP

    expected = {
        "global": 4,
        "regional": 24,
        "GLCM": 10,
        "LBP": 26,
        "HOG": 1764,
        "TOTAL": 1828,
    }

    for class_name in LABEL_MAP:
        sample = df_base[df_base["class_name"] == class_name]
        if sample.empty:
            continue
        path = sample.iloc[0]["image_path"]
        img = preprocess(path)
        assert img is not None, f"No se pudo preprocesar la imagen de {class_name}."
        assert img.shape == (256, 256), f"Shape inesperado: {img.shape}."

        g  = feat_intensity_global(img)
        ri = feat_intensity_regional(img)
        gl = feat_glcm(img)
        lb = feat_lbp(img)
        hg = feat_hog(img)
        total = len(g) + len(ri) + len(gl) + len(lb) + len(hg)

        print(
            f"{class_name:>6} | global={len(g)}  regional={len(ri)}"
            f"  GLCM={len(gl)}  LBP={len(lb)}  HOG={len(hg)}  TOTAL={total}"
        )
        assert len(g)  == expected["global"],   f"global: esperado {expected['global']}, obtenido {len(g)}"
        assert len(ri) == expected["regional"],  f"regional: esperado {expected['regional']}, obtenido {len(ri)}"
        assert len(gl) == expected["GLCM"],      f"GLCM: esperado {expected['GLCM']}, obtenido {len(gl)}"
        assert len(lb) == expected["LBP"],       f"LBP: esperado {expected['LBP']}, obtenido {len(lb)}"
        assert len(hg) == expected["HOG"],       f"HOG: esperado {expected['HOG']}, obtenido {len(hg)}"
        assert total   == expected["TOTAL"],     f"TOTAL: esperado {expected['TOTAL']}, obtenido {total}"

    print(f"\nEsperado: global={expected['global']}  regional={expected['regional']}"
          f"  GLCM={expected['GLCM']}  LBP={expected['LBP']}"
          f"  HOG={expected['HOG']}  TOTAL={expected['TOTAL']}")
    print("Verificación completada sin errores.")


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
        "Intensidad global":   [c for c in feat_cols if c in {"mean", "std", "contrast", "entropy"}],
        "Intensidad regional": [c for c in feat_cols if any(c.startswith(r) for r in REGIONS)],
        "GLCM":                [c for c in feat_cols if c.startswith("glcm_")],
        "LBP":                 [c for c in feat_cols if c.startswith("lbp_")],
        "HOG":                 [c for c in feat_cols if c.startswith("hog_")],
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
