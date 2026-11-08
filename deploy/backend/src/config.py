"""
config.py
=========
Configuración central del pipeline CAD-TBX11K.

Todas las variables de muestreo del proyecto se definen aquí UNA sola vez.
Regla del proyecto: **un nombre = un valor en todos los notebooks**. Si
necesitas cambiar cuántas imágenes se usan, cámbialo aquí y no dentro de los
notebooks.

Hay dos niveles:

1. ``N_MUESTRA`` fija cuántas imágenes por clase entran al DataFrame base; es
   el "techo" de datos disponible para todo el pipeline (lo aplica el
   notebook 01).
2. Las demás variables definen cuántas imágenes por clase consume cada
   análisis. Nunca deben superar el techo: los notebooks las acotan con
   ``min(...)``.
"""

# ---------------------------------------------------------------------------
# Nivel 1 — Techo de datos (lo aplica el notebook 01)
# ---------------------------------------------------------------------------

# Número de imágenes por clase cargadas en df_base.
#   None -> dataset completo (8.400 imágenes etiquetadas).
#   int  -> esa cantidad por clase (modo desarrollo, más rápido).
N_MUESTRA = None

# ---------------------------------------------------------------------------
# Nivel 2 — Muestra consumida por cada análisis
# ---------------------------------------------------------------------------

# Histogramas y métricas de intensidad (notebooks 02 y 03).
N_INTENSIDAD = 200

# Tabla de seguimiento antes/después del preprocesamiento (notebook 03).
N_SEGUIMIENTO = 200

# Análisis regional por región anatómica (notebook 04).
N_REGIONAL = 200

# Extracción de características para el modelo (notebook 05).
#   None -> dataset completo: 8.256 imágenes únicas tras deduplicación exacta.
#   int  -> esa cantidad por clase (modo desarrollo, submuestreo rápido).
# No se balancean las clases aquí: el desbalanceo real del dataset se
# conserva y se aborda en el notebook 07 (class_weight / SMOTE).
N_MODELO = None

# Normalización de intensidad dentro de la máscara pulmonar.
#   "zscore" -> media 0 y desviación 1 dentro del pulmón, recortado a ±3σ.
#   "none"   -> intensidad original.
INTENSITY_NORM = "zscore"

# Modo de textura del modelo principal.
#   "inner_lung" -> textura y HOG dentro de la máscara erosionada, sin borde.
#   "full_lung"  -> textura sobre la máscara completa.
FEATURE_MODE = "inner_lung"
INNER_ERODE_PX = 8

# Protocolo de modelado.
CV_OUTER = 5
CV_INNER = 3
RANDOM_STATE = 42
TARGET_SENSITIVITY = 0.90
