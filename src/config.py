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

# Protocolo de modelado v2.
CV_OUTER = 5
CV_INNER = 3
RANDOM_STATE = 42
TARGET_SENSITIVITY = 0.90
