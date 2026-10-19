"""
config.py
=========
Configuración central del pipeline CAD-TBX11K.

Todas las variables de muestreo del proyecto se definen aquí UNA sola vez.
Regla del proyecto: **un nombre = un valor en todos los notebooks**. Si necesitas
cambiar cuántas imágenes se usan, cámbialo aquí y no dentro de los notebooks.

Hay dos niveles:

1. ``N_MUESTRA`` fija cuántas imágenes por clase entran al DataFrame base; es el
   "techo" de datos disponible para todo el pipeline (lo aplica el notebook 01).
2. Las demás variables definen cuántas imágenes por clase consume cada análisis.
   Nunca deben superar el techo: los notebooks las acotan con ``min(...)``.

Ejemplo
-------
>>> from src.config import N_MUESTRA, N_INTENSIDAD
>>> N_MUESTRA          # techo de datos disponible
800
"""

# ---------------------------------------------------------------------------
# Nivel 1 — Techo de datos (lo aplica el notebook 01 al construir df_base)
# ---------------------------------------------------------------------------

# Número de imágenes por clase cargadas en df_base.
#   None -> dataset completo (~8.400 imágenes).
#   int  -> esa cantidad por clase (modo desarrollo, más rápido).
N_MUESTRA = 200

# ---------------------------------------------------------------------------
# Nivel 2 — Muestra consumida por cada análisis (nunca mayor que N_MUESTRA)
# ---------------------------------------------------------------------------

# Histogramas y métricas de intensidad (notebooks 02 y 03).
N_INTENSIDAD = 200

# Tabla de seguimiento antes/después del preprocesamiento (notebook 03).
N_SEGUIMIENTO = 200

# Análisis regional por región anatómica (notebook 04).
N_REGIONAL = 200

# Extracción de características para el modelo (notebook 05).
#   None -> dataset completo: TODAS las imágenes de df_base (recomendado).
#   int  -> esa cantidad por clase (modo desarrollo, submuestreo rápido).
# No se balancean las clases aquí: el desbalanceo real del dataset se
# conserva y se aborda en el notebook 07 (class_weight / SMOTE).
N_MODELO = None
