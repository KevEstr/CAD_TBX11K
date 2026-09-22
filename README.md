# cad-tbx11k

Pipeline de análisis y extracción de características sobre el dataset TBX11K para el proyecto de diagnóstico asistido de tuberculosis (CAD).

## Estructura

```
cad-tbx11k/
├── notebooks/
│   ├── 01_Carga_Datos.ipynb                 ← descarga desde Kaggle, mapeo del dataset, DataFrame base
│   ├── 02_Analisis_Exploratorio.ipynb       ← visualización de ejemplos, histogramas, métricas de intensidad
│   ├── 03_Preprocesamiento.ipynb            ← BF + CLAHE, comparativas visuales, efecto en histogramas
│   ├── 04_Analisis_Regional.ipynb           ← división en 6 zonas anatómicas, heatmap, distribuciones
│   └── 05_Extraccion_Caracteristicas.ipynb  ← GLCM, LBP, HOG, intensidad global/regional → CSV
├── requirements.txt
├── .gitignore
└── README.md
```

Cada notebook es independiente — se puede correr sin haber ejecutado los anteriores.
El `DATA_ROOT` en cada uno apunta a la caché local de kagglehub.

## Dataset

[TBX11K](https://mmcheng.net/tb/) — 8.400 radiografías etiquetadas de tórax (train+val).
Tres clases: `health (0)`, `sick (1)`, `tb (2)`.

Descarga automática desde Kaggle en `01_Carga_Datos.ipynb`.
Requiere `kaggle.json` en la raíz del proyecto.

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
python -m ipykernel install --user --name=cad-tbx11k
```

## Orden de ejecución

```
01 → 02 → 03 → 04 → 05
```

El notebook 05 genera `features_tbx11k.csv` en la raíz del proyecto.
Ese CSV es la entrada para la siguiente etapa: entrenamiento de modelos ML.

## Referencias

- `TB_detection` (https://github.com/Krishni1307/TB_detection.git) — pipeline de clasificación sobre TBX11K con PyTorch. De ahí se tomó la forma de recorrer las carpetas del dataset y construir el DataFrame base con rutas y etiquetas.
- `tbx11k-ensemble` (https://github.com/behera116/tbx11k-ensemble.git) — ensemble de modelos con preprocesamiento offline. De ahí vienen los parámetros del Bilateral Filter y CLAHE aplicados a radiografías de tórax.
- `CAD_Analisis_Exploratorio` — análisis exploratorio suministrado por el semillero de investigación sobre un dataset binario de TB. De ahí se tomó la idea de registrar dimensiones, calcular métricas de intensidad y dividir la imagen en regiones anatómicas.
