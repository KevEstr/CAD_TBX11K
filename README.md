# cad-tbx11k

Pipeline de análisis exploratorio, preprocesamiento y extracción de características sobre el dataset **TBX11K**, como parte del proyecto de diagnóstico asistido por computador (CAD) para tuberculosis a partir de radiografías de tórax.

---

## Estructura del repositorio

```text
cad-tbx11k/
│
├── src/                             ← Módulo Python reutilizable (framework)
│   ├── __init__.py                  ← Punto de entrada único del paquete
│   ├── data.py                      ← Carga del dataset y construcción del DataFrame base
│   ├── preprocessing.py             ← Cadena de preprocesamiento (BF + CLAHE)
│   ├── features.py                  ← Extractores de características (GLCM, LBP, HOG, intensidad)
│   └── visualization.py             ← Figuras de publicación para cada etapa del análisis
│
├── notebooks/
│   ├── 01_Carga_Datos.ipynb
│   ├── 02_Analisis_Exploratorio.ipynb
│   ├── 03_Preprocesamiento.ipynb
│   ├── 04_Analisis_Regional.ipynb
│   └── 05_Extraccion_Caracteristicas.ipynb
│
├── data/                            ← Dataset y archivos generados (excluido de Git)
│   ├── TBX11K/                      ← Dataset descargado manualmente desde Kaggle
│   │   └── imgs/
│   │       ├── health/
│   │       ├── sick/
│   │       └── tb/
│   ├── df_base.csv                  ← DataFrame base generado en notebook 01
│   └── features_tbx11k.csv         ← Matriz de características generada en notebook 05
│
├── kaggle.json                      ← Credenciales Kaggle (excluido de Git)
├── requirements.txt
├── .gitignore
└── README.md
```

Los notebooks importan directamente del módulo `src`. Cualquier cambio en un parámetro del framework (por ejemplo, `CLAHE_CLIP_LIMIT`) se propaga de forma consistente a todo el pipeline.

---

## Dataset

El proyecto utiliza **TBX11K**, un dataset de radiografías de tórax desarrollado para investigación en detección de tuberculosis.

**Fuente:** [TBX11K — mmcheng.net](https://mmcheng.net/tb/)  
**Kaggle:** [usmanshams/tbx-11](https://www.kaggle.com/datasets/usmanshams/tbx-11)

El subconjunto utilizado corresponde a **8.400 radiografías etiquetadas** de los conjuntos de entrenamiento y validación, organizadas en tres clases:

| Clase    | Etiqueta | Descripción                                         |
|----------|:--------:|-----------------------------------------------------|
| `health` |    0     | Radiografía de persona sana                         |
| `sick`   |    1     | Persona enferma, no diagnosticada con TB            |
| `tb`     |    2     | Tuberculosis confirmada                             |

### Descarga manual del dataset

Dado que la red institucional puede bloquear la descarga automática via `kagglehub`, el dataset debe descargarse manualmente:

1. Entrar a [kaggle.com/datasets/usmanshams/tbx-11](https://www.kaggle.com/datasets/usmanshams/tbx-11) desde el navegador.
2. Hacer clic en **Download**.
3. Descomprimir el ZIP dentro de `data/` de modo que quede `data/TBX11K/imgs/...`.
4. En el notebook 01, definir la ruta manualmente:

```python
DATA_ROOT = r"C:\ruta\al\proyecto\data\TBX11K"
```

---

## Setup

### 1. Crear el entorno virtual

```bash
python -m venv venv
```

### 2. Activar el entorno

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Registrar el kernel de Jupyter

```bash
python -m ipykernel install --user --name=cad-tbx11k
```

Seleccionar el kernel `cad-tbx11k` desde Jupyter Notebook o Visual Studio Code.

### 5. Credenciales de Kaggle (opcional)

Si la red permite conexión a Kaggle, descargar `kaggle.json` desde **kaggle.com → Settings → API → Create New Token** y colocarlo en la raíz del proyecto. El archivo está excluido por `.gitignore`.

---

## Orden de ejecución

```text
01 → 02 → 03 → 04 → 05
```

Cada notebook guarda sus salidas en `data/` para ser reutilizadas por los siguientes:

| Notebook | Genera |
|---|---|
| 01 | `data/df_base.csv` |
| 05 | `data/features_tbx11k.csv` |

---

## Referencia de funciones del módulo `src`

Todas las funciones se importan desde `src`:

```python
from src.data import build_base_dataframe
from src.preprocessing import preprocess
from src.features import extract_all_features
from src.visualization import set_publication_style
```

---

### `src.data` — Carga de datos

| Función / Constante | Descripción | Retorna |
|---|---|---|
| `LABEL_MAP` | `{'health': 0, 'sick': 1, 'tb': 2}` | `dict` |
| `PALETTE_ACADEMIC` | Colores por clase para visualización | `dict` |
| `VALID_EXTS` | Extensiones de imagen aceptadas | `tuple` |
| `setup_kaggle_credentials(kaggle_json=None)` | Configura `~/.kaggle/kaggle.json`. Detecta Google Colab automáticamente. | `None` |
| `download_dataset()` | Descarga TBX11K via `kagglehub`. Usa caché local si ya fue descargado. | `str` — ruta de descarga |
| `get_data_root(download_path)` | Verifica y retorna la ruta al directorio `TBX11K/`. | `str` |
| `map_dataset_structure(data_root)` | Mapea `imgs/` y cuenta imágenes por carpeta/subcarpeta. | `pd.DataFrame` — columnas `carpeta`, `subcarpeta`, `imagenes` |
| `build_base_dataframe(data_root)` | Carga el subconjunto etiquetado y construye el DataFrame base con metadatos por imagen. | `pd.DataFrame` — columnas `image_path`, `class_name`, `label`, `ancho`, `alto`, `modo` |

**Ejemplo:**

```python
from src.data import build_base_dataframe, LABEL_MAP
from pathlib import Path

DATA_ROOT = r"C:\ruta\al\proyecto\data\TBX11K"
df_base   = build_base_dataframe(DATA_ROOT)

# Cargar desde CSV si ya fue generado
df_base = pd.read_csv(Path('..') / 'data' / 'df_base.csv')
```

---

### `src.preprocessing` — Preprocesamiento

Pipeline: `imagen original → escala de grises → resize 256×256 → Bilateral Filter → CLAHE`

| Función / Constante | Descripción | Retorna |
|---|---|---|
| `IMG_SIZE` | Tamaño objetivo `(256, 256)` | `tuple` |
| `BF_D` | Diámetro del vecindario del Bilateral Filter (`9`) | `int` |
| `BF_SIGMA_COLOR` | Sigma de color del Bilateral Filter (`80.0`) | `float` |
| `BF_SIGMA_SPACE` | Sigma espacial del Bilateral Filter (`80.0`) | `float` |
| `CLAHE_CLIP_LIMIT` | Límite de recorte del histograma CLAHE (`40.0`) | `float` |
| `CLAHE_TILE_GRID` | Tamaño de la cuadrícula CLAHE `(8, 8)` | `tuple` |
| `load_grayscale(path)` | Carga la imagen en escala de grises. | `np.ndarray` o `None` |
| `resize_image(img, size=IMG_SIZE)` | Redimensiona con interpolación de área (`INTER_AREA`). | `np.ndarray` |
| `apply_bilateral_filter(img, ...)` | Aplica el Filtro Bilateral preservando bordes. | `np.ndarray` |
| `apply_clahe(img, ...)` | Aplica CLAHE para realce de contraste local. | `np.ndarray` |
| `preprocess(path)` | **Pipeline completo.** Ejecuta los cuatro pasos sobre una imagen. | `np.ndarray` o `None` |
| `preprocess_steps(path)` | Ejecuta el pipeline y retorna cada estado intermedio. | `dict` con claves `'original'`, `'bilateral'`, `'clahe'` |

**Ejemplo:**

```python
from src.preprocessing import preprocess, preprocess_steps

img   = preprocess('ruta/imagen.png')        # np.ndarray (256, 256)
pasos = preprocess_steps('ruta/imagen.png')
img_original  = pasos['original']
img_bilateral = pasos['bilateral']
img_clahe     = pasos['clahe']
```

---

### `src.features` — Extracción de características

| Función / Constante | Descripción | Retorna |
|---|---|---|
| `REGIONS` | Definición de las 6 regiones anatómicas `{nombre: (x, y, w, h)}` | `dict` |
| `GLCM_PROPS` | Propiedades GLCM calculadas | `list` |
| `LBP_P` | Puntos de muestreo LBP (`24`) | `int` |
| `LBP_R` | Radio LBP (`3.0`) | `float` |
| `LBP_BINS` | Bins del histograma LBP (`26`) | `int` |
| `HOG_PIXELS_PER_CELL` | Tamaño de celda HOG `(32, 32)` | `tuple` |
| `feat_intensity_global(img)` | Calcula `mean`, `std`, `contrast`, `entropy` sobre la imagen completa. | `dict` — 4 entradas |
| `feat_intensity_regional(img)` | Calcula las 4 métricas en cada una de las 6 regiones anatómicas. | `dict` — 24 entradas |
| `feat_glcm(img)` | GLCM a 4 ángulos, 64 niveles, 5 propiedades × (mean + std). | `dict` — 10 entradas |
| `feat_lbp(img)` | Histograma LBP uniform normalizado (P=24, R=3). | `dict` — 26 entradas |
| `feat_hog(img)` | HOG con celdas 32×32, bloques 2×2, normalización L2-Hys. | `dict` — 1764 entradas |
| `extract_features(image_path, label, class_name)` | Pipeline completo para una sola imagen. | `dict` — 1828 + 3 entradas, o `None` |
| `extract_all_features(df_base, sample, sample_size, random_state)` | Extracción masiva sobre el DataFrame base con barra de progreso `tqdm`. | `(pd.DataFrame, list[str])` — features y errores |
| `verify_extractors(df_base)` | Verifica la dimensionalidad de cada extractor con `assert`. | `None` |
| `get_feature_groups(features_df)` | Clasifica columnas por grupo de extractor. | `dict` |
| `print_feature_summary(features_df)` | Imprime recuento, distribución de clases, nulos e infinitos. | `None` |

**Dimensionalidad del vector de características:**

| Grupo | Features |
|---|:---:|
| Intensidad global | 4 |
| Intensidad regional (6 zonas × 4 métricas) | 24 |
| GLCM (5 propiedades × 2 estadísticos) | 10 |
| LBP (P=24, uniform) | 26 |
| HOG (celdas 32×32, bloques 2×2, 9 orientaciones) | 1764 |
| **Total** | **1828** |

**Ejemplo:**

```python
from src.features import extract_all_features, verify_extractors

verify_extractors(df_base)

N_POR_CLASE = df_base['class_name'].value_counts().min()
features_df, errores = extract_all_features(df_base, sample=True, sample_size=N_POR_CLASE)

# Extracción completa
features_df, errores = extract_all_features(df_base, sample=False)
features_df.to_csv('../data/features_tbx11k.csv', index=False)
```

---

### `src.visualization` — Visualización

Todas las figuras usan estilo de publicación científica (`seaborn-v0_8-whitegrid`, DPI 150 en pantalla, 300 para guardar). Cada función incluye una línea comentada para guardar:

```python
# fig.savefig(save_path, dpi=300, bbox_inches="tight")  # uncomment to save
```

Las funciones con muestras ajustan automáticamente el tamaño si hay menos imágenes disponibles que `n` solicitado.

| Función | Notebook | Descripción |
|---|:---:|---|
| `set_publication_style()` | todos | Configura `rcParams` globales de Matplotlib para estilo de artículo. |
| `plot_dataset_distribution(resumen_df, df_base, save_path)` | 01 | Panel de barras: dataset completo y subconjunto etiquetado. Leyendas debajo de las barras. |
| `plot_image_metadata(df_base, save_path)` | 01 | Distribución de modos de color y resoluciones más frecuentes. |
| `plot_sample_images(df_base, n, random_state, save_path)` | 02 | Cuadrícula de `n` imágenes por clase sin preprocesar. |
| `plot_intensity_histograms(df_base, n_hist, preprocessed, random_state, save_path)` | 02 / 03 | Histograma de intensidad promedio por clase. `preprocessed=True` aplica el pipeline. |
| `compute_intensity_stats(df_base, n_hist, random_state)` | 02 | Calcula `mean`, `std`, `contrast`, `entropy` por imagen. Retorna `pd.DataFrame`. |
| `plot_intensity_metrics(stats_df, title_suffix, save_path)` | 02 / 03 | Panel 2×2 de violin + box + strip para las 4 métricas de intensidad. |
| `plot_preprocessing_steps(df_base, save_path)` | 03 | Cuadrícula 3×3: original / BF / BF+CLAHE por clase. |
| `plot_histogram_comparison(df_base, class_name, save_path)` | 03 | Histogramas antes y después del preprocesamiento para una clase. |
| `build_preprocessing_stats_table(df_base, n_sample, random_state)` | 03 | Tabla `(Class, Stage)` con métricas promedio antes y después del preprocesamiento. |
| `plot_regions_on_images(df_base, save_path)` | 04 | Bordes de las 6 regiones anatómicas sobre imágenes preprocesadas. |
| `compute_regional_stats(df_base, n_reg, random_state)` | 04 | Intensidad media por región para una muestra por clase. Retorna `pd.DataFrame`. |
| `plot_regional_heatmap(reg_df, save_path)` | 04 | Mapa de calor de intensidad media por región y clase. |
| `plot_regional_distributions(reg_df, save_path)` | 04 | Panel 2×3 con eje Y compartido (`sharey=True`) para comparación directa entre regiones. |
| `build_regional_comparison_table(reg_df)` | 04 | Tabla `mean ± std` por región y clase, lista para publicación. |
| `plot_glcm_distributions(features_df, save_path)` | 05 | Panel 1×5 con distribución de las 5 propiedades GLCM por clase en su escala original. |
| `plot_lbp_histograms(features_df, save_path)` | 05 | Histograma LBP promedio por clase. Leyenda y título fuera del área del gráfico. |
| `plot_feature_summary_table(features_df)` | 05 | Tabla de recuento de features por grupo. Retorna `pd.DataFrame`. |

**Paleta académica:**

| Clase | Color | Hex |
|---|---|---|
| `health` | Azul oscuro | `#2166AC` |
| `sick` | Verde oscuro | `#4DAC26` |
| `tb` | Rojo burdeos | `#B2182B` |

---

## Pipeline completo — uso programático

```python
import sys
sys.path.insert(0, '..')   # apuntar a la raíz del proyecto

import pandas as pd
from pathlib import Path
from src.data import build_base_dataframe, LABEL_MAP
from src.features import verify_extractors, extract_all_features
from src.visualization import set_publication_style, plot_regional_heatmap, compute_regional_stats

# 1. Cargar datos
DATA_ROOT = r"C:\ruta\al\proyecto\data\TBX11K"
df_base   = build_base_dataframe(DATA_ROOT)
df_base.to_csv('../data/df_base.csv', index=False)

# 2. Verificar extractores
verify_extractors(df_base)

# 3. Extraer características
features_df, errores = extract_all_features(df_base, sample=False)
features_df.to_csv('../data/features_tbx11k.csv', index=False)

# 4. Visualizar
set_publication_style()
reg_df = compute_regional_stats(df_base, n_reg=200)
plot_regional_heatmap(reg_df)
```

---

## Pipeline de análisis

```text
TBX11K (data/TBX11K/)
   │
   ▼
01. Carga y construcción de metadata  →  data/df_base.csv
   │
   ▼
02. Análisis exploratorio (imágenes crudas)
   │
   ▼
03. Preprocesamiento
   │   ├── Bilateral Filter
   │   └── CLAHE
   │
   ▼
04. Análisis regional (6 zonas anatómicas)
   │
   ▼
05. Extracción de características      →  data/features_tbx11k.csv
   │   ├── GLCM
   │   ├── LBP
   │   ├── HOG
   │   ├── Intensidad global
   │   └── Intensidad regional
   │
   ▼
[Siguiente etapa] Modelado ML
```

---

## Próximas etapas

```text
Extracción de características
          │
          ▼
Análisis de características (correlación, importancia)
          │
          ▼
Reducción de dimensionalidad (PCA, selección de features)
          │
          ▼
Clustering / análisis de estructura
          │
          ▼
Modelos de Machine Learning
          │
          ▼
Evaluación del desempeño (ROC, AUC, F1 por clase)
```

---

## Referencias

- **TB_detection** — [github.com/Krishni1307/TB_detection](https://github.com/Krishni1307/TB_detection): referencia para la estrategia de construcción del DataFrame base.
- **tbx11k-ensemble** — [github.com/behera116/tbx11k-ensemble](https://github.com/behera116/tbx11k-ensemble): referencia para la configuración del Bilateral Filter y CLAHE.
- **Haralick et al. (1973):** Textural features for image classification. *IEEE Transactions on Systems, Man, and Cybernetics*.
- **Ojala et al. (2002):** Multiresolution gray-scale and rotation invariant texture classification with LBP. *IEEE TPAMI*.
- **Dalal & Triggs (2005):** Histograms of oriented gradients for human detection. *CVPR*.
- **Zuiderveld (1994):** Contrast Limited Adaptive Histogram Equalization. *Graphics Gems IV*.
- **Tomasi & Manduchi (1998):** Bilateral filtering for gray and color images. *ICCV*.
