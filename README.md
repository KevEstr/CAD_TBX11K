# cad-tbx11k

Pipeline de análisis exploratorio, preprocesamiento y extracción de características sobre el dataset **TBX11K**, como parte del proyecto de diagnóstico asistido por computador (CAD) para tuberculosis a partir de radiografías de tórax.

---

## Estructura del repositorio

```text
cad-tbx11k/
│
├── cad_tbx11k/                  ← Módulo Python reutilizable (framework)
│   ├── __init__.py              ← Punto de entrada único del paquete
│   ├── data.py                  ← Carga del dataset y construcción del DataFrame base
│   ├── preprocessing.py         ← Cadena de preprocesamiento (BF + CLAHE)
│   ├── features.py              ← Extractores de características (GLCM, LBP, HOG, intensidad)
│   └── visualization.py         ← Figuras de publicación para cada etapa del análisis
│
├── notebooks/
│   ├── 01_Carga_Datos.ipynb
│   ├── 02_Analisis_Exploratorio.ipynb
│   ├── 03_Preprocesamiento.ipynb
│   ├── 04_Analisis_Regional.ipynb
│   └── 05_Extraccion_Caracteristicas.ipynb
│
├── requirements.txt
└── README.md
```

Los notebooks importan directamente del módulo `cad_tbx11k`. Cualquier cambio en un parámetro del framework (por ejemplo, `CLAHE_CLIP_LIMIT`) se propaga de forma consistente a todo el pipeline.

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

### 5. Credenciales de Kaggle

Descargar `kaggle.json` desde **kaggle.com → Settings → API → Create New Token** y colocarlo en la raíz del proyecto. El archivo está excluido por `.gitignore`.

---

## Orden de ejecución

```text
01 → 02 → 03 → 04 → 05
```

---

## Referencia de funciones del módulo `cad_tbx11k`

Todas las funciones se importan desde el paquete principal o desde sus módulos específicos:

```python
import cad_tbx11k as cad
# o bien
from cad_tbx11k.data import build_base_dataframe
```

---

### `cad_tbx11k.data` — Carga de datos

| Función / Constante | Descripción | Retorna |
|---|---|---|
| `LABEL_MAP` | `{'health': 0, 'sick': 1, 'tb': 2}` | `dict` |
| `PALETTE` | Colores por clase para visualización | `dict` |
| `VALID_EXTS` | Extensiones de imagen aceptadas | `tuple` |
| `setup_kaggle_credentials(kaggle_json=None)` | Configura `~/.kaggle/kaggle.json`. Detecta Google Colab automáticamente. | `None` |
| `download_dataset()` | Descarga TBX11K via `kagglehub`. Usa caché local si ya fue descargado. | `str` — ruta de descarga |
| `get_data_root(download_path)` | Verifica y retorna la ruta al directorio `TBX11K/`. | `str` |
| `map_dataset_structure(data_root)` | Mapea `imgs/` y cuenta imágenes por carpeta/subcarpeta. | `pd.DataFrame` — columnas `carpeta`, `subcarpeta`, `imagenes` |
| `build_base_dataframe(data_root)` | Carga el subconjunto etiquetado y construye el DataFrame base con metadatos por imagen. | `pd.DataFrame` — columnas `image_path`, `class_name`, `label`, `ancho`, `alto`, `modo` |

**Ejemplo:**

```python
from cad_tbx11k.data import setup_kaggle_credentials, download_dataset
from cad_tbx11k.data import get_data_root, build_base_dataframe

setup_kaggle_credentials()
download_path = download_dataset()
DATA_ROOT     = get_data_root(download_path)
df_base       = build_base_dataframe(DATA_ROOT)
```

---

### `cad_tbx11k.preprocessing` — Preprocesamiento

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
from cad_tbx11k.preprocessing import preprocess, preprocess_steps

img = preprocess('ruta/imagen.png')           # np.ndarray (256, 256)

pasos = preprocess_steps('ruta/imagen.png')
img_original  = pasos['original']
img_bilateral = pasos['bilateral']
img_clahe     = pasos['clahe']
```

---

### `cad_tbx11k.features` — Extracción de características

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
| `extract_all_features(df_base, sample, sample_size, random_state)` | Extracción masiva sobre el DataFrame base. Muestra progreso con `tqdm`. | `(pd.DataFrame, list[str])` — features y errores |
| `verify_extractors(df_base)` | Verifica la dimensionalidad de cada extractor con `assert`. Lanza error si hay discrepancia. | `None` |
| `get_feature_groups(features_df)` | Clasifica columnas por grupo de extractor. | `dict` — grupos como claves |
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
from cad_tbx11k.features import extract_all_features, verify_extractors

# Verificar antes de procesar
verify_extractors(df_base)

# Extracción completa
features_df, errores = extract_all_features(df_base, sample=False)
features_df.to_csv('features_tbx11k.csv', index=False)
```

---

### `cad_tbx11k.visualization` — Visualización

Todas las figuras usan estilo de publicación científica (`seaborn-v0_8-whitegrid`, DPI 150 en pantalla, 300 para guardar). Cada función incluye una línea comentada para guardar la figura:

```python
# fig.savefig(save_path, dpi=300, bbox_inches="tight")  # descomentar para guardar
```

| Función | Notebook | Descripción |
|---|:---:|---|
| `set_publication_style()` | todos | Configura `rcParams` globales de Matplotlib para estilo de artículo. |
| `plot_dataset_distribution(resumen_df, df_base, save_path)` | 01 | Panel de barras: distribución completa y subconjunto etiquetado con porcentajes. |
| `plot_image_metadata(df_base, save_path)` | 01 | Distribución de modos de color y resoluciones más frecuentes. |
| `plot_sample_images(df_base, n, random_state, save_path)` | 02 | Cuadrícula de `n` imágenes por clase sin preprocesar. |
| `plot_intensity_histograms(df_base, n_hist, preprocessed, random_state, save_path)` | 02 / 03 | Histograma de intensidad promedio por clase. `preprocessed=True` aplica el pipeline. |
| `compute_intensity_stats(df_base, n_hist, random_state)` | 02 | Calcula `mean`, `std`, `contrast`, `entropy` por imagen. Retorna `pd.DataFrame`. |
| `plot_intensity_metrics(stats_df, title_suffix, save_path)` | 02 / 03 | Panel 2×2 de violin + box + strip para las 4 métricas de intensidad. |
| `plot_preprocessing_steps(df_base, save_path)` | 03 | Cuadrícula 3×3: original / BF / BF+CLAHE por clase. |
| `plot_histogram_comparison(df_base, class_name, save_path)` | 03 | Histogramas antes y después del preprocesamiento para una clase. |
| `build_preprocessing_stats_table(df_base, n_sample, random_state)` | 03 | Tabla `(Clase, Etapa)` con métricas promedio antes y después del preprocesamiento. |
| `plot_regions_on_images(df_base, save_path)` | 04 | Dibuja los bordes de las 6 regiones anatómicas sobre imágenes preprocesadas. |
| `compute_regional_stats(df_base, n_reg, random_state)` | 04 | Calcula intensidad media por región para una muestra por clase. Retorna `pd.DataFrame`. |
| `plot_regional_heatmap(reg_df, save_path)` | 04 | Mapa de calor de intensidad media por región y clase. |
| `plot_regional_distributions(reg_df, save_path)` | 04 | Panel 2×3 de violin + box + strip para cada región anatómica. |
| `build_regional_comparison_table(reg_df)` | 04 | Tabla `media ± std` por región y clase, lista para publicación. |
| `plot_glcm_distributions(features_df, save_path)` | 05 | Panel con distribución de las 5 propiedades GLCM por clase. |
| `plot_lbp_histograms(features_df, save_path)` | 05 | Histograma LBP promedio por clase con área sombreada. |
| `plot_feature_summary_table(features_df)` | 05 | Tabla de recuento de features por grupo. Retorna `pd.DataFrame`. |

**Ejemplo — guardar una figura:**

```python
from cad_tbx11k.visualization import plot_regional_heatmap

# Genera la figura en pantalla
plot_regional_heatmap(reg_df)

# Para guardar a archivo, pasar el argumento save_path
# y descomentar la línea fig.savefig() dentro de la función,
# o capturar la figura manualmente:
import matplotlib.pyplot as plt
plot_regional_heatmap(reg_df)
plt.savefig('figura_heatmap_regional.png', dpi=300, bbox_inches='tight')
```

---

## Pipeline completo — uso programático

El módulo puede usarse completamente fuera de los notebooks para integración en scripts o flujos automatizados:

```python
from cad_tbx11k.data import setup_kaggle_credentials, download_dataset
from cad_tbx11k.data import get_data_root, build_base_dataframe
from cad_tbx11k.features import verify_extractors, extract_all_features
from cad_tbx11k.visualization import set_publication_style, plot_regional_heatmap
from cad_tbx11k.visualization import compute_regional_stats

# 1. Configurar y cargar datos
setup_kaggle_credentials()
DATA_ROOT  = get_data_root(download_dataset())
df_base    = build_base_dataframe(DATA_ROOT)

# 2. Verificar extractores
verify_extractors(df_base)

# 3. Extraer características
features_df, errores = extract_all_features(df_base, sample=False)
features_df.to_csv('features_tbx11k.csv', index=False)

# 4. Visualizar
set_publication_style()
reg_df = compute_regional_stats(df_base, n_reg=60)
plot_regional_heatmap(reg_df)
```

---

## Pipeline de análisis

```text
TBX11K
   │
   ▼
01. Carga y construcción de metadata
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
05. Extracción de características
   │   ├── GLCM
   │   ├── LBP
   │   ├── HOG
   │   ├── Intensidad global
   │   └── Intensidad regional
   │
   ▼
features_tbx11k.csv
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
