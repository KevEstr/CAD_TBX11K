# cad-tbx11k

Pipeline de análisis, segmentación pulmonar y modelado sobre **TBX11K** para un
sistema de apoyo al diagnóstico (CAD) de tuberculosis en radiografías de tórax.

El proyecto cubre carga y auditoría del dataset, segmentación del campo pulmonar,
extracción de características, validación cruzada anidada, evaluación estadística,
calibración de probabilidades y predicción con los modelos finales.

## Aportes

- Segmentación pulmonar con PSPNet preentrenado (`torchxrayvision`) y validación
  con las 1.211 cajas de lesión de TBX11K.
- Características de **textura + HOG** calculadas sobre el campo pulmonar
  segmentado. HOG usa el exterior de la máscara relleno con la mediana pulmonar
  para no aprender el fondo ni la geometría del recorte.
- Validación cruzada anidada (5 folds externos × 3 internos) con el selector
  ajustado dentro de cada fold.
- Cinco modelos comparados con el mismo protocolo: regresión logística, SVM,
  árbol de decisión, Random Forest y XGBoost, más un baseline Dummy.
- Evaluación OOF con balanced accuracy, recall de TB, AP, AUC, bootstrap,
  McNemar con Holm y calibración.
- Predicción de una radiografía nueva con `scripts/predict.py`.

---

## Estructura del repositorio

```text
cad-tbx11k/
├── src/
│   ├── config.py              ← muestreo y protocolo
│   ├── data.py                ← carga del dataset
│   ├── lung_segmentation.py   ← segmentación pulmonar con PSPNet
│   ├── preprocessing.py       ← recorte torácico y modos de preprocesamiento
│   ├── features.py            ← textura, HOG, forma y contexto
│   ├── selection.py           ← selectores Chi2, MI y RF
│   ├── modeling.py            ← utilidades de CV y métricas
│   ├── artifacts.py           ← verificación de artefactos de notebooks
│   └── visualization.py       ← figuras
├── notebooks/
│   ├── 01_Carga_Datos.ipynb
│   ├── 02_Analisis_Exploratorio.ipynb
│   ├── 03_Preprocesamiento.ipynb
│   ├── 04_Analisis_Regional.ipynb
│   ├── 05_Extraccion_Caracteristicas.ipynb
│   ├── 06_Seleccion_Caracteristicas.ipynb
│   ├── 07_Desbalanceo.ipynb
│   ├── 08_Modelos_ML.ipynb
│   ├── 09_Evaluacion_Comparativa.ipynb
│   └── 10_Interpretabilidad.ipynb
├── scripts/
│   ├── build_model_meta.py      ← df_base.csv → df_model.csv
│   ├── validate_segmentation.py ← validación con cajas de lesión
│   ├── validate_masks.py        ← Dice/IoU contra máscaras manuales
│   ├── preview_segmentation.py  ← control visual de la máscara
│   ├── precompute_masks.py      ← caché de máscaras PSPNet
│   ├── train_lesion_patches.py  ← especialista de parches de lesión
│   ├── predict_lesion.py        ← predicción con parches de lesión
│   ├── evaluate_external.py     ← evaluación en un dataset externo
│   └── predict.py               ← predicción individual o por lote
├── data/                        ← dataset y artefactos (excluido de Git)
├── models/                      ← modelos finales (excluido de Git)
├── reports/                     ← figuras
└── exports/                     ← PDF/HTML generados (excluido de Git)
```

---

## Dataset

**Fuente:** [TBX11K — mmcheng.net](https://mmcheng.net/tb/)  
**Kaggle:** [usmanshams/tbx-11](https://www.kaggle.com/datasets/usmanshams/tbx-11)

| Clase | Etiqueta | Imágenes etiquetadas | Únicas tras deduplicar |
|---|---:|---:|---:|
| `health` | 0 | 3.800 | 3.800 |
| `sick` | 1 | 3.800 | 3.656 |
| `tb` | 2 | 800 | 800 |
| **Total** | | **8.400** | **8.256** |

- Los 144 duplicados exactos pertenecen a `sick`; la lista canónica está en
  `src/dedup_drop_names.csv`.
- El conjunto `test` (3.302 imágenes) no tiene etiquetas públicas y no se usa.
- No hay identificadores de paciente; la validación cruzada es estratificada
  por imagen.
- Anotaciones COCO: 1.211 cajas de lesión en 799 imágenes TB.

### Cadena de archivos

```text
01_Carga_Datos.ipynb
        └── data/df_base.csv                 (8.400 imágenes)
                ├── 02, 03, 04              (EDA y preprocesamiento)
                └── scripts/build_model_meta.py
                        └── data/df_model.csv (8.256 tras dedup exacta)
                                └── 05_Extraccion_Caracteristicas.ipynb
                                        └── data/features_tbx11k.csv (1.860)
                                                └── 06 → X_full.npy / y_full.npy / 06_seleccion.json
                                                        └── 07 → 07_estrategia.json
                                                                └── 08 → modelos 08_*.joblib + OOF
                                                                        └── 09 → evaluación
                                                                                └── 10 → interpretabilidad
```

---

## Pipeline

### 1. Segmentación pulmonar (`src/lung_segmentation.py`)

Se usa **PSPNet de `torchxrayvision`**, preentrenado para segmentación anatómica
de tórax. El modelo devuelve las clases `Left Lung` y `Right Lung`; la máscara
final es la unión de ambas, umbralizada y dilatada 12 px para incluir lesiones
periféricas.

- La inferencia corre en CPU.
- Las máscaras se guardan en `data/masks` para no recalcularlas.
- La primera ejecución descarga los pesos del modelo en
  `~/.torchxrayvision/models_data`.
- Si el modelo no detecta pulmones, se usa un campo torácico central marcado con
  `fallback=True` para no romper la extracción.

La validación con las cajas de lesión está en `scripts/validate_segmentation.py`.
La validación con máscaras manuales (Dice/IoU) está en
`scripts/validate_masks.py`.

### 2. Preprocesamiento (`src/preprocessing.py`)

| Modo | Descripción |
|---|---|
| `thorax` | recorte torácico + filtro bilateral + CLAHE |
| `lung` | como `thorax`, exterior relleno con la mediana pulmonar |
| `context` | imagen completa + máscara de fondo |
| `full` | imagen completa + filtro bilateral + CLAHE |
| `mask` | exterior a 0 (solo compatibilidad) |

El modo principal del modelo es `thorax`.

### 3. Características (`src/features.py`)

| Familia | N | Contenido |
|---|---:|---|
| `texture` | 96 | intensidad global/regional, GLCM, LBP y Gabor dentro del pulmón |
| `hog` | 1.764 | HOG 32×32, 7×7 bloques, calculado con la máscara rellena |
| **Total del modelo** | **1.832** | textura pura (68) + HOG (1.764) |

`shape`, `context` y `quality` existen en `src/features.py` para análisis de
sesgo, pero **no se usan** en la matriz principal del modelo. Las columnas
`mask_area_fraction`, `mask_lr_balance`, `mask_contrast` y `mask_fallback` se
guardan para control de calidad, no como features.

### 4. Modelado (`08_Modelos_ML.ipynb`)

- Validación cruzada externa de 5 folds e interna de 3 folds.
- Selector (Chi², MI o RF) elegido en el notebook 06 y ajustado dentro de cada
  fold.
- Modelos: Logistic Regression, SVM, Decision Tree, Random Forest y XGBoost,
  más Dummy.
- Métricas: balanced accuracy, F1 macro, recall de TB, AUC-ROC macro, AP de TB.
- El mejor modelo se elige por balanced accuracy OOF; los cinco modelos finales
  se reajustan con las 8.256 imágenes y se guardan en `models/`.

---

## Instalación

```powershell
python -m venv venv
.\venv\Scripts\activate

# PyTorch CPU (recomendado si no hay GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

La primera segmentación descarga los pesos de PSPNet en
`~/.torchxrayvision/models_data` (requiere internet una vez).

El dataset debe descargarse en `data/TBX11K/` con la estructura
`imgs/{health,sick,tb}` y las anotaciones COCO en
`data/TBX11K/annotations/json/`.

---

## Reproducción

`powershell
# 1. Carga: genera df_base.csv y df_model.csv
jupyter nbconvert --to notebook --execute notebooks/01_Carga_Datos.ipynb --inplace

# 2. Análisis exploratorio y preprocesamiento
jupyter nbconvert --to notebook --execute notebooks/02_Analisis_Exploratorio.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/03_Preprocesamiento.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/04_Analisis_Regional.ipynb --inplace

# 3. Features con PSPNet + validación automática de máscaras
jupyter nbconvert --to notebook --execute notebooks/05_Extraccion_Caracteristicas.ipynb --inplace

# 4. Selección y desbalanceo
jupyter nbconvert --to notebook --execute notebooks/06_Seleccion_Caracteristicas.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/07_Desbalanceo.ipynb --inplace

# 5. Entrenamiento de los cinco modelos
jupyter nbconvert --to notebook --execute notebooks/08_Modelos_ML.ipynb --inplace

# 6. Evaluación e interpretabilidad
jupyter nbconvert --to notebook --execute notebooks/09_Evaluacion_Comparativa.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/10_Interpretabilidad.ipynb --inplace
`

El notebook 05 ejecuta la validación de segmentación con las máscaras cacheadas.
Las herramientas de diagnóstico son opcionales:

`powershell
python scripts/validate_masks.py --masks-dir ruta/a/mascaras
python scripts/preview_segmentation.py --image ruta/a/radiografia.png --out-dir reports/seg
python scripts/precompute_masks.py   # opcional; acelera la primera corrida de 05
python scripts/evaluate_external.py --manifest ruta/manifest_externo.csv
`

### Predicción de una imagen

```powershell
python scripts/predict.py --image ruta/a/radiografia.png
python scripts/predict.py --image ruta.png --model "XGBoost"
python scripts/predict.py --image ruta.png --save-mask mascara.png --save-preprocessed thorax.png
python scripts/predict.py --image ruta.png --heatmap heatmap.png
python scripts/predict.py --csv data/imagenes.csv --out data/predicciones.csv
```

El manifiesto `models/08_manifest.json` indica el modelo seleccionado por la CV
y `models/08_feature_schema.json` fija el orden exacto de las features. Si el
CSV de lote incluye `label`, se calculan balanced accuracy, recall por clase y
matriz de confusión.

### Salidas principales

- `data/df_model.csv`
- `data/features_tbx11k.csv`
- `data/X_full.npy`, `data/y_full.npy`
- `data/06_seleccion.json`, `data/06_selected_feature_names.json`
- `data/07_estrategia.json`
- `data/08_oof_predicciones.csv`, `data/08_oof_scores.npz`
- `data/08_cv_resumen.csv`, `data/08_cv_por_fold.csv`
- `data/08_mejores_params.json`, `data/08_procedimiento.json`, `data/08_manifest.json`
- `models/08_*.joblib`, `models/08_manifest.json`
- `data/09_*.csv`, `data/09_*.json`
- `reports/03_segmentacion_*.png`, `reports/06_seleccion.png`, `reports/09_*.png`

---

## Artefactos generados

`data/`, `exports/` y `models/*.joblib` están excluidos de Git. Los notebooks,
scripts y figuras de `reports/` se versionan. Los PDF/HTML de `exports/` se
regeneran con `nbconvert` cuando se necesiten.

---

## Alcance y consideraciones

- **Segmentación**: se usa PSPNet preentrenado y las máscaras se dilatan 12 px
  para incluir lesiones periféricas. TBX11K no tiene máscaras manuales de pulmón,
  así que la validación principal usa las cajas de lesión. Para medir Dice/IoU
  contra un subconjunto anotado manualmente se incluye `scripts/validate_masks.py`.
- **Contexto de adquisición**: TBX11K presenta confusión entre clase y fuente.
  El modelo principal usa solo textura + HOG del campo pulmonar; el notebook 09
  incluye un baseline de fondo/borde para cuantificar ese sesgo.
- **Test**: las 3.302 imágenes de `test` no tienen etiquetas públicas; no se
  dispone de validación externa.
- **Pacientes**: no hay identificadores de paciente; la CV es estratificada por
  imagen.
- **Umbral**: se estima con CV y es una estimación interna.
