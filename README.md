# cad-tbx11k

Pipeline de análisis, segmentación pulmonar y modelado clásico sobre **TBX11K**
para un CAD de tuberculosis a partir de radiografías de tórax.

**Versión 2.0** — segmentación validada con cajas de lesión, features con
soporte de máscara, CV anidada, calibración y **control cuantitativo del atajo
de contexto/fuente**.

---

## Resultado principal (honesto)

La segmentación anterior (Otsu invertido) dejaba fuera las lesiones. La nueva
segmentación las incluye. Pero al medir el atajo de fuente aparece el hallazgo
central del proyecto:

| Modelo (5-fold anidada, OOF) | BalAcc | Sens TB | Espec | AP TB | AUC |
|---|---:|---:|---:|---:|---:|
| Contexto/fondo solo (SVM) | **0.927** | 0.920 | 0.934 | 0.889 | 0.974 |
| Textura pulmonar cruda (SVM) | 0.870 | 0.836 | 0.904 | 0.714 | 0.936 |
| Textura estricta, sin pared (SVM) | 0.874 | 0.868 | 0.881 | 0.745 | 0.941 |
| Textura + forma + contexto (LR) | 0.934 | 0.925 | 0.943 | 0.893 | 0.976 |
| **Textura estricta residualizada (SVM)** | **0.735** | 0.752 | 0.717 | 0.366 | 0.805 |

**Lectura:** el contexto solo supera a la textura pulmonar. Añadir forma y
contexto infla el rendimiento a 0.934. Al residualizar la textura contra el
contexto (eliminar la parte de la textura explicable por la fuente), el
rendimiento baja a **0.735**. Esa es la cota inferior conservadora atribuible
al pulmón; el 0.93 no es un modelo de TB, es un modelo de adquisición/fuente.

La conclusión defendible es: **con TBX11K, la señal de textura pulmonar
descontaminada es moderada (~0.74 BalAcc); el rendimiento alto reportado por
el pipeline anterior estaba contaminado por el contexto.** No se puede
“arreglar” el dataset con post-procesado; las opciones reales son datos
externos con adquisición balanceada o adaptación de dominio/DL.

---

## Qué cambió respecto a la versión anterior

1. **Segmentación reemplazada** (`src/lung_segmentation.py`):
   máscara corporal + contraste local multi-escala + separación izquierda/
   derecha + envolvente convexa + margen perilesional. Validada con las 1.211
   cajas de lesión de TBX11K.
2. **Validación cuantitativa de la máscara**:
   cobertura media de cajas 0.912 (antes 0.539); 89.1 % de imágenes con
   cobertura ≥ 0.8 (antes 15.3 %); centroide dentro 99.1 %; fallback 1.25 %.
3. **Features con soporte de máscara** (`src/features.py`):
   intensidad/GLCM/LBP/Gabor solo dentro del campo pulmonar; exterior relleno
   con la mediana (no con 0). HOG sobre el recorte completo, sin máscara.
   Contraste robusto `p95−p5` en lugar de `max−min` (elimina el artefacto
   `ID_contrast ≈ 245`).
4. **Separación explícita de familias**:
   `texture` (96), `hog` (1.764), `shape` (64), `context` (352),
   `quality` (4). Forma y contexto son **controles de sesgo**, no features
   del modelo principal.
5. **CV anidada** (5 folds externos × 3 internos), selección ANOVA F dentro
   del fold, umbral clínico anidado, calibración a prevalencia real,
   bootstrap y McNemar con Holm.
6. **Corrección del atajo**: `ContextResidualizer` en `src/modeling.py`
   residualiza la textura contra contexto/forma dentro de cada fold.
7. **COCO corregido**: se usan las categorías 1 y 2 (1.211 cajas); antes el
   filtro caía en la categoría 2 y solo veía 239 cajas.

---

## Estructura del repositorio

```text
cad-tbx11k/
├── src/
│   ├── config.py              ← muestreo y protocolo
│   ├── data.py                ← carga del dataset
│   ├── lung_segmentation.py   ← segmentación pulmonar validada
│   ├── preprocessing.py       ← recorte torácico, strict, armonizado
│   ├── features.py            ← textura, HOG, forma, contexto
│   ├── modeling.py            ← CV anidada, métricas, residualización
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
│   ├── build_model_meta.py          ← df_base.csv → df_model.csv (8.256)
│   ├── fit_final_model.py           ← guarda el SVM descontaminado
│   ├── rebuild_features.py          ← matriz principal (8.256 imágenes)
│   ├── extract_strict_features.py   ← textura estricta/armonizada
│   ├── validate_segmentation.py     ← validación con cajas de lesión
│   ├── run_modeling_v2.py           ← CV anidada de familias/modelos
│   ├── run_imbalance_v2.py          ← class_weight vs SMOTE/under/over
│   ├── run_deconfound_v2.py         ← residualización contexto/forma
│   ├── run_strict_deconfound.py     ← strict/harmonized + residualización
│   └── run_ipw_v2.py                ← ponderación por propensión inversa
├── data/                      ← dataset y artefactos (excluido de Git)
├── models/                    ← modelos finales
├── reports/                   ← figuras
└── exports/                   ← PDF/HTML
```

---

## Dataset

**Fuente:** [TBX11K — mmcheng.net](https://mmcheng.net/tb/) ·
**Kaggle:** [usmanshams/tbx-11](https://www.kaggle.com/datasets/usmanshams/tbx-11)

| Clase | Etiqueta | Imágenes etiquetadas | Únicas tras deduplicar |
|---|---:|---:|---:|
| `health` | 0 | 3.800 | 3.800 |
| `sick` | 1 | 3.800 | 3.656 |
| `tb` | 2 | 800 | 800 |
| **Total** | | **8.400** | **8.256** |

- Los 144 duplicados exactos están todos en `sick`.
- El conjunto `test` (3.302 imágenes, `unknown_*.png`) **no tiene etiquetas
  públicas** y no se usa. Se documenta como limitación.
- No hay IDs de paciente; la CV estratificada no puede agrupar por paciente.
- Anotaciones COCO: 1.211 cajas de lesión en 799 imágenes TB
  (`ActiveTuberculosis` 972, `ObsoletePulmonaryTuberculosis` 239).

### Cadena de archivos (sin dependencias invertidas)

```text
01_Carga_Datos.ipynb
        └── data/df_base.csv                 (8.400 imágenes)
                ├── 02, 03, 04              (EDA y preprocesamiento)
                └── scripts/build_model_meta.py
                        └── data/df_model.csv (8.256 tras dedup exacta)
                                ├── scripts/rebuild_features.py → features_tbx11k_v2.csv
                                ├── scripts/extract_strict_features.py → features_strict.csv
                                └── 05–10 (features, selección, modelos, evaluación)
```

`df_model.csv` se genera antes de extraer features. Ningún notebook anterior a
05 lee un archivo producido por 05 o 06. La lista exacta de los 144 duplicados
está en `src/dedup_drop_names.csv`, por lo que la deduplicación es reproducible
sin depender de la matriz antigua.

---

## Pipeline

### 1. Segmentación pulmonar (`src/lung_segmentation.py`)

1. Máscara corporal por componentes oscuras conectadas al borde.
2. Score de oscuridad local `gaussiano(σ=24) − imagen` (robusto a
   iluminación global).
3. Separación izquierda/derecha y eliminación del mediastino central.
4. Envolvente convexa + margen de 20 px: incluye opacidades y lesiones
   periféricas/apicales.
5. Control de calidad: área, balance, contraste y `fallback`.
6. Máscara **estricta** adicional (erosión 12 px) para análisis de
   parénquima sin pared torácica.

La máscara principal es un **campo pulmonar con margen**, no una
segmentación clínica de parénquima. Se reporta como tal.

### 2. Preprocesamiento (`src/preprocessing.py`)

| Modo | Qué hace | Uso |
|---|---|---|
| `thorax` | recorte torácico + BF + CLAHE | pipeline principal |
| `lung` | como `thorax`, exterior = mediana pulmonar | control |
| `strict` | máscara erosionada, recorte ajustado, exterior = mediana | textura sin pared |
| `strict_harmonized` | strict + normalización p2–p98, sin CLAHE | control de adquisición |
| `context` | imagen completa + máscara de fondo | control de fuente |
| `full` | imagen completa + BF + CLAHE | referencia |
| `mask` | exterior a 0 | solo compatibilidad |

### 3. Features (`src/features.py`)

| Familia | N | Contenido |
|---|---:|---|
| `texture` | 96 | intensidad global/regional, GLCM, LBP, Gabor dentro de la máscara |
| `hog` | 1.764 | HOG 32×32, 7×7 bloques, mapeable a posición espacial |
| `shape` | 64 | geometría de la máscara + miniatura 8×8 |
| `context` | 352 | intensidad/textura del fondo + miniatura 16×16 |
| `quality` | 4 | área, balance, contraste, fallback |
| `strict_*` / `stricth_*` | 212 | textura estricta y armonizada |

Total de la matriz principal: **2.290 features** para 8.256 imágenes.

### 4. Modelado (`src/modeling.py`)

- CV externa 5-fold estratificada; CV interna 3-fold.
- Selección ANOVA F (`f_classif`) **dentro del fold**; determinista.
- Modelos: Dummy, LR, SVM RBF, RandomForest, XGBoost.
- Métricas: balanced accuracy, sensibilidad/especificidad TB, AP, AUC,
  Brier, bootstrap estratificado y McNemar con Holm.
- Umbral clínico: el **mayor** umbral que mantiene sensibilidad ≥ 0.90
  (maximiza especificidad).
- Calibración: corrección de prior a la prevalencia real
  (0.0969) para modelos con `class_weight='balanced'`.
- `ContextResidualizer`: elimina de la textura la parte linealmente
  explicable por contexto/forma, ajustado dentro de cada fold.

---

## Resultados

### Segmentación

| Método | Cobertura media | Cobertura ≥0.8 | Centroide dentro |
|---|---:|---:|---:|
| Otsu anterior | 0.539 | 15.3 % | 73.1 % |
| Nueva segmentación | **0.912** | **89.1 %** | **99.1 %** |

Calidad por clase: área 0.49 (health), 0.46 (sick), 0.52 (tb); balance
0.82–0.85; fallback 0–1.2 %. La diferencia de área entre clases es una señal
residual de forma que el baseline `shape` cuantifica.

### Familias y modelos

| Conjunto | Modelo | BalAcc (Youden) | Sens | Espec | AP | AUC |
|---|---|---:|---:|---:|---:|---:|
| texture | LR | 0.815 | 0.811 | 0.819 | 0.591 | 0.890 |
| texture | SVM | 0.870 | 0.836 | 0.904 | 0.714 | 0.936 |
| texture | XGB | 0.851 | 0.850 | 0.852 | 0.729 | 0.923 |
| texture | RF | 0.854 | 0.842 | 0.866 | 0.732 | 0.927 |
| hog | LR | 0.779 | 0.774 | 0.785 | 0.468 | 0.850 |
| texture+hog | LR | 0.855 | 0.826 | 0.883 | 0.697 | 0.922 |
| context | LR | 0.892 | 0.888 | 0.897 | 0.807 | 0.952 |
| context | SVM | 0.927 | 0.920 | 0.934 | 0.889 | 0.974 |
| shape | LR | 0.652 | 0.698 | 0.607 | 0.191 | 0.697 |
| texture+shape+context | LR | 0.934 | 0.925 | 0.943 | 0.893 | 0.976 |

### Corrección del atajo

| Conjunto | Modelo | BalAcc | AP | AUC |
|---|---|---:|---:|---:|
| textura cruda | SVM | 0.870 | 0.714 | 0.936 |
| textura local (sin intensidad global) | SVM | 0.853 | 0.669 | 0.920 |
| textura residualizada vs contexto | SVM | 0.720 | 0.365 | 0.798 |
| textura estricta | SVM | 0.874 | 0.745 | 0.941 |
| **textura estricta residualizada vs contexto** | **SVM** | **0.735** | **0.366** | **0.805** |
| textura armonizada residualizada | SVM | 0.711 | 0.326 | 0.778 |
| IPW (propensión inversa) | SVM | 0.872 | 0.745 | 0.937 |

La residualización es la única corrección que elimina el atajo; IPW y la
armonización no lo corrigen porque el sesgo está **dentro de la textura**,
no solo en las esquinas. La máscara estricta mejora ligeramente la cota
descontaminada (0.735 vs 0.720).

### Calibración (LR textura)

- Brier sin calibrar: 0.120.
- Brier calibrada a prevalencia: **0.057**.
- Umbral Youden calibrado: 0.081 → sens 0.811, espec 0.819.
- Sensibilidad ≥ 0.90: umbral 0.041 → sens 0.900, espec 0.672. No alcanza el
  perfil OMS (≥90 % sens, ≥70 % espec).

### Desbalanceo

| Estrategia | BalAcc | Recall TB | AP |
|---|---:|---:|---:|
| SMOTE | 0.814 | 0.778 | 0.600 |
| `class_weight='balanced'` | 0.812 | 0.773 | 0.599 |
| RandomOverSampler | 0.805 | 0.760 | 0.596 |
| RandomUnderSampler | 0.803 | 0.774 | 0.586 |
| Sin desbalanceo | 0.702 | 0.418 | 0.626 |

Se recomienda `class_weight='balanced'` por simplicidad; SMOTE no aporta una
mejora que justifique su coste.

---

## Cómo reproducir

```powershell
# 1. Entorno
.\venv\Scripts\activate
pip install -r requirements.txt

# 2. Segmentación y validación
python scripts/validate_segmentation.py

# 3. Metadatos de modelado (8.256) y features principales
python scripts/build_model_meta.py
python scripts/rebuild_features.py --workers 8

# 4. Features estrictas/armonizadas
python scripts/extract_strict_features.py --workers 8

# 5. Modelado principal
python scripts/run_modeling_v2.py

# 6. Desbalanceo
python scripts/run_imbalance_v2.py

# 7. Corrección del atajo
python scripts/run_deconfound_v2.py
python scripts/run_strict_deconfound.py
python scripts/run_ipw_v2.py

# 8. Modelo final descontaminado
python scripts/fit_final_model.py

# 9. Notebooks
jupyter nbconvert --to notebook --execute notebooks/03_Preprocesamiento.ipynb --inplace
# ... 05 a 10
```

Salidas principales:

- `data/seg_summary_v2.json`
- `data/features_tbx11k_v2.csv`
- `data/features_strict.csv`
- `data/v2/summary_models.csv`
- `data/v2/deconfound_summary.csv`
- `data/v2/strict_deconfound_summary.csv`
- `data/v2/ipw_summary.csv`
- `models/v2_lr_texture.joblib`
- `models/v2_svm_strict_resid_context.joblib`
- `reports/03_segmentacion_*.png`
- `reports/v2/*.png`

---

## Limitaciones

1. **Sesgo de fuente**: TBX11K confunde clase con adquisición. La textura
   contiene señal de fuente; la residualización la elimina y el rendimiento
   baja a ~0.735. No hay forma de mantener 0.93 sin usar el atajo.
2. **Segmentación**: no hay máscaras manuales de pulmón; la validación usa
   cajas de lesión. La máscara es un campo pulmonar con margen, no
   parénquima fino.
3. **Test no usado**: 3.302 imágenes sin etiquetas públicas. No hay
   validación externa.
4. **Sin IDs de paciente**: la CV no puede agrupar por paciente; los
   duplicados perceptuales (dHash 8×8) son un control débil.
5. **Umbral**: el umbral de sensibilidad ≥0.90 se elige en CV anidada, pero
   sigue siendo una estimación interna.
6. **Modelo descontaminado**: es una cota inferior conservadora. La
   residualización puede eliminar también señal pulmonar correlacionada con
   la fuente; el valor verdadero está entre 0.735 y 0.870.

---

## Recomendación

- **No presentar** el modelo de 0.93 como detección de TB: es un clasificador
  de fuente/adquisición.
- **Modelo descontaminado**: SVM sobre textura estricta residualizada
  (`models/v2_svm_strict_resid_context.joblib`), BalAcc 0.735, AP 0.366,
  AUC 0.805. Es la referencia honesta del proyecto.
- **Modelo de contexto**: útil como **detector de dominio/fuente** para
  marcar imágenes fuera de distribución, no para diagnosticar.
- **Para mejorar sin atajo**: dataset externo con adquisición balanceada,
  segmentación con modelo preentrenado, o adaptación de dominio/DL. Con
  TBX11K solo, el techo honesto es ~0.74.
- El CAD debe ser **apoyo al triaje**, no diagnóstico, con probabilidades
  calibradas y umbral fijado por sensibilidad objetivo.
