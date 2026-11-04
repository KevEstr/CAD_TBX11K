# cad-tbx11k

Pipeline de análisis, segmentación pulmonar y modelado sobre **TBX11K**
para un sistema de apoyo al diagnóstico (CAD) de tuberculosis en radiografías
de tórax.

El proyecto cubre carga y auditoría del dataset, segmentación del campo
pulmonar, extracción de características, validación cruzada anidada,
calibración de probabilidades, evaluación estadística y control de sesgo por
contexto de adquisición.

## Aportes

- Segmentación pulmonar validada con las 1.211 cajas de lesión de TBX11K
  (cobertura media 0.912).
- Características con soporte de máscara, separadas en textura, HOG, forma y
  contexto para poder auditar qué información usa el modelo.
- Validación cruzada anidada, umbral clínico anidado, calibración a
  prevalencia real, bootstrap y McNemar con Holm.
- Control de sesgo de adquisición mediante residualización contextual y un
  modelo descontaminado como referencia conservadora.
- Pipeline reproducible con scripts, notebooks y verificación automática de
  artefactos.

---

## Estructura del repositorio

```text
cad-tbx11k/
├── src/
│   ├── config.py              ← muestreo y protocolo
│   ├── data.py                ← carga del dataset
│   ├── lung_segmentation.py   ← segmentación pulmonar
│   ├── preprocessing.py       ← recorte torácico, strict y armonizado
│   ├── features.py            ← textura, HOG, forma y contexto
│   ├── modeling.py            ← CV anidada, métricas y residualización
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
│   ├── build_model_meta.py          ← df_base.csv → df_model.csv
│   ├── fit_final_model.py           ← modelo final descontaminado
│   ├── rebuild_features.py          ← matriz principal
│   ├── extract_strict_features.py   ← textura estricta/armonizada
│   ├── validate_segmentation.py     ← validación con cajas de lesión
│   ├── run_modeling_v2.py           ← CV anidada
│   ├── run_imbalance_v2.py          ← desbalanceo
│   ├── run_deconfound_v2.py         ← residualización contexto/forma
│   ├── run_strict_deconfound.py     ← strict/harmonized + residualización
│   └── run_ipw_v2.py                ← ponderación por propensión inversa
├── data/                      ← dataset y artefactos (excluido de Git)
├── models/                    ← modelos finales
├── reports/                   ← figuras
└── exports/                   ← PDF/HTML generados (excluido de Git)
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

- Los 144 duplicados exactos pertenecen a `sick`.
- El conjunto `test` (3.302 imágenes, `unknown_*.png`) no tiene etiquetas
  públicas y no se usa.
- No hay identificadores de paciente; la validación cruzada es estratificada
  por imagen, no por paciente.
- Anotaciones COCO: 1.211 cajas de lesión en 799 imágenes TB
  (`ActiveTuberculosis` 972, `ObsoletePulmonaryTuberculosis` 239).

### Cadena de archivos

```text
01_Carga_Datos.ipynb
        └── data/df_base.csv                 (8.400 imágenes)
                ├── 02, 03, 04              (EDA y preprocesamiento)
                └── scripts/build_model_meta.py
                        └── data/df_model.csv (8.256 tras dedup exacta)
                                ├── scripts/rebuild_features.py
                                │       └── data/features_tbx11k_v2.csv
                                ├── scripts/extract_strict_features.py
                                │       └── data/features_strict.csv
                                └── 05–10 (features, modelos, evaluación)
```

La lista de los 144 duplicados está en `src/dedup_drop_names.csv`.

---

## Pipeline

### 1. Segmentación pulmonar (`src/lung_segmentation.py`)

1. Máscara corporal por componentes oscuras conectadas al borde.
2. Score de oscuridad local `gaussiano(σ=24) − imagen`.
3. Separación izquierda/derecha y eliminación del mediastino central.
4. Envolvente convexa y margen de 20 px para incluir opacidades y lesiones
   periféricas.
5. Control de calidad: área, balance, contraste y `fallback`.
6. Máscara estricta adicional (erosión de 12 px) para textura sin pared
   torácica.

La máscara principal es un campo pulmonar con margen; la máscara estricta se
usa para el análisis de parénquima.

### 2. Preprocesamiento (`src/preprocessing.py`)

| Modo | Descripción |
|---|---|
| `thorax` | recorte torácico + filtro bilateral + CLAHE |
| `lung` | como `thorax`, exterior relleno con la mediana pulmonar |
| `strict` | máscara erosionada, recorte ajustado, exterior con mediana |
| `strict_harmonized` | strict + normalización p2–p98, sin CLAHE |
| `context` | imagen completa + máscara de fondo |
| `full` | imagen completa + filtro bilateral + CLAHE |
| `mask` | exterior a 0 (solo compatibilidad) |

### 3. Características (`src/features.py`)

| Familia | N | Contenido |
|---|---:|---|
| `texture` | 96 | intensidad global/regional, GLCM, LBP y Gabor dentro de la máscara |
| `hog` | 1.764 | HOG 32×32, 7×7 bloques |
| `shape` | 64 | geometría de la máscara + miniatura 8×8 |
| `context` | 352 | intensidad/textura del fondo + miniatura 16×16 |
| `quality` | 4 | área, balance, contraste y fallback |
| `strict_*` / `stricth_*` | 212 | textura estricta y armonizada |

La textura usa la máscara como soporte; el exterior se rellena con la mediana
pulmonar. El contraste se calcula como `p95 − p5`.

### 4. Modelado (`src/modeling.py`)

- Validación cruzada externa de 5 folds e interna de 3 folds.
- Selección ANOVA F dentro de cada fold.
- Modelos: Dummy, regresión logística, SVM RBF, RandomForest y XGBoost.
- Métricas: balanced accuracy, sensibilidad/especificidad TB, average
  precision, AUC, Brier, bootstrap estratificado y McNemar con Holm.
- Umbral clínico: mayor umbral que mantiene sensibilidad ≥ 0.90.
- Calibración de probabilidades a la prevalencia observada.
- `ContextResidualizer`: elimina de la textura la componente lineal explicable
  por contexto/forma, ajustado dentro de cada fold.

---

## Resultados

Evaluación con predicciones out-of-fold, 5 folds externos y 3 internos.

### Segmentación

| Método | Cobertura media de cajas | Cobertura ≥ 0.8 | Centroide dentro |
|---|---:|---:|---:|
| Otsu | 0.539 | 15.3 % | 73.1 % |
| Segmentación anatómica | **0.912** | **89.1 %** | **99.1 %** |

### Familias y modelos

| Conjunto | Modelo | BalAcc | Sens TB | Espec | AP TB | AUC |
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

### Control de sesgo de contexto

| Conjunto | Modelo | BalAcc | AP | AUC |
|---|---|---:|---:|---:|
| textura | SVM | 0.870 | 0.714 | 0.936 |
| textura local (sin intensidad global) | SVM | 0.853 | 0.669 | 0.920 |
| textura residualizada vs contexto | SVM | 0.720 | 0.365 | 0.798 |
| textura estricta | SVM | 0.874 | 0.745 | 0.941 |
| textura estricta residualizada vs contexto | SVM | 0.735 | 0.366 | 0.805 |
| textura armonizada residualizada | SVM | 0.711 | 0.326 | 0.778 |
| IPW (propensión inversa) | SVM | 0.872 | 0.745 | 0.937 |

### Calibración (LR textura)

- Brier sin calibrar: 0.120.
- Brier calibrada: **0.057**.
- Umbral Youden calibrado: 0.081 → sensibilidad 0.811, especificidad 0.819.
- Sensibilidad ≥ 0.90: umbral 0.041 → sensibilidad 0.900, especificidad 0.672.

### Desbalanceo

| Estrategia | BalAcc | Recall TB | AP |
|---|---:|---:|---:|
| SMOTE | 0.814 | 0.778 | 0.600 |
| `class_weight='balanced'` | 0.812 | 0.773 | 0.599 |
| RandomOverSampler | 0.805 | 0.760 | 0.596 |
| RandomUnderSampler | 0.803 | 0.774 | 0.586 |
| Sin desbalanceo | 0.702 | 0.418 | 0.626 |

---

## Instalación

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

El dataset debe descargarse en `data/TBX11K/` con la estructura
`imgs/{health,sick,tb}` y las anotaciones COCO en
`data/TBX11K/annotations/json/`.

---

## Reproducción

```powershell
# 1. Carga y DataFrame base
jupyter nbconvert --to notebook --execute notebooks/01_Carga_Datos.ipynb --inplace

# 2. Metadatos de modelado y features
python scripts/build_model_meta.py
python scripts/rebuild_features.py --workers 8

# 3. Features estrictas y validación de segmentación
python scripts/extract_strict_features.py --workers 8
python scripts/validate_segmentation.py

# 4. Modelado principal
python scripts/run_modeling_v2.py

# 5. Desbalanceo y control de sesgo
python scripts/run_imbalance_v2.py
python scripts/run_deconfound_v2.py
python scripts/run_strict_deconfound.py
python scripts/run_ipw_v2.py

# 6. Modelo final
python scripts/fit_final_model.py

# 7. Notebooks 03 y 05–10
jupyter nbconvert --to notebook --execute notebooks/03_Preprocesamiento.ipynb --inplace
```

Los notebooks 03 y 05–10 verifican sus artefactos y ejecutan el productor
correspondiente si falta algún archivo.

### Salidas principales

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

## Artefactos generados

`data/`, `exports/` y `models/*.joblib` están excluidos de Git. Los notebooks,
scripts y figuras de `reports/` se versionan. Los PDF/HTML de `exports/` se
regeneran con `nbconvert` cuando se necesiten.

---

## Alcance y consideraciones

- **Contexto de adquisición**: TBX11K presenta confusión entre clase y fuente.
  El pipeline incluye un control explícito de ese sesgo y reporta un modelo
  residualizado como referencia conservadora. Los modelos que incorporan
  contexto/forma deben interpretarse como clasificadores de fuente, no como
  detección clínica.
- **Segmentación**: no hay máscaras manuales de pulmón; la validación usa las
  cajas de lesión. La máscara principal es un campo pulmonar con margen.
- **Test**: las 3.302 imágenes de `test` no tienen etiquetas públicas; no se
  dispone de validación externa.
- **Pacientes**: no hay identificadores de paciente; la CV es estratificada
  por imagen.
- **Umbral**: se estima con CV anidada y es una estimación interna.
- **Modelo residualizado**: la residualización puede eliminar también señal
  pulmonar correlacionada con la fuente; se reporta como cota conservadora.
