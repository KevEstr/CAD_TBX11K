# CAD-TBX11K

Prototipo de **diagnóstico asistido por computador (CAD)** para apoyar la detección de tuberculosis pulmonar en radiografías de tórax. El proyecto implementa un flujo de segmentación pulmonar, extracción de características y clasificación con aprendizaje automático tradicional, utilizando el conjunto de datos TBX11K.

> **Aplicación desplegada:** [http://3.134.162.144/](http://3.134.162.144/)

El CAD recibe una radiografía y genera una predicción mediante los modelos entrenados del proyecto. Es un prototipo experimental: no constituye un diagnóstico médico, no sustituye la valoración profesional y no ha sido validado clínicamente.

## ¿Qué hace el proyecto?

- Segmenta los pulmones con PSPNet preentrenado de `torchxrayvision`.
- Extrae descriptores de textura y características HOG del campo pulmonar.
- Selecciona características y compara cinco clasificadores: regresión logística, SVM, árbol de decisión, Random Forest y XGBoost.
- Evalúa los modelos mediante validación cruzada anidada y métricas por clase.
- Permite generar predicciones para una radiografía individual o para un lote de imágenes.

## Resultados y datos

El modelado utiliza **8.256 radiografías etiquetadas** de TBX11K:

| Clase | Etiqueta | Imágenes originales | Imágenes utilizadas |
|---|---:|---:|---:|
| `health` | 0 | 3.800 | 3.800 |
| `sick` | 1 | 3.800 | 3.656 |
| `tb` | 2 | 800 | 800 |
| **Total** | | **8.400** | **8.256** |

Las 144 imágenes excluidas corresponden a registros de la clase `sick` identificados en la lista canónica de exclusión `src/dedup_drop_names.csv`. El conjunto `test` descargado no tiene etiquetas públicas y no se utiliza en la evaluación principal. TBX11K incluye 1.211 cajas de lesión en 799 imágenes de la clase `tb`.

**Fuentes del conjunto de datos:** [TBX11K](https://mmcheng.net/tb/) · [Copia en Kaggle](https://www.kaggle.com/datasets/usmanshams/tbx-11).

## Flujo de procesamiento

1. **Segmentación pulmonar:** PSPNet identifica los pulmones. Las máscaras se combinan y se dilatan 12 píxeles para incluir regiones periféricas.
2. **Preprocesamiento:** conversión a escala de grises, filtrado bilateral, CLAHE y recorte torácico.
3. **Extracción de características:** se calculan descriptores de textura (GLCM, LBP y Gabor) y HOG. El conjunto principal contiene 1.832 características: 68 de textura y 1.764 HOG.
4. **Selección y clasificación:** se seleccionan 300 características y se comparan cinco clasificadores con validación cruzada anidada (5 particiones externas y 3 internas).
5. **Evaluación:** se calculan, entre otras métricas, exactitud balanceada, F1 macro, sensibilidad para TB, AUC-ROC y average precision.

La selección de características se ajusta dentro de los datos de entrenamiento de cada partición externa. Los modelos finales se reajustan con las 8.256 imágenes y se guardan en `models/`.

## Estructura del repositorio

```text
cad-tbx11k/
├── src/                 # carga de datos, segmentación, preprocesamiento,
│                        # características, selección, modelado y visualización
├── notebooks/           # flujo experimental del 01 al 10
├── scripts/             # validación, auditorías, entrenamiento auxiliar y predicción
├── data/                # dataset y artefactos locales (excluidos de Git)
├── models/              # modelos entrenados (excluidos de Git)
├── reports/             # figuras y reportes
├── exports/             # archivos generados (excluidos de Git)
└── requirements.txt
```

## Instalación

Los siguientes comandos están escritos para PowerShell en Windows:

```powershell
python -m venv venv
.\venv\Scripts\activate

# PyTorch para CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

La primera ejecución de la segmentación descarga los pesos de PSPNet y requiere conexión a internet.

Descarga TBX11K y organiza los archivos de esta forma:

```text
data/TBX11K/
├── imgs/
│   ├── health/
│   ├── sick/
│   └── tb/
└── annotations/
    └── json/
```

## Cómo ejecutar los notebooks

La reproducción del proyecto se realiza desde **Jupyter Notebook**. No es necesario ejecutar los notebooks mediante comandos `nbconvert`.

1. Abre una terminal en la carpeta raíz del repositorio.
2. Activa el entorno virtual e inicia Jupyter Notebook:

   ```powershell
   .\\venv\\Scripts\\activate
   jupyter notebook
   ```

3. En la interfaz de Jupyter, abre la carpeta `notebooks/`.
4. Abre los archivos y ejecuta sus celdas en orden, desde la primera hasta la última. Espera a que cada notebook termine antes de pasar al siguiente.

Sigue este orden para mantener la secuencia de generación de datos y artefactos:

| Orden | Notebook | Etapa |
|---:|---|---|
| 1 | `01_Carga_Datos.ipynb` | Carga y preparación de datos |
| 2 | `02_Analisis_Exploratorio.ipynb` | Análisis exploratorio |
| 3 | `03_Preprocesamiento.ipynb` | Preprocesamiento |
| 4 | `04_Analisis_Regional.ipynb` | Análisis regional |
| 5 | `05_Extraccion_Caracteristicas.ipynb` | Segmentación y extracción de características |
| 6 | `06_Seleccion_Caracteristicas.ipynb` | Selección de características |
| 7 | `07_Desbalanceo.ipynb` | Tratamiento del desbalanceo |
| 8 | `08_Modelos_ML.ipynb` | Entrenamiento de clasificadores |
| 9 | `09_Evaluacion_Comparativa.ipynb` | Evaluación comparativa |
| 10 | `10_Interpretabilidad.ipynb` | Interpretabilidad |

El notebook 05 utiliza las máscaras cacheadas y ejecuta la validación de segmentación. Si necesitas generarlas previamente, ejecuta `scripts/precompute_masks.py` desde una terminal, como se indica en la sección de herramientas auxiliares.

## Usar el CAD desde la línea de comandos

Para predecir una radiografía:

```powershell
python scripts/predict.py --image ruta/a/radiografia.png
```

También se puede indicar un modelo, guardar la máscara o el preprocesamiento, generar un mapa de calor o procesar un CSV por lotes:

```powershell
python scripts/predict.py --image ruta.png --model "XGBoost"
python scripts/predict.py --image ruta.png --save-mask mascara.png --save-preprocessed thorax.png
python scripts/predict.py --image ruta.png --heatmap heatmap.png
python scripts/predict.py --csv data/imagenes.csv --out data/predicciones.csv
```

`models/08_manifest.json` registra el modelo seleccionado y `models/08_feature_schema.json` define el orden de las características. Si el CSV de entrada incluye una columna `label`, el script también calcula métricas y una matriz de confusión.

## Archivos generados

Entre los principales artefactos del flujo experimental se encuentran:

- `data/df_model.csv`: metadatos de las imágenes utilizadas.
- `data/features_tbx11k.csv`: características extraídas.
- `data/X_full.npy` y `data/y_full.npy`: matriz de características y etiquetas.
- `data/08_oof_predicciones.csv` y `data/08_oof_scores.npz`: predicciones y puntuaciones fuera de muestra.
- `data/08_cv_resumen.csv` y `data/08_mejores_params.json`: resumen de validación y mejores parámetros.
- `models/08_*.joblib`: modelos entrenados.
- `reports/`: figuras y reportes del análisis.

Los directorios `data/`, `models/` y `exports/` contienen archivos locales o generados y están excluidos de Git. Los notebooks, scripts y figuras de `reports/` se versionan.

## Alcance y limitaciones

- **Uso experimental:** el CAD es un prototipo de apoyo a la investigación; no está validado clínicamente ni debe utilizarse como diagnóstico autónomo.
- **Datos:** la evaluación principal se realiza sobre TBX11K. El conjunto `test` no tiene etiquetas públicas y no se dispone de una validación externa etiquetada en este proyecto.
- **Pacientes:** TBX11K no proporciona identificadores de paciente; la validación principal se realiza por imagen y no permite garantizar separación por paciente.
- **Segmentación:** la validación principal de la máscara utiliza cajas de lesión, no máscaras anatómicas de pulmón de referencia.
- **Contexto de adquisición:** las auditorías del proyecto identifican señales asociadas al contexto y a la adquisición de las radiografías. Por ello, el rendimiento no debe interpretarse como evidencia de que el clasificador se base exclusivamente en hallazgos pulmonares.
- **Umbral:** el umbral se estima a partir de validación cruzada y requiere evaluación independiente antes de considerarse generalizable.
