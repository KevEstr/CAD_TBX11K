# cad-tbx11k

Pipeline de análisis exploratorio, preprocesamiento y extracción de características sobre el dataset **TBX11K**, como parte del proyecto de diagnóstico asistido por computador (CAD) para tuberculosis a partir de radiografías de tórax.

## Estructura

```text
cad-tbx11k/
│
├── notebooks/
│   ├── 01_Carga_Datos.ipynb
│   │   └── Descarga, exploración de la estructura del dataset
│   │       y construcción del DataFrame base.
│   │
│   ├── 02_Analisis_Exploratorio.ipynb
│   │   └── Visualización de radiografías, histogramas
│   │       y métricas de intensidad.
│   │
│   ├── 03_Preprocesamiento.ipynb
│   │   └── Bilateral Filter (BF), CLAHE y comparación
│   │       visual y estadística del preprocesamiento.
│   │
│   ├── 04_Analisis_Regional.ipynb
│   │   └── División de las radiografías en seis regiones
│   │       anatómicas, heatmaps y análisis de distribuciones.
│   │
│   └── 05_Extraccion_Caracteristicas.ipynb
│       └── Extracción de características GLCM, LBP, HOG
│           e intensidad global y regional.
│
├── requirements.txt
├── .gitignore
└── README.md
```

Los notebooks constituyen actualmente el núcleo del proyecto. Cada uno contiene las operaciones necesarias para desarrollar su respectiva etapa del análisis.

El acceso al dataset se realiza mediante `kagglehub`. La variable `DATA_ROOT` utilizada en los notebooks apunta a la ubicación local de la copia descargada y almacenada en la caché de Kaggle.

---

## Dataset

El proyecto utiliza **TBX11K**, un dataset de radiografías de tórax desarrollado para investigación en detección de tuberculosis.

**Fuente:** [TBX11K](https://mmcheng.net/tb/)

El conjunto utilizado en este pipeline corresponde a **8.400 radiografías etiquetadas** correspondientes a los conjuntos de entrenamiento y validación.

Las imágenes se organizan en tres clases:

| Clase    | Etiqueta | Descripción                                            |
| -------- | -------: | ------------------------------------------------------ |
| `health` |        0 | Persona sin tuberculosis                               |
| `sick`   |        1 | Persona enferma, pero no clasificada como tuberculosis |
| `tb`     |        2 | Tuberculosis                                           |

La descarga del dataset se realiza automáticamente en el notebook `01_Carga_Datos.ipynb` mediante `kagglehub`.

### Credenciales de Kaggle

Para realizar la descarga es necesario contar con las credenciales de Kaggle configuradas localmente.

El archivo `kaggle.json` **no debe incluirse en el repositorio**. Se encuentra excluido mediante `.gitignore`.

---

## Pipeline de análisis

El flujo actual del proyecto está organizado en cinco etapas:

```text
TBX11K
   │
   ▼
01. Carga y construcción de metadata
   │
   ▼
02. Análisis exploratorio
   │
   ▼
03. Preprocesamiento
   │
   ├── Bilateral Filter
   └── CLAHE
   │
   ▼
04. Análisis regional
   │
   └── División en 6 regiones anatómicas
   │
   ▼
05. Extracción de características
   │
   ├── GLCM
   ├── LBP
   ├── HOG
   ├── Intensidad global
   └── Intensidad regional
   │
   ▼
Matriz de características
   │
   ▼
Etapa posterior de modelado ML
```

### 01. Carga de datos

`01_Carga_Datos.ipynb`

* Descarga del dataset mediante `kagglehub`.
* Exploración de la estructura de directorios.
* Recorrido de las imágenes.
* Identificación de las clases.
* Construcción del DataFrame base.
* Registro de rutas y etiquetas.
* Caracterización inicial de las imágenes.

### 02. Análisis exploratorio

`02_Analisis_Exploratorio.ipynb`

Incluye:

* Visualización de ejemplos de las diferentes clases.
* Revisión de dimensiones de las imágenes.
* Histogramas de intensidad.
* Estadísticos descriptivos.
* Métricas globales de intensidad.
* Comparación de las distribuciones entre clases.

### 03. Preprocesamiento

`03_Preprocesamiento.ipynb`

Se evalúan técnicas de procesamiento de imágenes orientadas a mejorar la representación de las radiografías:

* **Bilateral Filter (BF)** para reducción de ruido preservando bordes.
* **CLAHE** (*Contrast Limited Adaptive Histogram Equalization*) para realce local del contraste.
* Comparación visual entre imágenes originales y procesadas.
* Comparación de histogramas.
* Análisis del efecto del preprocesamiento sobre las características de intensidad.

### 04. Análisis regional

`04_Analisis_Regional.ipynb`

Las radiografías se dividen en **seis regiones anatómicas** para analizar la distribución espacial de las características de intensidad.

Incluye:

* División espacial de las imágenes.
* Cálculo de métricas por región.
* Visualización mediante heatmaps.
* Comparación de distribuciones regionales.
* Análisis de diferencias entre clases.

### 05. Extracción de características

`05_Extraccion_Caracteristicas.ipynb`

Se construye una matriz de características para cada radiografía.

Las características consideradas incluyen:

* **GLCM** (*Gray-Level Co-occurrence Matrix*).
* **LBP** (*Local Binary Patterns*).
* **HOG** (*Histogram of Oriented Gradients*).
* Características de intensidad global.
* Características de intensidad por región anatómica.

El resultado constituye la representación tabular de las radiografías que será utilizada en la siguiente etapa del proyecto.

---

## Setup

### 1. Crear entorno virtual

En Windows:

```bash
python -m venv venv
```

### 2. Activar el entorno

```bash
venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Registrar el kernel de Jupyter

```bash
python -m ipykernel install --user --name=cad-tbx11k
```

Posteriormente, seleccionar el kernel `cad-tbx11k` desde Jupyter Notebook o Visual Studio Code.

---

## Orden de ejecución

El flujo recomendado para reproducir el análisis completo es:

```text
01 → 02 → 03 → 04 → 05
```

Los notebooks están organizados por etapas y contienen la lógica correspondiente a cada análisis. El `DATA_ROOT` permite acceder directamente a la copia local del dataset descargada mediante `kagglehub`.

La etapa `05_Extraccion_Caracteristicas.ipynb` produce la matriz final de características, que constituye la entrada para la siguiente fase del proyecto: **análisis y entrenamiento de modelos de machine learning**.

---

## Salidas

Las principales salidas generadas durante el pipeline son:

```text
01_Carga_Datos
    └── DataFrame con rutas y etiquetas

02_Analisis_Exploratorio
    └── Visualizaciones y estadísticas descriptivas

03_Preprocesamiento
    └── Comparaciones de imágenes e histogramas

04_Analisis_Regional
    └── Características y distribuciones regionales

05_Extraccion_Caracteristicas
    └── Matriz final de características
```

Los archivos generados durante el análisis no forman parte actualmente de la estructura versionada del repositorio.

---

## Referencias

### TB_detection

[TB_detection](https://github.com/Krishni1307/TB_detection)

Pipeline de clasificación sobre TBX11K desarrollado con PyTorch. Se tomó como referencia la estrategia para recorrer la estructura de directorios del dataset y construir el DataFrame base con las rutas de las imágenes y sus respectivas etiquetas.

### tbx11k-ensemble

[tbx11k-ensemble](https://github.com/behera116/tbx11k-ensemble)

Implementación de un ensemble de modelos sobre TBX11K con preprocesamiento offline. Se tomó como referencia la configuración utilizada para el **Bilateral Filter** y **CLAHE** aplicados a las radiografías de tórax.

### CAD_Analisis_Exploratorio

Análisis exploratorio suministrado por el semillero de investigación sobre un dataset binario de tuberculosis. Se tomó como referencia la estrategia para:

* registrar dimensiones de las imágenes;
* calcular métricas de intensidad;
* realizar análisis exploratorio de las radiografías; y
* dividir las imágenes en regiones anatómicas para el análisis espacial.

---

## Próximas etapas

La matriz de características obtenida en este repositorio constituye la entrada para las siguientes etapas del proyecto:

```text
Extracción de características
          │
          ▼
Análisis de características
          │
          ▼
Reducción de dimensionalidad
          │
          ▼
Clustering / análisis de estructura
          │
          ▼
Modelos de Machine Learning
          │
          ▼
Evaluación del desempeño
```

La incorporación de estas etapas se realizará progresivamente a medida que avance el desarrollo del proyecto.
