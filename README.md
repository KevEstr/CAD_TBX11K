# cad-tbx11k

Pipeline de análisis exploratorio sobre el dataset TBX11K para el proyecto de diagnóstico asistido de tuberculosis (CAD).

## ¿Qué hay acá?

`CAD_TBX11K_Pipeline.ipynb` — notebook principal. Cubre la primera etapa del proyecto:

- Carga del dataset y revisión de estructura (clases, balance, dimensiones, imágenes corruptas)
- Visualización de ejemplos por clase
- Exploración de histogramas de intensidad
- Preprocesamiento con Bilateral Filter + CLAHE
- Análisis por regiones anatómicas del pulmón

## Dataset

[TBX11K](https://mmcheng.net/tb/) — 11,200 radiografías de tórax con 3 clases: `health`, `sick`, `tb`.

Estructura esperada:
```
data/TBX11K/
  imgs/
    health/
    sick/
    tb/
```

## Dependencias

```
opencv-python
numpy
pandas
matplotlib
seaborn
Pillow
scipy
```

## Referencias

- `TB_detection` — estructura de carga del dataset
- `tbx11k-ensemble` — parámetros de preprocesamiento (BF + CLAHE)
- `CAD_Analisis_Exploratorio` — métricas de intensidad y análisis por regiones
