"""
cad_tbx11k.data
===============
Módulo de carga y construcción del DataFrame base del pipeline CAD-TBX11K.

Contiene las funciones responsables de configurar las credenciales de Kaggle,
descargar el dataset TBX11K y construir el DataFrame con rutas, etiquetas
y metadatos básicos de cada radiografía.

Documentación de referencia
---------------------------
- kagglehub : https://github.com/Kaggle/kagglehub
- PIL/Pillow : https://pillow.readthedocs.io/en/stable/
"""

import os
import shutil
import random
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image

warnings.filterwarnings("ignore")
random.seed(42)

# ---------------------------------------------------------------------------
# Constantes del dominio
# ---------------------------------------------------------------------------

LABEL_MAP: dict[str, int] = {"health": 0, "sick": 1, "tb": 2}
"""Mapeo de nombre de clase a etiqueta numérica."""

VALID_EXTS: tuple[str, ...] = (".png", ".jpg", ".jpeg")
"""Extensiones de imagen aceptadas."""

PALETTE: dict[str, str] = {
    "health": "#4CAF50",
    "sick": "#FF9800",
    "tb": "#F44336",
}
"""Paleta de color consistente para las tres clases a lo largo del pipeline."""


# ---------------------------------------------------------------------------
# Configuración de credenciales Kaggle
# ---------------------------------------------------------------------------

def setup_kaggle_credentials(kaggle_json: Optional[str] = None) -> None:
    """
    Configura las credenciales de Kaggle copiando ``kaggle.json``
    al directorio ``~/.kaggle/`` si aún no está presente.

    La función detecta automáticamente si el entorno es Google Colab
    y, en ese caso, solicita la subida manual del archivo.

    Parámetros
    ----------
    kaggle_json : str, opcional
        Ruta explícita al archivo ``kaggle.json``.  Si se omite, la función
        busca el archivo en la carpeta padre del directorio de trabajo y,
        si no lo encuentra, en el directorio de trabajo actual.

    Raises
    ------
    FileNotFoundError
        Si ``kaggle.json`` no se encuentra en ninguna ubicación conocida.

    Referencias
    -----------
    - Guía de API de Kaggle: https://www.kaggle.com/docs/api
    """
    # Detección de Google Colab
    try:
        import google.colab  # noqa: F401
        in_colab = True
    except ImportError:
        in_colab = False

    if in_colab:
        from google.colab import files  # type: ignore
        print("Sube tu kaggle.json:")
        files.upload()
        kaggle_json = "kaggle.json"
    elif kaggle_json is None:
        # Búsqueda automática relativa al directorio de trabajo
        candidate_parent = os.path.join(os.path.dirname(os.getcwd()), "kaggle.json")
        candidate_cwd = os.path.join(os.getcwd(), "kaggle.json")
        if os.path.exists(candidate_parent):
            kaggle_json = candidate_parent
        elif os.path.exists(candidate_cwd):
            kaggle_json = candidate_cwd

    if not kaggle_json or not os.path.exists(kaggle_json):
        raise FileNotFoundError(
            "No se encontró kaggle.json.\n"
            "Descárgalo desde kaggle.com → Settings → API → Create New Token\n"
            "y colócalo en la raíz del proyecto (cad-tbx11k/)."
        )

    kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    dest = os.path.join(kaggle_dir, "kaggle.json")
    if not os.path.exists(dest):
        shutil.copy(kaggle_json, dest)
        os.chmod(dest, 0o600)

    print("kaggle.json configurado correctamente.")


# ---------------------------------------------------------------------------
# Descarga del dataset
# ---------------------------------------------------------------------------

def download_dataset() -> str:
    """
    Descarga el dataset TBX11K desde Kaggle mediante ``kagglehub``
    y retorna la ruta local donde fue almacenado.

    La función utiliza la caché local de ``kagglehub``: si el dataset
    ya fue descargado previamente, la descarga se omite y se retorna
    directamente la ruta cacheada.

    Retorna
    -------
    str
        Ruta al directorio raíz del dataset descargado.

    Raises
    ------
    RuntimeError
        Si ``kagglehub`` no puede descargar el dataset.

    Referencias
    -----------
    - kagglehub: https://github.com/Kaggle/kagglehub
    - Dataset TBX11K en Kaggle: https://www.kaggle.com/datasets/usmanshams/tbx-11
    """
    import kagglehub  # type: ignore

    path = kagglehub.dataset_download("usmanshams/tbx-11")
    print(f"Dataset disponible en: {path}")
    return path


def get_data_root(download_path: str) -> str:
    """
    Construye y verifica la ruta al directorio ``TBX11K/`` dentro del
    directorio descargado por ``kagglehub``.

    Parámetros
    ----------
    download_path : str
        Ruta retornada por :func:`download_dataset`.

    Retorna
    -------
    str
        Ruta absoluta al directorio ``TBX11K/``.

    Raises
    ------
    FileNotFoundError
        Si el directorio ``TBX11K/`` no existe dentro de ``download_path``.
    """
    data_root = os.path.join(download_path, "TBX11K")
    if not os.path.exists(data_root):
        raise FileNotFoundError(
            f"No se encontró el directorio TBX11K en: {download_path}\n"
            "Verifica que la descarga finalizó correctamente."
        )
    print(f"DATA_ROOT : {data_root}")
    return data_root


# ---------------------------------------------------------------------------
# Exploración de estructura
# ---------------------------------------------------------------------------

def map_dataset_structure(data_root: str) -> pd.DataFrame:
    """
    Recorre el directorio ``imgs/`` del dataset y genera un resumen
    tabular con el conteo de imágenes por carpeta y subcarpeta.

    El dataset TBX11K contiene:

    - ``health``, ``sick``, ``tb`` — imágenes etiquetadas (train + val).
    - ``test``                      — sin etiqueta pública (challenge oficial).
    - ``extra``                     — datasets externos (Montgomery, Shenzhen, etc.).

    Parámetros
    ----------
    data_root : str
        Ruta al directorio raíz del dataset (``TBX11K/``).

    Retorna
    -------
    pd.DataFrame
        DataFrame con columnas ``['carpeta', 'subcarpeta', 'imagenes']``.
    """
    imgs_dir = Path(data_root) / "imgs"
    resumen: list[dict] = []

    for carpeta in sorted(imgs_dir.iterdir()):
        if not carpeta.is_dir():
            continue
        subcarpetas = [s for s in carpeta.iterdir() if s.is_dir()]
        if subcarpetas:
            for sub in sorted(subcarpetas):
                n = len([f for f in sub.iterdir() if f.suffix.lower() in VALID_EXTS])
                resumen.append({"carpeta": carpeta.name, "subcarpeta": sub.name, "imagenes": n})
        else:
            n = len([f for f in carpeta.iterdir() if f.suffix.lower() in VALID_EXTS])
            resumen.append({"carpeta": carpeta.name, "subcarpeta": "-", "imagenes": n})

    return pd.DataFrame(resumen)


# ---------------------------------------------------------------------------
# Construcción del DataFrame base
# ---------------------------------------------------------------------------

def build_base_dataframe(data_root: str) -> pd.DataFrame:
    """
    Carga el subconjunto etiquetado del dataset (``health``, ``sick``, ``tb``)
    y construye el DataFrame base con metadatos por imagen.

    Para cada imagen se registran:

    - ``image_path``  — ruta absoluta al archivo.
    - ``class_name``  — nombre de la clase (``'health'``, ``'sick'``, ``'tb'``).
    - ``label``       — etiqueta numérica según :data:`LABEL_MAP`.
    - ``ancho``       — ancho original en píxeles.
    - ``alto``        — alto original en píxeles.
    - ``modo``        — modo de color PIL (p. ej. ``'RGB'``, ``'L'``).

    Las imágenes corruptas o ilegibles se omiten con una advertencia.

    Parámetros
    ----------
    data_root : str
        Ruta al directorio raíz del dataset (``TBX11K/``).

    Retorna
    -------
    pd.DataFrame
        DataFrame base con una fila por imagen válida.

    Referencias
    -----------
    - PIL Image.open: https://pillow.readthedocs.io/en/stable/reference/Image.html
    """
    imgs_dir = Path(data_root) / "imgs"
    records: list[dict] = []
    skipped = 0

    for class_name, label in LABEL_MAP.items():
        class_dir = imgs_dir / class_name
        if not class_dir.is_dir():
            print(f"[AVISO] Carpeta no encontrada: {class_dir}")
            continue
        files = [f for f in class_dir.iterdir() if f.suffix.lower() in VALID_EXTS]
        for f in files:
            try:
                with Image.open(f) as img:
                    records.append(
                        {
                            "image_path": str(f),
                            "class_name": class_name,
                            "label":      label,
                            "ancho":      img.width,
                            "alto":       img.height,
                            "modo":       img.mode,
                        }
                    )
            except Exception:
                print(f"[ERROR] Imagen corrupta o ilegible: {f}")
                skipped += 1

    df = pd.DataFrame(records)
    print(
        f"Dataset cargado: {len(df)} imágenes en {df['class_name'].nunique()} clases "
        f"({skipped} omitidas por error)."
    )
    print(df["class_name"].value_counts().to_string())
    return df