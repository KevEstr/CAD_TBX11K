"""
cad_tbx11k
==========
Framework de análisis exploratorio, segmentación, preprocesamiento y
extracción de características para el dataset TBX11K de radiografías
torácicas de tuberculosis.

Módulos
-------
config            : variables de muestreo y protocolo compartidas.
data              : carga del dataset y construcción del DataFrame base.
lung_segmentation : segmentación pulmonar anatómica validada.
preprocessing     : recorte torácico + Bilateral Filter + CLAHE.
features          : textura, HOG, forma y contexto (con soporte de máscara).
modeling          : CV anidada, métricas, calibración y McNemar.
visualization     : figuras de publicación.

Uso rápido
----------
>>> from src.preprocessing import preprocess_with_mask
>>> from src.features import extract_features, get_feature_groups
>>> from src.modeling import nested_cv, binary_metrics
"""

from .config import (
    N_MUESTRA,
    N_INTENSIDAD,
    N_SEGUIMIENTO,
    N_REGIONAL,
    N_MODELO,
    CV_OUTER,
    CV_INNER,
    RANDOM_STATE,
    TARGET_SENSITIVITY,
)

from .data import (
    LABEL_MAP,
    PALETTE,
    VALID_EXTS,
    setup_kaggle_credentials,
    download_dataset,
    get_data_root,
    map_dataset_structure,
    build_base_dataframe,
)

from .lung_segmentation import (
    MaskQuality,
    body_mask,
    segment_lungs as segment_lungs_512,
    thorax_bbox,
    thorax_envelope,
)

from .preprocessing import (
    IMG_SIZE,
    MODES,
    preprocess,
    preprocess_steps,
    preprocess_with_mask,
    preprocess_bundle,
    apply_bilateral_filter,
    apply_clahe,
)

from .features import (
    REGIONS,
    extract_features,
    extract_all_features,
    verify_extractors,
    get_feature_groups,
    print_feature_summary,
)

# visualization module excluded (requires seaborn; not needed for deployment)

__version__ = "2.0.0"
__author__ = "CAD-TBX11K Pipeline"
