"""
cad_tbx11k
==========
Framework de análisis exploratorio, preprocesamiento y extracción de características
para el dataset TBX11K de radiografías torácicas de tuberculosis.

Módulos
-------
data          : carga del dataset y construcción del DataFrame base.
preprocessing : cadena de preprocesamiento (Bilateral Filter + CLAHE).
features      : extractores de características (GLCM, LBP, HOG, intensidad).
visualization : figuras de publicación para cada etapa del análisis.

Uso rápido
----------
>>> from cad_tbx11k.data import setup_kaggle_credentials, download_dataset
>>> from cad_tbx11k.data import get_data_root, build_base_dataframe
>>> from cad_tbx11k.preprocessing import preprocess
>>> from cad_tbx11k.features import extract_all_features
>>> from cad_tbx11k.visualization import set_publication_style
"""

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

from .preprocessing import (
    IMG_SIZE,
    preprocess,
    preprocess_steps,
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

from .visualization import (
    set_publication_style,
    plot_dataset_distribution,
    plot_image_metadata,
    plot_sample_images,
    plot_intensity_histograms,
    compute_intensity_stats,
    plot_intensity_metrics,
    plot_preprocessing_steps,
    plot_histogram_comparison,
    build_preprocessing_stats_table,
    plot_regions_on_images,
    compute_regional_stats,
    plot_regional_heatmap,
    plot_regional_distributions,
    build_regional_comparison_table,
    plot_glcm_distributions,
    plot_lbp_histograms,
    plot_feature_summary_table,
)

__version__ = "1.0.0"
__author__  = "CAD-TBX11K Pipeline"
