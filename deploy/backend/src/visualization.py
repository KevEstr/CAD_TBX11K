"""Funciones de visualización para los notebooks del pipeline."""

from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy.stats import entropy as scipy_entropy

from .data import LABEL_MAP
from .preprocessing import preprocess, preprocess_steps
from .features import REGIONS, GLCM_PROPS

# ---------------------------------------------------------------------------
# Paleta académica global — diferenciable en escala de grises e impresión
# ---------------------------------------------------------------------------

PALETTE_ACADEMIC = {
    "health": "#2166AC",   # azul oscuro
    "sick":   "#4DAC26",   # verde oscuro
    "tb":     "#D01C8B",   # magenta
}

PALETTE_FULL = {
    "health": "#2166AC",
    "sick":   "#4DAC26",
    "tb":     "#D01C8B",
    "test":   "#B0B0B0",
    "extra":  "#D9D9D9",
}


# ---------------------------------------------------------------------------
# Estilo de publicación
# ---------------------------------------------------------------------------

def set_publication_style() -> None:
    """Configura el estilo global de las figuras."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi":        150,
            "savefig.dpi":       300,
            "font.family":       "DejaVu Sans",
            "font.size":         11,
            "axes.titlesize":    12,
            "axes.labelsize":    11,
            "xtick.labelsize":   10,
            "ytick.labelsize":   10,
            "legend.fontsize":   10,
            "axes.spines.top":   False,
            "axes.spines.right": False,
            "figure.facecolor":  "white",
            "axes.facecolor":    "white",
        }
    )


# ---------------------------------------------------------------------------
# Notebook 01 — Distribución del dataset
# ---------------------------------------------------------------------------

def plot_dataset_distribution(
    resumen_df: pd.DataFrame,
    df_base: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """Grafica la distribución de imágenes por clase."""
    set_publication_style()

    por_carpeta = resumen_df.groupby("carpeta")["imagenes"].sum().sort_values(ascending=False)
    colores_barras = [PALETTE_FULL.get(c, "#B0B0B0") for c in por_carpeta.index]
    counts = df_base["class_name"].value_counts().reindex(["health", "sick", "tb"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Panel izquierdo: dataset completo ---
    bars = axes[0].bar(
        por_carpeta.index, por_carpeta.values,
        color=colores_barras, edgecolor="white", linewidth=0.6, width=0.6,
    )
    y_max = por_carpeta.max()
    for bar, cnt in zip(bars, por_carpeta.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            cnt + y_max * 0.02,
            f"{cnt:,}", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#333333",
        )
    axes[0].set_title("Full TBX11K Dataset", fontsize=12, fontweight="bold", pad=55)
    axes[0].set_ylabel("Number of Images")
    axes[0].set_xlabel("Partition")
    axes[0].set_ylim(0, y_max * 1.22)
    axes[0].tick_params(axis="x", rotation=15)

    legend_left = [
        Patch(facecolor="#2166AC", edgecolor="white", label="Health (labeled)"),
        Patch(facecolor="#4DAC26", edgecolor="white", label="Sick (labeled)"),
        Patch(facecolor="#D01C8B", edgecolor="white", label="TB (labeled)"),
        Patch(facecolor="#B0B0B0", edgecolor="white", label="Unlabeled / External"),
    ]
    axes[0].legend(
        handles=legend_left, loc="upper center",
        bbox_to_anchor=(0.5, 1.28), ncol=2,
        fontsize=8, frameon=True, framealpha=0.9, edgecolor="#cccccc",
    )

    # --- Panel derecho: subconjunto etiquetado ---
    class_colors = [PALETTE_ACADEMIC[c] for c in counts.index]
    bars2 = axes[1].bar(
        counts.index, counts.values,
        color=class_colors, edgecolor="white", linewidth=0.6, width=0.5,
    )
    y_max2 = counts.max()
    for bar, (cls, cnt) in zip(bars2, counts.items()):
        pct = 100 * cnt / counts.sum()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            cnt + y_max2 * 0.02,
            f"{cnt:,}\n({pct:.1f}%)", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#333333",
        )
    axes[1].set_title("Labeled Subset (Train + Val)", fontsize=12, fontweight="bold", pad=55)
    axes[1].set_ylabel("Number of Images")
    axes[1].set_xlabel("Class")
    axes[1].set_ylim(0, y_max2 * 1.28)

    legend_right = [
        Patch(facecolor="#2166AC", edgecolor="white", label="Health (0)"),
        Patch(facecolor="#4DAC26", edgecolor="white", label="Sick (1)"),
        Patch(facecolor="#D01C8B", edgecolor="white", label="TB (2)"),
    ]
    axes[1].legend(
        handles=legend_right, loc="upper center",
        bbox_to_anchor=(0.5, 1.28), ncol=3,
        fontsize=8, frameon=True, framealpha=0.9, edgecolor="#cccccc",
    )

    fig.suptitle("Image Distribution — TBX11K Dataset", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    # fig.savefig(save_path, dpi=300, bbox_inches="tight")  # descomentar para guardar
    plt.show()


def plot_image_metadata(df_base: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """Muestra modos de color y resoluciones frecuentes."""
    set_publication_style()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Modos de color
    modo_counts = df_base["modo"].value_counts()
    axes[0].bar(modo_counts.index, modo_counts.values, color="#2166AC", edgecolor="white", width=0.5)
    for i, (idx, val) in enumerate(modo_counts.items()):
        axes[0].text(i, val + modo_counts.max() * 0.02, str(val), ha="center", fontweight="bold")
    axes[0].set_title("Modos de color", fontweight="bold")
    axes[0].set_ylabel("Number of Images")
    axes[0].set_xlabel("PIL Mode")
    axes[0].set_ylim(0, modo_counts.max() * 1.15)

    # Resoluciones más frecuentes
    res_df = (
        df_base.groupby(["ancho", "alto"]).size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(8)
    )
    res_labels = [f"{int(r.ancho)}×{int(r.alto)}" for _, r in res_df.iterrows()]
    axes[1].barh(res_labels[::-1], res_df["count"].values[::-1], color="#4DAC26", edgecolor="white")
    axes[1].set_title("Most Frequent Resolutions", fontweight="bold")
    axes[1].set_xlabel("Number of Images")

    fig.suptitle("Image Metadata — Labeled Dataset", fontweight="bold")
    plt.tight_layout()
    # fig.savefig(save_path, dpi=300, bbox_inches="tight")  # descomentar para guardar
    plt.show()


# ---------------------------------------------------------------------------
# Notebook 02 — Análisis exploratorio
# ---------------------------------------------------------------------------

def plot_sample_images(
    df_base: pd.DataFrame,
    n: int = 4,
    random_state: int = 42,
    save_path: Optional[str] = None,
) -> None:
    """Muestra ejemplos de radiografías por clase."""
    set_publication_style()

    fig, axes = plt.subplots(3, n, figsize=(3.5 * n, 10))

    for row, class_name in enumerate(["health", "sick", "tb"]):
        muestras = df_base[df_base["class_name"] == class_name].sample(n, random_state=random_state)
        for col, (_, sample) in enumerate(muestras.iterrows()):
            img = cv2.imread(sample["image_path"], cv2.IMREAD_GRAYSCALE)
            axes[row][col].imshow(img, cmap="gray", aspect="equal")
            axes[row][col].axis("off")
            if col == 0:
                axes[row][col].set_ylabel(
                    class_name.upper(), fontsize=12, fontweight="bold",
                    rotation=90, labelpad=8,
                )

    fig.suptitle(
        "Radiografías de ejemplo por clase — sin preprocesamiento",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    # fig.savefig(save_path, dpi=300, bbox_inches="tight")  # descomentar para guardar
    plt.show()


def plot_intensity_histograms(
    df_base: pd.DataFrame,
    n_hist: int = 80,
    preprocessed: bool = False,
    random_state: int = 42,
    save_path: Optional[str] = None,
) -> None:
    """Histograma de intensidad promedio por clase."""
    set_publication_style()

    state = "Preprocesada (BF + CLAHE)" if preprocessed else "Original (sin preprocesamiento)"
    fig, ax = plt.subplots(figsize=(10, 4))

    n_used = n_hist

    for class_name, color in PALETTE_ACADEMIC.items():
        disponibles = df_base[df_base["class_name"] == class_name]
        n = min(n_hist, len(disponibles))
        n_used = min(n_used, n)
        muestras = disponibles.sample(n, random_state=random_state)

        hist_acum = np.zeros(256)
        valid = 0
        for _, row in muestras.iterrows():
            img = preprocess(row["image_path"]) if preprocessed else cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            h = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
            hist_acum += h / h.sum()
            valid += 1

        if valid > 0:
            hist_acum /= valid
        ax.plot(hist_acum, color=color, label=class_name.capitalize(), linewidth=1.8, alpha=0.85)
        ax.fill_between(range(256), hist_acum, alpha=0.08, color=color)

    ax.set_title(
        f"Histograma de intensidad promedio por clase — {state} (n={n_used} por clase)",
        fontweight="bold",
    )
    ax.set_xlabel("Gray Level (0 = black, 255 = white)")
    ax.set_ylabel("Mean Relative Frequency")
    ax.legend(title="Class", frameon=True, loc="upper center",
              bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    # fig.savefig(save_path, dpi=300, bbox_inches="tight")  # descomentar para guardar
    plt.show()

def compute_intensity_stats(
    df_base: pd.DataFrame,
    n_hist: int = 80,
    random_state: int = 42,
) -> pd.DataFrame:
    """Calcula métricas de intensidad por clase."""
    stats: list[dict] = []

    for class_name in ["health", "sick", "tb"]:
        disponibles = df_base[df_base["class_name"] == class_name]
        n = min(n_hist, len(disponibles))
        muestras = disponibles.sample(n, random_state=random_state)

        for _, row in muestras.iterrows():
            img = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            h = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
            h_norm = h / (h.sum() + 1e-9)
            stats.append({
                "class_name": class_name,
                "mean":       float(np.mean(img)),
                "std":        float(np.std(img)),
                "contrast":   float(int(img.max()) - int(img.min())),
                "entropy":    float(scipy_entropy(h_norm + 1e-9)),
            })

    return pd.DataFrame(stats)

def plot_intensity_metrics(
    stats_df: pd.DataFrame,
    title_suffix: str = "Imágenes originales",
    save_path: Optional[str] = None,
) -> None:
    """Panel de violín, caja y puntos para las métricas de intensidad."""
    set_publication_style()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    labels = {
        "mean":     "Mean Intensity (0–255)",
        "std":      "Intensity Standard Deviation",
        "contrast": "Contraste (máx − mín)",
        "entropy":  "Histogram Entropy (nats)",
    }

    for ax, col in zip(axes, ["mean", "std", "contrast", "entropy"]):
        _violin_box_strip(ax, stats_df, col)
        ax.set_title(labels[col], fontweight="bold")
        ax.set_xlabel("Class")
        ax.set_ylabel(labels[col])

    fig.suptitle(
        f"Intensity Metrics by Class — {title_suffix}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    # fig.savefig(save_path, dpi=300, bbox_inches="tight")  # descomentar para guardar
    plt.show()


# ---------------------------------------------------------------------------
# Notebook 03 — Preprocesamiento
# ---------------------------------------------------------------------------

def plot_preprocessing_steps(
    df_base: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """Muestra los pasos de preprocesamiento por clase."""
    set_publication_style()

    col_titles = ["Original (redimensionada)", "Filtro bilateral", "BF + CLAHE"]
    fig, axes = plt.subplots(3, 3, figsize=(11, 11))

    for row, class_name in enumerate(["health", "sick", "tb"]):
        path = df_base[df_base["class_name"] == class_name].iloc[0]["image_path"]
        steps = preprocess_steps(path)
        if steps is None:
            continue
        imgs = [steps["original"], steps["bilateral"], steps["clahe"]]

        for col, img in enumerate(imgs):
            axes[row][col].imshow(img, cmap="gray", aspect="equal")
            axes[row][col].axis("off")
            if row == 0:
                axes[row][col].set_title(col_titles[col], fontsize=11, fontweight="bold")
            if col == 0:
                axes[row][col].set_ylabel(
                    class_name.upper(), fontsize=11, fontweight="bold",
                    rotation=90, labelpad=8,
                )

    fig.suptitle(
        "Effect of Preprocessing Steps by Class",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    # fig.savefig(save_path, dpi=300, bbox_inches="tight")  # descomentar para guardar
    plt.show()


def plot_histogram_comparison(
    df_base: pd.DataFrame,
    class_name: str = "tb",
    save_path: Optional[str] = None,
) -> None:
    """Compara histogramas antes y después del preprocesamiento."""
    set_publication_style()

    path = df_base[df_base["class_name"] == class_name].iloc[0]["image_path"]
    steps = preprocess_steps(path)
    if steps is None:
        return

    original  = steps["original"]
    processed = steps["clahe"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, img, title in zip(
        axes,
        [original, processed],
        ["Original (256×256)", "Preprocesada (BF + CLAHE)"],
    ):
        h = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
        ax.plot(h / h.sum(), color="#1565C0", linewidth=1.6)
        ax.fill_between(range(256), h / h.sum(), alpha=0.15, color="#1565C0")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Gray Level")
        ax.set_ylabel("Relative Frequency")
        ax.set_xlim(0, 255)

    fig.suptitle(
        f"Histogram Comparison Before and After Preprocessing — Class: {class_name.upper()}",
        fontweight="bold",
    )
    plt.tight_layout()
    # fig.savefig(save_path, dpi=300, bbox_inches="tight")  # descomentar para guardar
    plt.show()


def build_preprocessing_stats_table(
    df_base: pd.DataFrame,
    n_sample: int = 60,
    random_state: int = 42,
) -> pd.DataFrame:
    """Tabla de métricas antes y después del preprocesamiento."""
    metrics = ["mean", "std", "contrast", "entropy"]
    rows: list[dict] = []

    for class_name in ["health", "sick", "tb"]:
        muestras = df_base[df_base["class_name"] == class_name].sample(n_sample, random_state=random_state)
        for stage, use_prep in [("Original", False), ("Preprocesada", True)]:
            acum = {m: [] for m in metrics}
            for _, row in muestras.iterrows():
                img = preprocess(row["image_path"]) if use_prep else cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                h = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
                h_norm = h / (h.sum() + 1e-9)
                acum["mean"].append(float(np.mean(img)))
                acum["std"].append(float(np.std(img)))
                acum["contrast"].append(float(int(img.max()) - int(img.min())))
                acum["entropy"].append(float(scipy_entropy(h_norm + 1e-9)))

            rows.append({
                "Class":    class_name,
                "Stage":    stage,
                "mean":     round(float(np.mean(acum["mean"])), 2),
                "std":      round(float(np.mean(acum["std"])), 2),
                "contrast": round(float(np.mean(acum["contrast"])), 2),
                "entropy":  round(float(np.mean(acum["entropy"])), 4),
            })

    return pd.DataFrame(rows).set_index(["Class", "Stage"])


# ---------------------------------------------------------------------------
# Notebook 04 — Análisis regional
# ---------------------------------------------------------------------------

def plot_regions_on_images(
    df_base: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """Dibuja las regiones anatómicas sobre imágenes preprocesadas."""
    set_publication_style()

    region_colors = [
        (0, 180, 0), (0, 140, 0),
        (200, 120, 0), (180, 100, 0),
        (0, 0, 200), (0, 0, 160),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))

    for ax, class_name in zip(axes, ["health", "sick", "tb"]):
        path = df_base[df_base["class_name"] == class_name].iloc[0]["image_path"]
        img_prep = preprocess(path)
        if img_prep is None:
            continue
        img_rgb = cv2.cvtColor(img_prep, cv2.COLOR_GRAY2BGR)

        for (name, (x, y, w, h)), color in zip(REGIONS.items(), region_colors):
            cv2.rectangle(img_rgb, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img_rgb, name, (x + 4, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        ax.imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB), aspect="equal")
        ax.set_title(class_name.upper(), fontsize=12, fontweight="bold")
        ax.axis("off")

    fig.suptitle(
        "Regiones anatómicas sobre imágenes preprocesadas (BF + CLAHE)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    # fig.savefig(save_path, dpi=300, bbox_inches="tight")  # descomentar para guardar
    plt.show()


def compute_regional_stats(
    df_base: pd.DataFrame,
    n_reg: int = 60,
    random_state: int = 42,
) -> pd.DataFrame:
    """Calcula la intensidad media por región anatómica."""
    reg_stats: list[dict] = []

    for class_name in ["health", "sick", "tb"]:
        muestras = df_base[df_base["class_name"] == class_name].sample(n_reg, random_state=random_state)
        for _, row in muestras.iterrows():
            img = preprocess(row["image_path"])
            if img is None:
                continue
            entry: dict = {"class_name": class_name}
            for name, (x, y, w, h) in REGIONS.items():
                entry[f"{name}_mean"] = float(np.mean(img[y: y + h, x: x + w]))
            reg_stats.append(entry)

    return pd.DataFrame(reg_stats)


def plot_regional_heatmap(
    reg_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """Mapa de calor de intensidad por región y clase."""
    set_publication_style()

    region_cols = [f"{r}_mean" for r in REGIONS]
    media_regional = reg_df.groupby("class_name")[region_cols].mean()
    media_regional.columns = [c.replace("_mean", "") for c in media_regional.columns]

    fig, ax = plt.subplots(figsize=(10, 3.5))
    sns.heatmap(
        media_regional,
        annot=True, fmt=".1f",
        cmap="Blues",
        linewidths=0.6, linecolor="white",
        cbar_kws={"label": "Mean Intensity (0–255)"},
        ax=ax,
    )
    ax.set_title("Mean Intensity by Anatomical Region and Class", fontweight="bold")
    ax.set_ylabel("Class")
    ax.set_xlabel("Region")
    plt.tight_layout()
    # fig.savefig(save_path, dpi=300, bbox_inches="tight")  # descomentar para guardar
    plt.show()


def plot_regional_distributions(reg_df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """Distribuciones de intensidad por región anatómica."""
    set_publication_style()
    region_cols = [f"{r}_mean" for r in REGIONS]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    axes = axes.flatten()
    for ax, col in zip(axes, region_cols):
        _violin_box_strip(ax, reg_df, col)
        ax.set_title(col.replace("_mean", ""), fontsize=12, fontweight="bold")
        ax.set_xlabel("Class")
        ax.set_ylabel("Mean Intensity (0–255)")
    fig.suptitle("Mean Intensity Distribution by Anatomical Region", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    # fig.savefig(save_path, dpi=300, bbox_inches="tight")  # descomentar para guardar
    plt.show()


def build_regional_comparison_table(reg_df: pd.DataFrame) -> pd.DataFrame:
    """Tabla de intensidad media ± desviación por región."""
    region_cols = [f"{r}_mean" for r in REGIONS]
    mean_df = reg_df.groupby("class_name")[region_cols].mean().round(1)
    std_df  = reg_df.groupby("class_name")[region_cols].std().round(1)

    tabla = mean_df.copy().astype(str)
    for col in region_cols:
        tabla[col] = mean_df[col].astype(str) + " ± " + std_df[col].astype(str)

    tabla.columns = [c.replace("_mean", "") for c in tabla.columns]
    return tabla


# ---------------------------------------------------------------------------
# Notebook 05 — Características extraídas
# ---------------------------------------------------------------------------

def plot_glcm_distributions(features_df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """Distribuciones de las propiedades GLCM por clase."""
    set_publication_style()
    glcm_mean_cols = [f"glcm_{p}_mean" for p in GLCM_PROPS]
    labels = {"glcm_contrast_mean": "Contraste", "glcm_correlation_mean": "Correlación",
               "glcm_energy_mean": "Energía", "glcm_homogeneity_mean": "Homogeneidad",
               "glcm_dissimilarity_mean": "Disimilitud"}
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    for ax, col in zip(axes, glcm_mean_cols):
        _violin_box_strip(ax, features_df, col)
        ax.set_title(labels[col], fontsize=11, fontweight="bold")
        ax.set_xlabel("Class")
        ax.set_ylabel(labels[col])
    fig.suptitle("GLCM Properties (Mean over 4 Angles) — Distribution by Class", fontsize=13, fontweight="bold")
    plt.tight_layout()
    # fig.savefig(save_path, dpi=300, bbox_inches="tight")  # descomentar para guardar
    plt.show()


def plot_lbp_histograms(features_df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """Histograma LBP promedio por clase."""
    set_publication_style()
    lbp_cols = [c for c in features_df.columns if c.startswith("lbp_")]
    fig, ax = plt.subplots(figsize=(11, 5))
    for class_name, color in PALETTE_ACADEMIC.items():
        mean_hist = features_df[features_df["class_name"] == class_name][lbp_cols].mean().values
        ax.plot(mean_hist, color=color, label=class_name.capitalize(), linewidth=2)
        ax.fill_between(range(len(lbp_cols)), mean_hist, alpha=0.1, color=color)
    ax.set_xlabel("LBP Bin (uniform patterns)")
    ax.set_ylabel("Mean Normalized Frequency")
    ax.set_title("LBP — histograma promedio por clase (P=24, R=3, uniforme)",
                 fontweight="bold", pad=45)
    ax.legend(title="Class", frameon=True, loc="upper center",
              bbox_to_anchor=(0.5, 1.18), ncol=3, fontsize=9, edgecolor="#cccccc")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    # fig.savefig(save_path, dpi=300, bbox_inches="tight")  # descomentar para guardar
    plt.show()

def plot_feature_summary_table(features_df: pd.DataFrame) -> pd.DataFrame:
    """Tabla del número de características por grupo."""
    from .features import get_feature_groups

    grupos = get_feature_groups(features_df)
    rows = [
        {"Group": nombre, "Features": len(cols)}
        for nombre, cols in grupos.items()
    ]
    rows.append({"Group": "TOTAL", "Features": sum(len(v) for v in grupos.values())})
    return pd.DataFrame(rows).set_index("Group")


# ---------------------------------------------------------------------------
# Helper interno: violín + caja + puntos
# ---------------------------------------------------------------------------

def _violin_box_strip(ax: plt.Axes, data: pd.DataFrame, col: str) -> None:
    """Dibuja un violín con caja y puntos en un eje."""
    sns.violinplot(
        data=data, x="class_name", y=col,
        order=["health", "sick", "tb"], palette=PALETTE_ACADEMIC,
        inner=None, linewidth=1.2, ax=ax,
    )
    sns.boxplot(
        data=data, x="class_name", y=col,
        order=["health", "sick", "tb"], width=0.15,
        boxprops=dict(facecolor="white", zorder=2),
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=2, alpha=0.4),
        ax=ax,
    )
    sns.stripplot(
        data=data, x="class_name", y=col,
        order=["health", "sick", "tb"],
        color="black", alpha=0.2, size=2, jitter=True, ax=ax,
    )
