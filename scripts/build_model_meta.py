"""Genera `data/df_model.csv` (8.256 imágenes) de forma reproducible.

Orden del pipeline:

    01 → data/df_base.csv        (8.400 imágenes etiquetadas)
    05 → data/df_model.csv       (8.256 tras deduplicación exacta)
    06–10 → data/df_model.csv

La lista canónica de 144 duplicados exactos está en
`src/dedup_drop_names.csv` (todos de `sick`). Esa lista se obtuvo de la
deduplicación exacta de la matriz antigua y permite reproducir las 8.256
imágenes sin depender de `data/features_tbx11k.csv`.

Orden de preferencia:
1. `src/dedup_drop_names.csv` (canónico, reproducible).
2. Deduplicación sobre `data/features_tbx11k.csv` (si existe).
3. Hash MD5 exacto de imagen (respaldo; puede dar 8.274).

Uso:
    python scripts/build_model_meta.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def dedup_by_names(df_base: pd.DataFrame) -> pd.DataFrame | None:
    """Elimina los duplicados exactos listados en `src/dedup_drop_names.csv`."""
    path = ROOT / "src" / "dedup_drop_names.csv"
    if not path.exists():
        return None
    names = set(pd.read_csv(path)["image_name"])
    keep = ~df_base["image_path"].map(lambda p: Path(p).name).isin(names)
    return df_base.loc[keep].copy()


def dedup_by_features(df_base: pd.DataFrame) -> pd.DataFrame | None:
    path = DATA / "features_tbx11k.csv"
    if not path.exists():
        return None
    full = pd.read_csv(path)
    feature_cols = [c for c in full.columns if c not in {"image_path", "class_name", "label"}]
    dedup = full.drop_duplicates(subset=feature_cols, keep="first")
    return df_base[df_base["image_path"].isin(set(dedup["image_path"]))].copy()


def dedup_by_md5(df_base: pd.DataFrame) -> pd.DataFrame:
    hashes = []
    for p in df_base["image_path"]:
        with open(p, "rb") as f:
            hashes.append(hashlib.md5(f.read()).hexdigest())
    tmp = df_base.assign(_md5=hashes)
    return tmp.loc[~tmp["_md5"].duplicated()].drop(columns="_md5")


def main() -> None:
    df_base = pd.read_csv(DATA / "df_base.csv")
    print(f"df_base: {len(df_base)} | {df_base['class_name'].value_counts().to_dict()}")

    df_model = dedup_by_names(df_base)
    if df_model is None:
        print("AVISO: no existe src/dedup_drop_names.csv; se intenta con features.")
        df_model = dedup_by_features(df_base)
    if df_model is None:
        print("AVISO: no existe data/features_tbx11k.csv; se usa dedup por MD5.")
        df_model = dedup_by_md5(df_base)

    df_model = df_model.reset_index(drop=True)
    out = DATA / "df_model.csv"
    df_model[["image_path", "class_name", "label"]].to_csv(out, index=False)
    print(f"df_model: {len(df_model)} | {df_model['class_name'].value_counts().to_dict()}")
    print(f"Guardado en {out}")


if __name__ == "__main__":
    main()
