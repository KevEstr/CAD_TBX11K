"""Sincroniza el CAD con el modelo y el código actuales.

Uso:
    python scripts/sync_cad.py

Copia los módulos de src/, el modelo seleccionado del notebook 08, el esquema,
el manifiesto y el umbral a deploy/backend.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy" / "backend"
SRC = ROOT / "src"
MODELS = ROOT / "models"


def main() -> None:
    manifest = json.loads((MODELS / "08_manifest.json").read_text(encoding="utf-8"))
    seleccionado = manifest["seleccionado_cv"]
    modelo = MODELS / manifest["modelos"][seleccionado]

    (DEPLOY / "src").mkdir(parents=True, exist_ok=True)
    (DEPLOY / "models").mkdir(parents=True, exist_ok=True)

    for archivo in SRC.glob("*.py"):
        shutil.copy2(archivo, DEPLOY / "src" / archivo.name)

    for archivo in [
        modelo,
        MODELS / "08_manifest.json",
        MODELS / "08_feature_schema.json",
        MODELS / "08_umbral_tb.json",
    ]:
        shutil.copy2(archivo, DEPLOY / "models" / archivo.name)

    print("CAD sincronizado:")
    print(f"  modelo: {modelo.name}")
    print(f"  src: {SRC}")
    print(f"  destino: {DEPLOY}")


if __name__ == "__main__":
    main()
