"""Utilidades para que los notebooks no fallen por artefactos ausentes.

Cada notebook declara qué archivos necesita y qué script los produce. Si falta
alguno, se ejecuta el productor una vez y se continúa. Así el orden real del
pipeline queda explícito y no hay dependencias silenciosas.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]


def ensure_artifacts(
    paths: Iterable[str | Path],
    command: list[str],
    description: str = "",
    auto: bool = True,
) -> list[Path]:
    """Comprueba que existan ``paths``; si falta alguno, ejecuta ``command``.

    Parameters
    ----------
    paths : iterable de str o Path
        Archivos que el notebook necesita.
    command : list de str
        Comando que los genera, p. ej.
        ``[sys.executable, "scripts/build_model_meta.py"]``.
    description : str
        Texto para el usuario.
    auto : bool
        Si es ``False``, lanza ``FileNotFoundError`` en lugar de ejecutar.
    """
    paths = [Path(p) for p in paths]
    missing = [p for p in paths if not p.exists()]
    if not missing:
        return paths

    print(f"Faltan {len(missing)} archivo(s): {', '.join(p.name for p in missing)}")
    if description:
        print(description)
    if not auto:
        raise FileNotFoundError(
            "Ejecuta manualmente: " + " ".join(command)
        )

    print("Ejecutando:", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)

    still_missing = [p for p in paths if not p.exists()]
    if still_missing:
        raise FileNotFoundError(
            "El comando terminó pero siguen faltando: "
            + ", ".join(str(p) for p in still_missing)
        )
    return paths
