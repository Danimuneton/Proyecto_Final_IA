"""Descarga los datos reales (SIATA + Open-Meteo) y genera los CSV de data/processed.

Uso:
    python scripts/download_data.py            # descarga de nuevo desde las fuentes
    python scripts/download_data.py --cache    # reutiliza data/raw si ya existe

Si falla la descarga, el proyecto sigue funcionando con los CSV versionados
en data/processed (plan B).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aire.data import build_all  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true", help="usar data/raw si existe")
    args = ap.parse_args()
    try:
        info = build_all(force_download=not args.cache)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] No se pudieron descargar/procesar los datos: {exc}")
        print("Plan B: el repositorio ya incluye los CSV procesados en data/processed/.")
        sys.exit(1)
    print("Datos listos:", info)


if __name__ == "__main__":
    main()
