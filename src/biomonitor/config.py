"""Carga de configuración (config.yaml) y credenciales (.env).

Uso:
    from biomonitor.config import CONFIG, get_env, PROJECT_ROOT
    CONFIG["region"]["bbox"]
    get_env("EBIRD_API_KEY")           # lanza si falta
    get_env("EE_PROJECT", required=False)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Raíz del proyecto = dos niveles sobre este archivo (src/biomonitor/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Carga .env de la raíz si existe (no falla si no está).
load_dotenv(PROJECT_ROOT / ".env")


@lru_cache(maxsize=1)
def load_config() -> dict:
    """Lee config/config.yaml una sola vez."""
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


CONFIG: dict = load_config()


def get_env(name: str, required: bool = True, default: str | None = None) -> str | None:
    """Devuelve una variable de entorno. Si required y falta, lanza error claro."""
    value = os.environ.get(name, default)
    if required and not value:
        raise RuntimeError(
            f"Falta la credencial '{name}'. Cópiala en .env "
            f"(ver docs/CREDENCIALES.md). Plantilla: .env.example"
        )
    return value


def data_path(kind: str, *parts: str) -> Path:
    """Ruta absoluta dentro de data/<kind>, creando el directorio si hace falta.

    kind ∈ {raw, interim, processed, external}
    """
    base = PROJECT_ROOT / CONFIG["paths"][kind]
    path = base.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
