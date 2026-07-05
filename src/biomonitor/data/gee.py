"""Inicialización de Google Earth Engine — Sentinel-2 / NDVI (Módulo 2).

Dos modos de autenticación (elige uno):
  1. Cuenta de servicio (recomendado, sin navegador): define en .env
     GOOGLE_APPLICATION_CREDENTIALS con la ruta al JSON de la service account.
  2. Usuario (OAuth): tras correr `earthengine authenticate`.

Uso:
    from biomonitor.data.gee import init_ee
    init_ee()
    import ee
    img = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import ee

from biomonitor.config import PROJECT_ROOT, get_env


def _resolve_key_path() -> str | None:
    """Ruta al JSON de la cuenta de servicio; relativa se resuelve desde la raíz."""
    raw = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path) if path.exists() else None


def init_ee() -> str:
    """Inicializa Earth Engine y devuelve el project id usado."""
    project = get_env("EE_PROJECT")
    key_path = _resolve_key_path()

    if key_path:
        # Cuenta de servicio: lee el email del propio JSON y autentica sin navegador.
        with open(key_path, encoding="utf-8") as fh:
            sa_email = json.load(fh)["client_email"]
        credentials = ee.ServiceAccountCredentials(sa_email, key_path)
        ee.Initialize(credentials, project=project)
    else:
        # Credenciales de usuario (earthengine authenticate).
        ee.Initialize(project=project)
    return project


def check() -> str:
    """Smoke test mínimo para 00_check_credentials.py."""
    project = init_ee()
    ee.Number(1).getInfo()  # llamada trivial al backend
    mode = "service account" if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") else "usuario"
    return f"OK — inicializado (proyecto {project}, auth: {mode})"
