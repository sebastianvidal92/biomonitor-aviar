"""Cliente de eBird API 2.0 — avistamientos puntuales (Módulo 2, consultas en vivo).

Usa el paquete `ebird-api` (import: ebird.api). Para el dataset histórico completo
(EBD) ver docs/CREDENCIALES.md §2 — eso se descarga por separado, no por esta API.

Ejemplo:
    from biomonitor.data import ebird
    regiones = ebird.chile_regions()                 # 16 regiones de Chile
    obs = ebird.recent_observations("CL-RM")          # Región Metropolitana
"""

from __future__ import annotations

from ebird.api.requests import get_observations, get_regions

from biomonitor.config import get_env


def _key() -> str:
    return get_env("EBIRD_API_KEY")


def chile_regions() -> list[dict]:
    """Lista de regiones (subnational1) de Chile: [{'code': 'CL-RM', 'name': ...}]."""
    return get_regions(_key(), "subnational1", "CL")


def recent_observations(region_code: str, days_back: int = 14, **kwargs) -> list[dict]:
    """Observaciones recientes en una región (ej. 'CL' o 'CL-RM').

    Devuelve registros con comName, sciName, lat, lng, obsDt, howMany, etc.
    """
    return get_observations(_key(), region_code, back=days_back, **kwargs)


def check() -> str:
    """Smoke test mínimo para 00_check_credentials.py."""
    regions = chile_regions()
    return f"OK — {len(regions)} regiones de Chile accesibles (ej. {regions[0]['name']})"
