"""Disponibilidad de datos por especie: grabaciones (Xeno-canto) y fotos (GBIF).

Sirve para decidir, con datos, qué especies son entrenables (≥N grabaciones) y
para elegir el relleno de especies comunes hasta completar la lista objetivo.
"""

from __future__ import annotations

from pygbif import occurrences

from biomonitor.data import xenocanto


def _split(scientific_name: str) -> tuple[str, str]:
    parts = str(scientific_name).split()
    genus = parts[0] if parts else ""
    epithet = parts[1] if len(parts) > 1 else ""
    return genus, epithet


def xc_recordings(scientific_name: str, country: str | None = None) -> int:
    """Nº de grabaciones en Xeno-canto para la especie (opcionalmente filtrado por país)."""
    genus, epithet = _split(scientific_name)
    query = f"gen:{genus} sp:{epithet} grp:birds"
    if country:
        query += f" cnt:{country}"
    return xenocanto.count(query)


def gbif_photos(scientific_name: str, country: str | None = None) -> int:
    """Nº de registros con foto (StillImage) en GBIF para la especie."""
    params = {"scientificName": scientific_name, "mediaType": "StillImage", "limit": 0}
    if country:
        params["country"] = country
    return occurrences.search(**params)["count"]
