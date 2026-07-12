"""Avistamientos (puntos de presencia) desde GBIF para el SDM (Módulo 2).

GBIF agrega eBird + iNaturalist + otras fuentes. Se descargan ocurrencias con
coordenadas válidas dentro de Chile continental, con fecha (para el eje temporal).
La búsqueda de GBIF es pública (no requiere credenciales).
"""

from __future__ import annotations

import time

from pygbif import occurrences

from biomonitor.data.gbif_images import ALIASES  # mismos alias de taxonomía


def _search(max_retries: int = 4, **params) -> dict:
    """occurrences.search con reintentos ante timeouts transitorios de GBIF."""
    for attempt in range(max_retries):
        try:
            return occurrences.search(**params)
        except Exception:  # noqa: BLE001
            if attempt == max_retries - 1:
                raise
            time.sleep(3 * (attempt + 1))
    return {}


def species_occurrences(scientific_name: str, max_records: int, bbox: dict,
                        page_size: int = 300) -> list[dict]:
    """Puntos de presencia (lat, lon, año, mes) de la especie dentro del bbox continental."""
    name = ALIASES.get(scientific_name, scientific_name)
    out: list[dict] = []
    offset = 0
    while len(out) < max_records:
        res = _search(
            scientificName=name,
            country="CL",
            hasCoordinate=True,
            hasGeospatialIssue=False,
            limit=page_size,
            offset=offset,
        )
        results = res.get("results", [])
        for r in results:
            lat, lon = r.get("decimalLatitude"), r.get("decimalLongitude")
            if lat is None or lon is None:
                continue
            if not (bbox["lat_min"] <= lat <= bbox["lat_max"]
                    and bbox["lon_min"] <= lon <= bbox["lon_max"]):
                continue  # excluye territorio insular/oceánico
            out.append({
                "gbif_id": r.get("key"),
                "lat": lat, "lon": lon,
                "year": r.get("year"), "month": r.get("month"),
                "basis": r.get("basisOfRecord"),
            })
        offset += page_size
        if not results or offset >= res.get("count", 0) or offset >= 100000:
            break  # GBIF limita offset+limit a 100.000
    return out[:max_records]
