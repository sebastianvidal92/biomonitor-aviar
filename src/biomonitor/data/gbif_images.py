"""Descarga de fotos vía GBIF — iNaturalist Research-grade (Módulo 1, imagen).

Usa el dataset de iNaturalist Research-grade en GBIF: fotos con identificación
validada por la comunidad, geolocalizadas y con licencia Creative Commons. La
búsqueda de GBIF es pública (no requiere credenciales).

Ejemplo:
    from biomonitor.data import gbif_images as gi
    recs = gi.image_occurrences("Mimus thenca", max_records=50)
    gi.download_image(recs[0]["url"], "foto.jpg")
"""

from __future__ import annotations

from pathlib import Path

import requests
from pygbif import occurrences

# Dataset "iNaturalist Research-grade Observations" en GBIF.
INAT_RG_DATASET = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"

# Alias de taxonomía: nombres de la lista → nombre que usa GBIF/iNaturalist.
# (iNat reclasificó o usa otra grafía; sin esto la búsqueda da 0 fotos.)
ALIASES = {
    "Elaenia chilensis": "Elaenia albiceps",          # iNat la trata como subesp. de E. albiceps
    "Sylviorthorhynchus desmurii": "Sylviorthorhynchus desmursii",  # grafía
}

CC_MARKERS = ("creativecommons.org", "publicdomain")

# Hosts de colecciones/museos con media de especímenes (lentos/forbidden, no fotos de
# aves vivas). Se excluyen del fallback.
BAD_HOSTS = ("arctos.database.museum", "cvcoll.org", "biocol.org", "mczbase.mcz.harvard.edu")


def _to_medium(url: str) -> str:
    """Reescribe URLs de iNaturalist al tamaño 'medium' (ahorra disco; ~500 px)."""
    if "inaturalist-open-data" in url:
        for size in ("/original.", "/large."):
            url = url.replace(size, "/medium.")
    return url


def _is_photo(url: str, fmt: str, license: str, strict: bool) -> bool:
    """Filtra media que no son fotos de ave viva (espectrogramas, especímenes de museo)."""
    if not url or "xeno-canto" in url:
        return False
    if strict:  # fallback fuera de iNat RG: exige JPEG + licencia CC + host confiable
        if "jpeg" not in (fmt or "") and "jpg" not in (fmt or ""):
            return False
        if not any(mk in (license or "") for mk in CC_MARKERS):
            return False
        if any(h in url for h in BAD_HOSTS):
            return False
    return True


def _search(name: str, max_records: int, dataset: str | None, strict: bool,
            seen: set, source: str, basis_of_record: str | None = None,
            page_size: int = 300) -> list[dict]:
    out: list[dict] = []
    offset = 0
    while len(out) < max_records:
        params = dict(scientificName=name, mediaType="StillImage",
                      limit=page_size, offset=offset)
        if dataset:
            params["datasetKey"] = dataset
        if basis_of_record:  # excluye especímenes de museo (PRESERVED_SPECIMEN)
            params["basisOfRecord"] = basis_of_record
        res = occurrences.search(**params)
        results = res.get("results", [])
        for r in results:
            gid = r.get("key")
            if gid in seen:
                continue
            for m in r.get("media", []):
                url, fmt, lic = m.get("identifier"), m.get("format"), r.get("license")
                if m.get("type") == "StillImage" and _is_photo(url, fmt, lic, strict):
                    seen.add(gid)
                    out.append({
                        "gbif_id": gid, "url": _to_medium(url), "license": lic,
                        "lat": r.get("decimalLatitude"), "lon": r.get("decimalLongitude"),
                        "country": r.get("countryCode"), "source": source,
                    })
                    break  # una foto por observación (diversidad)
        offset += page_size
        if not results or offset >= res.get("count", 0):
            break
    return out


def image_occurrences(scientific_name: str, max_records: int, min_records: int = 0) -> list[dict]:
    """Fotos de la especie: iNaturalist RG primero; si no alcanza `min_records`,
    complementa con el resto de GBIF (solo JPEG con licencia CC, sin espectrogramas)."""
    name = ALIASES.get(scientific_name, scientific_name)
    seen: set = set()
    recs = _search(name, max_records, INAT_RG_DATASET, strict=False, seen=seen, source="inat_rg")
    if len(recs) < min_records:  # especie escasa en iNat RG → ampliar a todo GBIF
        recs += _search(name, max_records - len(recs), None, strict=True,
                        seen=seen, source="gbif_other", basis_of_record="HUMAN_OBSERVATION")
    return recs[:max_records]


def download_image(url: str, dest, timeout: int = 20) -> None:
    """Descarga una imagen a dest (atómico)."""
    resp = requests.get(url, timeout=timeout, stream=True)
    resp.raise_for_status()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=8192):
            fh.write(chunk)
    tmp.rename(dest)


def check() -> str:
    """Smoke test mínimo."""
    n = occurrences.search(
        scientificName="Mimus thenca", mediaType="StillImage",
        datasetKey=INAT_RG_DATASET, limit=0,
    )["count"]
    return f"OK — {n} fotos iNat RG para Mimus thenca"
