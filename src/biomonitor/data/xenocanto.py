"""Cliente de Xeno-canto API v3 — audio etiquetado de cantos de aves (Módulo 1).

Docs: https://xeno-canto.org/explore/api  (v3 requiere key en cada petición).

Ejemplo:
    from biomonitor.data import xenocanto as xc
    n = xc.count('grp:birds cnt:Chile')
    recs = xc.search('grp:birds cnt:Chile sp:"Spinus barbatus"', max_pages=2)
"""

from __future__ import annotations

import time
from typing import Iterator

import requests

from biomonitor.config import get_env

API_URL = "https://xeno-canto.org/api/3/recordings"
_SLEEP = 1.0  # cortesía: 1 req/s


def _request(query: str, page: int = 1, max_retries: int = 4) -> dict:
    key = get_env("XENOCANTO_API_KEY")
    backoff = 5.0
    for attempt in range(max_retries):
        resp = requests.get(
            API_URL,
            params={"query": query, "key": key, "page": page},
            timeout=60,
        )
        if resp.status_code == 429:  # rate limited: espera y reintenta
            time.sleep(backoff)
            backoff *= 2
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()
    return resp.json()


def count(query: str) -> int:
    """Número total de grabaciones que matchean la consulta (sin descargarlas)."""
    return int(_request(query, page=1).get("numRecordings", 0))


def search(query: str, max_pages: int | None = None) -> list[dict]:
    """Devuelve la lista de metadatos de grabaciones para la consulta.

    Cada item incluye, entre otros: id, gen, sp, en (nombre inglés), cnt (país),
    loc, lat, lng, q (calidad A–E), length y `file` (URL de descarga del audio).
    """
    return list(iter_search(query, max_pages=max_pages))


def iter_search(query: str, max_pages: int | None = None) -> Iterator[dict]:
    first = _request(query, page=1)
    total_pages = int(first.get("numPages", 1))
    last = total_pages if max_pages is None else min(total_pages, max_pages)
    yield from first.get("recordings", [])
    for page in range(2, last + 1):
        time.sleep(_SLEEP)
        yield from _request(query, page=page).get("recordings", [])


def _length_seconds(length: str) -> int:
    """Convierte 'M:SS' o 'H:MM:SS' a segundos; 0 si no parsea."""
    try:
        parts = [int(p) for p in str(length).split(":")]
    except ValueError:
        return 0
    secs = 0
    for p in parts:
        secs = secs * 60 + p
    return secs


def select_recordings(
    scientific_name: str,
    qualities: list[str],
    max_recordings: int,
    max_length_seconds: int | None = None,
) -> list[dict]:
    """Recolecta hasta `max_recordings` grabaciones de la especie filtradas por calidad.

    Itera páginas y corta apenas junta suficientes (evita bajar metadatos de más).
    Descarta grabaciones más largas que `max_length_seconds` (WAV gigantes).
    """
    genus, epithet = scientific_name.split()[0], scientific_name.split()[1]
    quals = set(qualities)
    out: list[dict] = []
    for rec in iter_search(f"gen:{genus} sp:{epithet} grp:birds"):
        if rec.get("q") not in quals:
            continue
        if max_length_seconds and _length_seconds(rec.get("length", "0:00")) > max_length_seconds:
            continue
        out.append(rec)
        if len(out) >= max_recordings:
            break
    return out


def download_audio(url: str, dest_path, max_retries: int = 4) -> None:
    """Descarga un archivo de audio a dest_path (atómico, con reintentos en 429)."""
    from pathlib import Path

    dest_path = Path(dest_path)
    backoff = 5.0
    for _ in range(max_retries):
        resp = requests.get(url, timeout=120, stream=True)
        if resp.status_code == 429:
            time.sleep(backoff)
            backoff *= 2
            continue
        resp.raise_for_status()
        tmp = dest_path.with_suffix(dest_path.suffix + ".part")
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=8192):
                fh.write(chunk)
        tmp.rename(dest_path)
        return
    resp.raise_for_status()


def check() -> str:
    """Smoke test mínimo para 00_check_credentials.py."""
    n = count("grp:birds cnt:Chile")
    return f"OK — {n} grabaciones de aves para Chile disponibles"
