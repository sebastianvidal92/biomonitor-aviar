"""Covariables ambientales para el SDM (Módulo 2), extraídas de Google Earth Engine.

Un solo muestreo por punto entrega: NDVI (Sentinel-2), temperatura y precipitación
(ERA5-Land) y elevación (SRTM).

Los puntos de fondo (pseudo-ausencias) se toman como **target-group background**: una
muestra de los avistamientos de todas las especies (ya en Chile, con coordenadas reales),
que representan el esfuerzo de observación y corrigen el sesgo de muestreo de la ciencia
ciudadana. Se arma en el script; aquí solo va el muestreo de covariables.
"""

from __future__ import annotations

import time

import ee
import pandas as pd

from biomonitor.config import CONFIG

BBOX = CONFIG["region"]["bbox"]


def study_area() -> ee.Geometry:
    return ee.Geometry.BBox(BBOX["lon_min"], BBOX["lat_min"], BBOX["lon_max"], BBOX["lat_max"])


def covariate_image() -> ee.Image:
    """Imagen multibanda [ndvi, temp, precip, elev] para muestrear en puntos.

    Se usan capas mayormente PRECALCULADAS para que el muestreo en decenas de miles de
    puntos no exceda el cómputo de GEE (promediar colecciones grandes da "Computation
    timed out"):
      - NDVI: MODIS MOD13A2 (1 km), promedio de 1 año (reducción mínima).
      - Clima: WorldClim BIO — temperatura media anual (bio01) y precipitación anual
        (bio12), imágenes estáticas. Es el covariable climático estándar en SDM.
      - Elevación: SRTM (estática).
    (El plan original era ERA5/Sentinel-2, pero su muestreo a esta escala es inviable en
    GEE; WorldClim + MODIS son sustitutos estándar y de cómputo liviano.)
    """
    ndvi = (ee.ImageCollection("MODIS/061/MOD13A2")
            .filterDate("2022-01-01", "2022-12-31")
            .select("NDVI").mean().multiply(0.0001).rename("ndvi"))
    bio = ee.Image("WORLDCLIM/V1/BIO")
    temp = bio.select("bio01").multiply(0.1).rename("temp")   # temp media anual (°C)
    precip = bio.select("bio12").rename("precip")             # precipitación anual (mm)
    elev = ee.Image("USGS/SRTMGL1_003").select("elevation").rename("elev")
    return ndvi.addBands([temp, precip, elev])


def embedding_image() -> ee.Image:
    """Embedding satelital de Google (AlphaEarth, 64 bandas): representación aprendida
    de la imagen satelital por un foundation model. Rico y de muestreo liviano (mosaico)."""
    return (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
            .filterDate("2023-01-01", "2024-01-01").mosaic())


def sample_covariates(points: list[dict], cov_img: ee.Image,
                      scale: int = 1000, batch: int = 500) -> pd.DataFrame:
    """Muestrea las covariables en `points` (con sus propiedades) por lotes.

    `points`: lista de dicts con 'lat', 'lon' y propiedades extra (species, month, label)
    que se conservan en la salida.
    """
    rows = []
    for i in range(0, len(points), batch):
        chunk = points[i:i + batch]
        feats = [
            ee.Feature(ee.Geometry.Point([p["lon"], p["lat"]]),
                       {k: v for k, v in p.items() if k not in ("lat", "lon")})
            for p in chunk
        ]
        sampled = cov_img.sampleRegions(
            collection=ee.FeatureCollection(feats), scale=scale, geometries=True)
        for attempt in range(4):  # reintentos ante timeouts transitorios de GEE
            try:
                info = sampled.getInfo()
                break
            except Exception:  # noqa: BLE001
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))
        for f in info["features"]:
            props = dict(f["properties"])
            lon, lat = f["geometry"]["coordinates"]
            props["lat"], props["lon"] = lat, lon
            rows.append(props)
    return pd.DataFrame(rows)
