"""Lectura y limpieza del listado RCE del MMA — estado de conservación (sin API key).

Fuente: Nómina de Especies según Estado de Conservación (procesos 1°–20° RCE),
descargada de https://clasificacionespecies.mma.gob.cl/ a data/external/mma/.

Construye la lista candidata de aves objetivo del proyecto: aves chilenas
continentales con foco en endémicas y categorías amenazadas (CR/EN/VU).
"""

from __future__ import annotations

import re

import pandas as pd

from biomonitor.config import PROJECT_ROOT

RCE_XLSX = (
    PROJECT_ROOT
    / "data/external/mma/RCE_especies_clasificadas_20proc_jun2026.xlsx"
)

# Orden de severidad para resolver categorías mixtas (se toma la más severa).
SEVERITY = {"EX": 7, "EW": 6, "RE": 5, "CR": 4, "EN": 3, "VU": 2, "NT": 1, "LC": 0, "DD": -1}
THREATENED = {"CR", "EN", "VU"}  # categorías que definen "amenazada"

# Órdenes pelágicos: no vocalizan en tierra y su distribución es oceánica →
# fuera del clasificador de cantos y del SDM terrestre.
PELAGIC_ORDERS = {"Procellariiformes", "Sphenisciformes", "Phaethontiformes"}

# Aves endémicas de Chile CONTINENTAL (excluye endémicas insulares de Juan
# Fernández/Masafuera). Varias son comunes y NO están en el RCE (no amenazadas),
# pero son de máximo interés para un sistema de monitoreo de avifauna chilena.
CONTINENTAL_ENDEMICS = {
    "Nothoprocta perdicaria",      # perdiz chilena
    "Pteroptochos megapodius",     # turca
    "Pteroptochos castaneus",      # huet-huet del sur (castaño)
    "Scelorchilus albicollis",     # tapaculo
    "Scytalopus fuscus",           # churrín del norte
    "Ochetorhynchus melanurus",    # bandurrilla / chilia (Chilia melanura)
    "Mimus thenca",                # tenca
    "Enicognathus leptorhynchus",  # choroy
    "Eulidia yarrellii",           # picaflor de Arica
}


def habitat_group(order: str, family: str) -> str:
    """Clasifica el hábitat para decidir si la especie sirve al audio/SDM terrestre."""
    order = str(order).strip()
    if order in PELAGIC_ORDERS:
        return "pelagic"
    if order == "Charadriiformes":
        return "coastal_shorebird"
    return "terrestrial"

# Regiones continentales (Arica → Magallanes). Excluye Antártica e islas oceánicas
# (Pascua, Salas y Gómez, Juan Fernández, Desventuradas) — fuera del alcance.
CONTINENTAL_REGIONS = [
    "Arica y Parinacota", "Tarapacá", "Antofagasta", "Atacama", "Coquimbo",
    "Valparaíso continental", "Metropolitana", "O'higgins", "Maule", "Ñuble",
    "Bío-Bío", "Araucanía", "De Los Ríos", "De Los Lagos", "Aysén",
    "Magallanes continental e insular",
]


def _short(col: str) -> str:
    """Acorta los encabezados multilínea del Excel a su primera línea."""
    return str(col).split("\n")[0].strip().rstrip(":").strip()


def _primary_category(raw: str) -> str | None:
    """Extrae la categoría más severa de un texto que puede tener varias por región."""
    codes = re.findall(r"\b(EX|EW|RE|CR|EN|VU|NT|LC|DD)\b", str(raw).upper())
    if not codes:
        return None
    return max(codes, key=lambda c: SEVERITY.get(c, -99))


def load_rce_birds() -> pd.DataFrame:
    """DataFrame limpio de aves del RCE, con flags de endemismo, amenaza y foco.

    Columnas: scientific_name, common_name, order, family, endemic, continental,
    category_primary, category_raw, proceso_rce, in_focus.
    """
    df = pd.read_excel(RCE_XLSX, sheet_name="Especies", header=0)
    df = df.rename(columns={c: _short(c) for c in df.columns})

    aves = df[df["CLASE"].astype(str).str.strip().str.lower() == "aves"].copy()

    out = pd.DataFrame()
    out["scientific_name"] = aves["NOMBRE CIENTÍFICO"].astype(str).str.strip()
    out["common_name"] = aves["NOMBRE COMÚN"].astype(str).str.strip()
    out["order"] = aves["ORDEN"].astype(str).str.strip()
    out["family"] = aves["FAMILIA"].astype(str).str.strip()

    endemic_raw = aves["ENDÉMICA"].astype(str).str.strip().str.upper()
    out["endemic"] = endemic_raw.isin(["SI", "SÍ"])

    present = aves[CONTINENTAL_REGIONS].apply(pd.to_numeric, errors="coerce").fillna(0)
    out["continental"] = present.sum(axis=1) > 0

    cat_raw = aves["CATEGORÍA VIGENTE"].astype(str).str.strip()
    out["category_raw"] = cat_raw
    out["category_primary"] = cat_raw.map(_primary_category)
    out["proceso_rce"] = aves["NÚMERO PROCESO RCE"].astype(str).str.strip()

    out["in_focus"] = out["endemic"] | out["category_primary"].isin(THREATENED)
    out["habitat_group"] = [habitat_group(o, f) for o, f in zip(out["order"], out["family"])]

    out = out[out["scientific_name"].str.len() > 0].reset_index(drop=True)
    return out
