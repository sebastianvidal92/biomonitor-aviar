# Módulo 2 — Modelo de distribución espaciotemporal (SDM)

Documentación técnica del SDM: predice la probabilidad de presencia de cada especie en
función de la ubicación, la época del año y variables ambientales. Cubre datos, modelos
comparados, resultados y hallazgos. **Entregable:** SDM basado en ML/deep learning
comparado con el baseline clásico MaxEnt, evaluado con AUC-ROC.

Última actualización: 2026-07-06.

---

## Índice
1. [Datos](#1-datos)
2. [Covariables ambientales](#2-covariables-ambientales)
3. [Modelos comparados](#3-modelos-comparados)
4. [Resultados](#4-resultados)
5. [Hallazgos](#5-hallazgos)
6. [Reproducibilidad](#6-reproducibilidad)

---

## 1. Datos

**Presencia:** avistamientos de GBIF (agrega eBird + iNaturalist), con coordenadas y fecha,
dentro de Chile continental (`scripts/15`). **100.270 puntos**, 55 especies, 1953-2026.

**Fondo (pseudo-ausencias):** *target-group background* — una muestra de los avistamientos
de todas las especies (representan el esfuerzo de observación). Corrige el **sesgo de
muestreo** de la ciencia ciudadana, en vez de puntos aleatorios uniformes.

**Dataset final** (`data/interim/sdm_dataset.csv`, `scripts/16`): 25.171 presencia
(adelgazadas a ≤500/especie) + 9.755 fondo = **34.926 puntos** con covariables.

## 2. Covariables ambientales

Muestreadas por punto en Google Earth Engine:

| Covariable | Fuente |
|---|---|
| NDVI (vegetación) | MODIS MOD13A2 (1 km) |
| Temperatura media anual | WorldClim BIO (bio01) |
| Precipitación anual | WorldClim BIO (bio12) |
| Elevación | SRTM (30 m) |
| Estacionalidad | mes del avistamiento → sin/cos (eje temporal) |

> **Nota:** el plan original (inscripción) era ERA5 (clima) y Sentinel-2 (NDVI). Muestrear
> la mediana de S2 o el promedio de ERA5 en decenas de miles de puntos **excede el cómputo
> de GEE** ("Computation timed out"). MODIS NDVI y WorldClim (climatologías precalculadas)
> son sustitutos estándar en SDM y de muestreo liviano.

## 3. Modelos comparados

Presencia (1) vs. fondo (0), por especie, con split estratificado 70/30. Features: las 6
covariables (base). Se evaluaron:

- **MaxEnt** (baseline clásico): regresión logística penalizada con features cuadráticas.
- **Deep learning** (`scripts/17`): red neuronal **conjunta multi-especie** (embedding de especie).
- **Random Forest** y **Gradient Boosting** (`scripts/22`): ensambles de árboles (Boosted
  Regression Trees es un método estándar de SDM).
- **Multimodal** (exploratorio): DL + datos satelitales no estructurados — parche de
  cobertura de suelo vía CNN (`scripts/18-19`) y embedding de foundation model 64-dim
  (`scripts/20-21`).

## 4. Resultados

AUC-ROC promedio en test (54 especies con ≥20 presencias):

| Modelo | AUC | ¿Supera a MaxEnt? |
|---|---|---|
| MaxEnt (base) | 0.863 | baseline |
| Red neuronal (DL, base) | 0.849 | No |
| **Random Forest (base)** | **0.884** | **Sí — 46/54 especies** |
| Gradient Boosting (base) | 0.883 | Sí — 44/54 |
| Random Forest (base + embedding sat.) | 0.878 | Sí, pero el embedding baja |
| Gradient Boosting (base + embedding sat.) | 0.881 | Sí, pero el embedding baja |
| DL + parche cobertura de suelo (CNN) | 0.613 | No (sobreajuste) |
| DL + embedding satelital (64-dim) | 0.767 | No (sobreajuste) |

Detalle por especie en `reports/sdm_auc_por_especie.csv`, `sdm_ml_auc.csv`,
`sdm_multimodal_auc.csv`, `sdm_embedding_auc.csv`.

## 5. Hallazgos

1. **Los ensambles de árboles superan a MaxEnt** (RF 0.884 vs 0.863, en ~85% de las
   especies). Random Forest / Boosted Regression Trees son métodos SOTA de SDM; el
   resultado es sólido y esperable.
2. **La red neuronal NO supera a MaxEnt** (0.849) en este problema tabular con muestras
   moderadas — fenómeno conocido: modelos regularizados/árboles suelen ganarle a las redes
   en datos tabulares.
3. **Los datos satelitales no estructurados NO ayudan**: tanto el parche de cobertura de
   suelo (CNN) como el embedding de 64-dim **degradaron** el desempeño por sobreajuste, y
   el embedding bajó incluso a los árboles. Las covariables ambientales ya capturan la
   señal relevante; la imagen satelital resulta redundante para este SDM.
4. **Implicancia:** para SDM de avifauna chilena con buenas covariables ambientales, un
   Random Forest bien regularizado es la mejor opción; la complejidad extra (deep learning,
   imágenes) no se justifica. (Trabajo futuro: RGB crudo de Sentinel-2 en la nube.)

## 6. Reproducibilidad

```bash
uv run python scripts/15_download_occurrences.py     # avistamientos (GBIF)
uv run python scripts/16_extract_covariates.py        # covariables ambientales (GEE)
uv run python scripts/17_train_sdm.py                 # MaxEnt vs DL conjunto
uv run python scripts/22_train_sdm_ml.py              # + Random Forest / Gradient Boosting
# Exploratorio (satélite no estructurado — no mejoró):
uv run python scripts/18_extract_patches.py           # parches cobertura de suelo
uv run python scripts/19_train_sdm_multimodal.py      # DL + CNN
uv run python scripts/20_extract_embedding.py         # embedding satelital 64-dim
uv run python scripts/21_train_sdm_embedding.py       # DL + embedding
```

Hiperparámetros en `config/config.yaml` (`sdm`, `train_sdm`).
