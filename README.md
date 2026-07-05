# Sistema Inteligente de Monitoreo de Biodiversidad Aviar en Chile

Proyecto de titulación — Magíster en Inteligencia Artificial, Pontificia Universidad
Católica de Chile. Semestre 2026/1. Profesor: Marcelo Mendoza.

Sistema de aprendizaje profundo multimodal para (1) **identificar especies de aves
chilenas** a partir de audio y/o imagen, y (2) **modelar su distribución
espaciotemporal** combinando avistamientos, clima y NDVI satelital.

## Módulos

| Módulo | Descripción | Datos | Modelos |
|---|---|---|---|
| **1 — Identificación multimodal** | Clasifica especie desde canto y/o foto, con fusión tardía | Xeno-canto (audio), iNaturalist/GBIF (imagen) | AST/CNN (audio) + EfficientNet/ViT (imagen) |
| **2 — Distribución espaciotemporal (SDM)** | Predice probabilidad de presencia por lugar/época | eBird, ERA5 (clima), Sentinel-2/NDVI (GEE) | SDM con deep learning vs. baseline MaxEnt |

## Setup

Requisitos: [uv](https://docs.astral.sh/uv/) (gestiona Python 3.12 automáticamente).

```bash
# 1. Crear entorno e instalar dependencias núcleo (adquisición de datos)
uv sync

# 2. Configurar credenciales
cp .env.example .env        # luego rellena .env (ver docs/CREDENCIALES.md)

# 3. Verificar accesos
uv run python scripts/00_check_credentials.py
```

Instalación por etapas (para no bajar todo de una):

```bash
uv sync --extra geo     # geoespacial (SDM, Módulo 2)
uv sync --extra ml      # PyTorch + audio/visión (entrenamiento)
uv sync --extra app     # demo web (Streamlit)
```

> En Mac con Apple Silicon, PyTorch usa la GPU vía Metal (MPS). Si el entrenamiento
> local queda corto, el código es agnóstico al entorno y se migra a Colab/nube.

## Estado y resultados

| Módulo | Estado | Resultado |
|---|---|---|
| **1a — Clasificador de audio** | ✅ completo | **AST 0.868 · CNN 0.858** acc en test (meta > 0.85) |
| 1b — Clasificador de imagen | ⬜ siguiente | — |
| 1c — Fusión multimodal | ⬜ pendiente | — |
| 2 — SDM espaciotemporal | ⬜ pendiente | — |

Detalle completo del módulo de audio (datos, hiperparámetros, épocas, métricas,
análisis de errores): **[docs/MODULO_AUDIO.md](docs/MODULO_AUDIO.md)**.
Decisiones de diseño y su justificación: **[docs/DECISIONES.md](docs/DECISIONES.md)**.

## Pipeline (scripts numerados, reanudables)

```
00_check_credentials.py   verifica las 5 fuentes de datos (eBird, Xeno-canto, GBIF, CDS, GEE)
01_build_species_list.py  lista candidata desde el RCE del MMA
02_enrich_availability.py disponibilidad de datos por especie (Xeno-canto + GBIF)
03_build_final_list.py    lista final de 55 especies objetivo
04_download_audio.py      descarga de audio (Xeno-canto)
05_split_dataset.py       partición train/val/test (a nivel de grabación, sin leakage)
06_preprocess_audio.py    cache a 32 kHz mono FLAC
07_train_audio.py         entrena el CNN (EfficientNet-B0)
08_evaluate_audio.py      evalúa con agregación por grabación + F1 por especie
09_train_audio_ast.py     entrena + evalúa el AST (transformer)
```

## Estructura

```
config/config.yaml   alcance geográfico, ventanas temporales, hiperparámetros
data/
  raw/ interim/ processed/   datos (en .gitignore, no se versionan)
  external/          referencia liviana versionada (listas de especies, RCE del MMA)
docs/                CREDENCIALES.md · MODULO_AUDIO.md · DECISIONES.md
scripts/             scripts ejecutables numerados (00…09)
src/biomonitor/      paquete: data, audio, vision, multimodal, sdm
reports/             métricas (F1 por especie, etc.)
models/              pesos entrenados (en .gitignore)
secrets/             credenciales locales (en .gitignore)
```

## Hitos

1. ✅ **Datos** — recopilación y preprocesamiento (55 especies, 2.508 grabaciones)
2. ✅ **Clasificador de audio** — CNN + AST, ambos > 0.85 accuracy
3. ⬜ **Clasificador de imagen + fusión multimodal**
4. ⬜ **SDM espaciotemporal**
5. ⬜ **Prototipo web (demo)**
6. ⬜ **Informe final y documentación**
