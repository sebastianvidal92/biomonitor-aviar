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
| **1a — Clasificador de audio** | ✅ completo | **AST 0.868 · CNN 0.858** acc en test (agregado) |
| **1b — Clasificador de imagen** | ✅ completo | **0.848** acc en test (TTA, 53 especies) |
| **1c — Fusión multimodal** | ✅ completo | **0.950** acc (fusión > audio 0.867 > imagen 0.825) |
| **2 — SDM espaciotemporal** | ✅ completo | **Random Forest 0.884 AUC > MaxEnt 0.863** |

**Ambos módulos cerrados.** Módulo 1: la fusión multimodal (0.950) supera a cada modalidad
(>0.85). Módulo 2: los ensambles de árboles superan a MaxEnt; el DL y las features
satelitales no aportan (hallazgo documentado).

Detalle completo (datos, hiperparámetros, métricas, análisis):
**[audio](docs/MODULO_AUDIO.md)** · **[imagen](docs/MODULO_IMAGEN.md)** · **[fusión](docs/MODULO_FUSION.md)** · **[SDM](docs/MODULO_SDM.md)**.
Decisiones de diseño y su justificación: **[docs/DECISIONES.md](docs/DECISIONES.md)**.

> La meta *">85% accuracy"* del entregable es para el **sistema multimodal**: el audio
> ya la supera y la fusión audio+imagen la excede con holgura. La imagen-sola (0.848)
> está en su techo para el backbone local; ver decisión en MODULO_IMAGEN §8.

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
10_download_images.py     descarga fotos (GBIF / iNaturalist Research-grade)
11_split_images.py        partición de imágenes (excluye especies con <15 fotos)
12_train_image.py         entrena el clasificador de imagen (EfficientNet/ViT)
13_multimodal_fusion.py   fusión tardía audio + imagen (evaluación por pareo)
14_evaluate_image.py      evalúa imagen con TTA (10-crop) + F1 por especie
15_download_occurrences.py avistamientos para el SDM (GBIF)
16_extract_covariates.py  covariables ambientales por punto (GEE)
17_train_sdm.py           SDM: MaxEnt vs deep learning conjunto
18-21 (…)                 experimentos multimodales del SDM (parches / embedding satelital)
22_train_sdm_ml.py        SDM: MaxEnt vs Random Forest vs Gradient Boosting
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

1. ✅ **Datos** — recopilación y preprocesamiento (55 especies; 2.508 audios + 7.068 fotos)
2. ✅ **Clasificador de audio** — CNN + AST, ambos > 0.85 accuracy
3. ✅ **Clasificador de imagen (0.848) + fusión multimodal (0.950)** — Módulo 1 completo
4. ✅ **SDM espaciotemporal** — Random Forest 0.884 AUC > MaxEnt 0.863
5. ⬜ **Prototipo web (demo)** — siguiente
6. ⬜ **Informe final y documentación**
