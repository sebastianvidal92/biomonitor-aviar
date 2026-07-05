# Registro de decisiones (bitácora)

Decisiones de diseño del proyecto y su justificación, para trazabilidad en el informe.

---

## Alcance y pregunta de investigación

- **Alcance geográfico:** Chile continental (bbox aprox. lat −17.5°…−56.0°, lon
  −66.4°…−75.7°). Se **excluye** territorio antártico e islas oceánicas (Rapa Nui,
  Juan Fernández, Desventuradas) — avifauna distinta que sesgaría los modelos.
- **Foco de especies:** endémicas + amenazadas (VU/EN/CR) según el Reglamento de
  Clasificación de Especies (RCE) del MMA. Ventana temporal 2015-2024.
- **Pregunta principal:** ¿puede un sistema de deep learning **multimodal** identificar
  especies de aves chilenas (audio y/o imagen) y **modelar su distribución
  espaciotemporal** mejor que enfoques de modalidad única y que MaxEnt?
- **Hipótesis:** la fusión multimodal supera a cada modalidad sola; un SDM con deep
  learning supera a MaxEnt.

## Fuentes de datos y proveedores

| Necesidad | Elección | Motivo |
|---|---|---|
| Audio | Xeno-canto API v3 | Cantos etiquetados; la v2 fue retirada |
| Imágenes | iNaturalist vía GBIF | Descarga masiva reproducible, con licencia y DOI |
| Avistamientos | eBird (API + EBD) | Estándar; EBD requiere solicitud con lead time |
| **Satélite / NDVI** | **Google Earth Engine** | Calcula NDVI en la nube, exporta CSV → **no bajar rásters pesados** (disco limitado) |
| **Clima** | **ERA5 (Copernicus CDS)** | Grillado, accesible por API, covariables consistentes para SDM (vs. estaciones DMC, irregulares) |
| Estado conservación | MMA RCE (Excel) | Fuente oficial, sin API key |

- **Autenticación GEE:** se usa **cuenta de servicio** (JSON) en vez de OAuth de
  usuario, porque el autenticador de Earth Engine se **bloqueaba** ("Se bloqueó esta
  app") sin pantalla de consentimiento configurada. La service account evita todo eso
  y sirve igual en la nube/Colab.

## Entorno y cómputo

- **Gestor:** `uv` con **Python 3.12** fijado (el sistema tiene 3.14, sin wheels
  estables de PyTorch todavía).
- **Cómputo:** local en Mac Apple M4 Pro (GPU vía **MPS**). Los cómputos pesados
  (descargas, entrenamiento) los ejecuta el usuario manualmente.
- **Sin librosa:** se usa `torchaudio` para resampleo y espectrogramas (evita el
  conflicto `numba`/`llvmlite` con Python 3.12).

## Módulo 1 — Audio (ver [MODULO_AUDIO.md](MODULO_AUDIO.md))

- **Lista de 55 especies:** núcleo de conservación (14) + endémicas comunes (6) +
  relleno común (35). **No se puede llegar a 50 solo con amenazadas** por escasez de
  audio → decisión de completar con comunes (la propuesta dice "foco", no exclusividad).
- **Umbral de inclusión:** ≥20 grabaciones/especie (opciones evaluadas: 10/20/30). Las
  raras usan data augmentation.
- **Pelágicas excluidas** del clasificador de cantos (no vocalizan en tierra; ~25 de las
  amenazadas eran marinas).
- **Cache en FLAC, no OGG** — el encoder Vorbis de libsndfile hace segfault con ciertos
  WAV grandes.
- **Split a nivel de grabación** (no de clip) para evitar *data leakage*.
- **Evaluación agregada por grabación** como métrica operacional (además de la de
  ventana única).
- **Dos modelos:** baseline CNN (EfficientNet-B0) + AST (transformer), para la
  comparación que pide la propuesta. Resultado: ambos > 0.85; AST mejor pero ~20× más caro.

## Pendientes / próximos

- Módulo 1 imagen: descargar fotos (GBIF/iNaturalist), entrenar EfficientNet/ViT.
- Fusión multimodal tardía audio + imagen.
- Módulo 2 (SDM): eBird EBD + ERA5 + NDVI; comparar deep learning vs. MaxEnt.
- Solicitar **eBird EBD** (aprobación humana, días) si no se ha hecho.
