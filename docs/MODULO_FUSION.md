# Módulo 1 — Fusión multimodal (audio + imagen)

Culminación del Módulo 1: combina el clasificador de cantos y el de fotos para la
identificación final de especie. Es la parte que responde la hipótesis central del
proyecto (¿la fusión multimodal supera a cada modalidad por separado?).

Última actualización: 2026-07-05.

---

## 1. Método — fusión tardía (late fusion)

Cada modalidad produce un vector de probabilidades por especie; la predicción final es
la **combinación ponderada** de ambos:

```
P_fusión = w · P_audio + (1 − w) · P_imagen          (w = 0.5 por defecto)
especie  = argmax(P_fusión)
```

- El **audio** aporta sus 55 clases (probabilidad agregada sobre las ventanas de la
  grabación). La **imagen** aporta 53 clases, mapeadas al espacio de 55 (las 2 especies
  solo-audio reciben 0 desde la rama de imagen).
- Es **fusión tardía** (a nivel de decisión), no requiere reentrenar los modelos.

## 2. Evaluación — pareo por especie

Los datasets **no están naturalmente pareados** (el audio viene de Xeno-canto, las fotos
de iNaturalist; no hay un registro con canto y foto del mismo individuo). Por eso la
fusión se evalúa **emparejando un audio y una foto de la misma especie** —simulando a un
observador que graba y fotografía la misma ave—, y se comparan las tres estrategias
sobre exactamente las mismas instancias pareadas.

- 53 especies (las presentes en ambas modalidades), 30 pares por especie → **1.590 pares**.
- Comparación *apples-to-apples*: audio-solo vs imagen-sola vs fusión sobre los mismos pares.

## 3. Resultados

| Modalidad | Accuracy | F1 macro |
|---|---|---|
| Audio solo | 0.867 | 0.863 |
| Imagen sola | 0.825 | 0.812 |
| **Fusión (w_audio = 0.5)** | **0.950** | **0.949** |

**Conclusiones:**
- La **fusión supera a ambas modalidades**: +8.3 pts sobre audio, +12.5 sobre imagen.
- **Cumple la meta del entregable** (accuracy > 0.85) con amplio margen.
- Confirma que los errores de audio e imagen son **complementarios**: cuando una
  modalidad confunde congéneres (acústica o visualmente), la otra suele resolverlos.

## 4. Comparación de estrategias de fusión (late / early / intermediate)

Además de la fusión tardía del sistema final, se compararon las tres estrategias clásicas
de fusión **bajo condiciones idénticas** (`scripts/24_fusion_comparison.py`): mismos
backbones ya entrenados (congelados como extractores de features), mismos pares por especie,
y **cabezas de fusión con configuración idéntica** (40 épocas, lr 1e-3, batch 64). Así se
aísla el efecto de la arquitectura de fusión:

- **Late fusion** (decisión): un clasificador por modalidad → se promedian las probabilidades.
- **Early fusion** (temprana): se concatenan los embeddings de audio (1280) e imagen (1536)
  a la entrada de una única cabeza.
- **Intermediate fusion** (intermedia): cada embedding pasa por una proyección propia, se
  concatenan las proyecciones y una cabeza conjunta clasifica.

| Estrategia | Accuracy | F1 macro |
|---|---|---|
| **Late fusion** (decisión) | **0.931** | 0.928 |
| **Early fusion** (concatenación) | 0.930 | 0.925 |
| Intermediate fusion (proyección) | 0.906 | 0.899 |

**Hallazgos:**
- **Late y early fusion empatan** (~0.93) y son las mejores: la forma más simple de combinar
  (promediar decisiones o concatenar features) es suficiente y óptima.
- **Intermediate fusion queda por debajo** (0.906): las proyecciones por modalidad agregan
  parámetros que sobreajustan con los datos pareados limitados, y como los embeddings ya son
  de alto nivel, esa transformación extra no aporta.
- Refuerza la elección de **late fusion** para el sistema final. (Nota: estos números usan
  cabezas entrenadas sobre embeddings congelados; el 0.950 del §3 promedia los softmax de los
  modelos completos, sin cabeza extra — coherente y algo superior.)

## 5. Notas y posibles mejoras

- La rama de imagen usa single-crop en la fusión; con TTA (10-crop) el techo sería aún
  algo mayor.
- La rama de audio usa el CNN (EfficientNet-B0, 0.858). Enganchar el AST (0.868) subiría
  el aporte del audio.
- El peso `w` es ajustable (`--w-audio`); 0.5 ya da 0.950. Un barrido de `w` puede
  reportarse en el informe como análisis de sensibilidad.

## 6. Reproducibilidad

```bash
uv run python scripts/13_multimodal_fusion.py                 # fusión tardía (sistema final)
uv run python scripts/13_multimodal_fusion.py --w-audio 0.6   # ajustar peso
uv run python scripts/24_fusion_comparison.py                 # late vs early vs intermediate
```

Requiere los modelos entrenados: `models/audio_efficientnet_b0.pt` y
`models/image_efficientnet_b3.pt` (con sus `*_labels.json`).
