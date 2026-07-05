# Credenciales y accesos a datos

Guía paso a paso para obtener cada API key / cuenta. Todas son **gratuitas** (uso
académico / no comercial). Copia `.env.example` a `.env` y rellénalo a medida que
las obtengas. Verifica todo con:

```bash
uv run python scripts/00_check_credentials.py
```

> ⏳ **Pide HOY las que tienen lead time**: el eBird Basic Dataset (EBD) puede
> tardar días en aprobarse y es la base del Módulo 2.

---

## 1. eBird API 2.0 — avistamientos puntuales · `EBIRD_API_KEY`

Para consultas en vivo (observaciones recientes, hotspots, taxonomía).

1. Crea una cuenta en https://ebird.org/ (gratuita).
2. Solicita la key en https://ebird.org/api/keygen → se genera al instante.
3. Pégala en `.env` → `EBIRD_API_KEY=...`

## 2. eBird Basic Dataset (EBD) — dataset histórico para el SDM ⏳

El export completo de avistamientos (lo que de verdad alimenta el Módulo 2).
**No es una API key**, es una solicitud de acceso a datos que un humano aprueba.

1. Con tu cuenta eBird, ve a https://ebird.org/data/download
2. Completa el formulario de solicitud (uso: investigación / tesis).
3. La aprobación puede tardar **días o un par de semanas** → pídelo cuanto antes.
4. Una vez aprobado, podrás descargar un *custom download* filtrado por país = Chile.

> Alternativa inmediata mientras esperas: descargar occurrences de eBird vía **GBIF**
> (punto 4), que no requiere aprobación manual.

## 3. Xeno-canto API v3 — audio etiquetado · `XENOCANTO_API_KEY`

Cantos de aves para el clasificador de audio (Módulo 1).

1. Crea cuenta en https://xeno-canto.org/
2. En tu perfil → **Account** → sección **API**, genera/copia tu key.
3. Pégala en `.env` → `XENOCANTO_API_KEY=...`

> Nota: la API v2 (sin key) fue retirada; la v3 exige key en cada petición.

## 4. GBIF — descargas masivas de occurrences · `GBIF_USER` / `GBIF_PWD` / `GBIF_EMAIL`

Agrega iNaturalist (fotos research-grade) + eBird en un solo formato, con filtro por
país y descargas reproducibles (DOI citable en el informe).

1. Regístrate en https://www.gbif.org/user/profile
2. Pon usuario, contraseña y email en `.env`.
3. Las descargas se piden por API (`pygbif`) y llegan como ZIP con un DOI.

## 5. Copernicus CDS / ERA5 — clima · `CDSAPI_KEY`

Reanálisis climático grillado (temperatura, precipitación, etc.) como covariables
del SDM.

1. Regístrate en https://cds.climate.copernicus.eu/
2. Acepta los *Terms of use* del dataset **ERA5** (una vez, en la página del dataset).
3. En **Your profile** copia tu *Personal Access Token*.
4. Opción A — `.env`: `CDSAPI_KEY=<token>` y `CDSAPI_URL=https://cds.climate.copernicus.eu/api`
   Opción B — archivo `~/.cdsapirc`:
   ```
   url: https://cds.climate.copernicus.eu/api
   key: <token>
   ```

## 6. Google Earth Engine — Sentinel-2 / NDVI · `EE_PROJECT`

Calcula NDVI en la nube y exporta solo tablas/CSV (no descargamos rásters pesados).

1. Regístrate en https://earthengine.google.com/ (proyecto **no comercial**).
2. Asocia/crea un proyecto de Google Cloud y copia su **Project ID** → `.env` `EE_PROJECT=`
3. Autentícate una sola vez (abre el navegador):
   ```bash
   uv run earthengine authenticate
   ```
4. En código: `ee.Initialize(project=os.environ["EE_PROJECT"])`

> Para correr sin navegador (Colab/servidor) crea un *service account* y apunta
> `GOOGLE_APPLICATION_CREDENTIALS` al JSON.

## 7. MMA / SAG — estado de conservación · sin credenciales

Listas del **Reglamento de Clasificación de Especies (RCE)** del Ministerio del Medio
Ambiente: descargables directo (Excel/PDF), sin trámite ni key.
- https://clasificacionespecies.mma.gob.cl/

## 8. iNaturalist directo — opcional

La lectura pública (observaciones, fotos con licencia CC) **no requiere key**. Solo
necesitarías una *OAuth app* para escribir o subir límites de tasa. Para este proyecto
basta con leer vía su API pública o, mejor aún, vía GBIF (punto 4).

---

### Resumen de urgencia

| Prioridad | Acción | Tiempo |
|---|---|---|
| 🔴 HOY | Solicitar **eBird EBD** (aprobación humana) | días–semanas |
| 🟢 Ya | eBird API key, Xeno-canto key, GBIF, CDS, GEE | minutos |
| ⚪ Después | iNaturalist OAuth (probablemente innecesario) | — |
