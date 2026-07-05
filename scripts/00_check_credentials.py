"""Verifica qué credenciales están configuradas y prueba conectividad con cada API.

Diseñado para no fallar nunca: cada chequeo es independiente y reporta su estado.

    uv run python scripts/00_check_credentials.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Permite ejecutar el script sin instalar el paquete (añade src/ al path).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomonitor.config import PROJECT_ROOT  # noqa: E402

OK, WARN, FAIL = "✅", "⚠️ ", "❌"


def line(status: str, name: str, detail: str) -> None:
    print(f"  {status} {name:<22} {detail}")


def check_ebird() -> None:
    if not os.environ.get("EBIRD_API_KEY"):
        line(WARN, "eBird API", "EBIRD_API_KEY no configurada (.env)")
        return
    try:
        from biomonitor.data import ebird

        line(OK, "eBird API", ebird.check())
    except Exception as exc:  # noqa: BLE001
        line(FAIL, "eBird API", f"key presente pero falló: {exc}")


def check_xenocanto() -> None:
    if not os.environ.get("XENOCANTO_API_KEY"):
        line(WARN, "Xeno-canto", "XENOCANTO_API_KEY no configurada (.env)")
        return
    try:
        from biomonitor.data import xenocanto

        line(OK, "Xeno-canto", xenocanto.check())
    except Exception as exc:  # noqa: BLE001
        line(FAIL, "Xeno-canto", f"key presente pero falló: {exc}")


def check_gbif() -> None:
    if not all(os.environ.get(k) for k in ("GBIF_USER", "GBIF_PWD", "GBIF_EMAIL")):
        line(WARN, "GBIF", "GBIF_USER/PWD/EMAIL incompletos (.env)")
        return
    try:
        from pygbif import occurrences

        n = occurrences.search(country="CL", taxonKey=212, limit=0)["count"]  # 212 = Aves
        line(OK, "GBIF", f"OK — {n:,} occurrences de aves en Chile")
    except Exception as exc:  # noqa: BLE001
        line(FAIL, "GBIF", f"credenciales presentes pero falló: {exc}")


def check_cds() -> None:
    has_env = bool(os.environ.get("CDSAPI_KEY"))
    has_file = (Path.home() / ".cdsapirc").exists()
    if not (has_env or has_file):
        line(WARN, "Copernicus CDS", "sin CDSAPI_KEY ni ~/.cdsapirc")
        return
    try:
        import cdsapi

        cdsapi.Client()  # valida que encuentre url+key
        src = ".env" if has_env else "~/.cdsapirc"
        line(OK, "Copernicus CDS", f"cliente inicializado desde {src}")
    except Exception as exc:  # noqa: BLE001
        line(FAIL, "Copernicus CDS", f"config presente pero falló: {exc}")


def check_gee() -> None:
    if not os.environ.get("EE_PROJECT"):
        line(WARN, "Earth Engine", "EE_PROJECT no configurado (.env)")
        return
    try:
        from biomonitor.data import gee

        line(OK, "Earth Engine", gee.check())
    except Exception as exc:  # noqa: BLE001
        hint = (
            "¿el JSON de la cuenta de servicio existe y tiene permisos?"
            if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            else "configura GOOGLE_APPLICATION_CREDENTIALS (cuenta de servicio) en .env"
        )
        line(FAIL, "Earth Engine", f"falló: {exc} — {hint}")


def main() -> None:
    env_exists = (PROJECT_ROOT / ".env").exists()
    print("\n== Verificación de credenciales ==\n")
    if not env_exists:
        print("  (No existe .env todavía — copia .env.example a .env)\n")
    check_ebird()
    check_xenocanto()
    check_gbif()
    check_cds()
    check_gee()
    print("\n  Leyenda:  ✅ funciona   ⚠️  falta configurar   ❌ configurada pero con error")
    print("  Guía:     docs/CREDENCIALES.md\n")


if __name__ == "__main__":
    main()
