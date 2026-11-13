"""Verifica qué fuentes de datos de calidad del aire son accesibles desde tu máquina.
Uso: python src/probe_data.py
No asume ningún endpoint como correcto: prueba y reporta.
"""
import pathlib
import requests

RAW = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
TIMEOUT = 20


def catalogo_socrata(consulta: str, dominio: str = "www.datos.gov.co", limite: int = 10):
    """Busca datasets en datos.gov.co mediante el catálogo público de Socrata."""
    url = "https://api.us.socrata.com/api/catalog/v1"
    r = requests.get(url, params={"domains": dominio, "q": consulta, "limit": limite}, timeout=TIMEOUT)
    r.raise_for_status()
    for item in r.json().get("results", []):
        res = item["resource"]
        print(f"  id={res['id']}  nombre={res['name'][:80]}  actualizado={res.get('updatedAt', '?')[:10]}")
    return r.json()


def probar(url: str, nombre: str):
    try:
        r = requests.get(url, timeout=TIMEOUT)
        print(f"[{r.status_code}] {nombre}  ({len(r.content)} bytes)")
        if r.ok:
            destino = RAW / f"probe_{nombre}.txt"
            destino.write_bytes(r.content[:200_000])
            print(f"   muestra guardada en {destino}")
    except Exception as e:
        print(f"[ERROR] {nombre}: {e}")


if __name__ == "__main__":
    print("== 1. Catálogo datos.gov.co ==")
    for q in ["calidad del aire", "PM2.5", "SIATA"]:
        print(f"Búsqueda: {q}")
        try:
            catalogo_socrata(q)
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n== 2. Sitio SIATA (accesibilidad) ==")
    probar("https://siata.gov.co/", "siata_home")
    probar("https://siata.gov.co/CalidadAire/InformesDiarios/Informe_AMVA_24042026.pdf", "informe_amva_ejemplo")
    # Endpoint JSON de PM2.5 de SIATA: NO verificado por mí; si da 404, ignóralo.
    probar("https://siata.gov.co/EntregaData1/Datos_SIATA_Aire_pm25.json", "siata_pm25_json")
