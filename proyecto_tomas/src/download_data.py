"""Descarga el JSON horario de PM2.5 de SIATA a data/raw/siata_pm25.json.

Paso OPCIONAL: el repositorio ya trae el resultado limpio en data/processed/.
Uso:  python src/download_data.py && python src/clean_data.py
"""
import pathlib

import requests

URL = "https://siata.gov.co/EntregaData1/Datos_SIATA_Aire_pm25.json"
DESTINO = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw" / "siata_pm25.json"

if __name__ == "__main__":
    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(URL, timeout=180)
    r.raise_for_status()
    datos = r.json()                      # falla si la respuesta no es un JSON válido
    DESTINO.write_bytes(r.content)
    print(f"Guardado {DESTINO} ({len(r.content) / 1e6:.1f} MB, {len(datos)} estaciones)")
    print("Aviso: SIATA puede actualizar este archivo; los resultados del informe usan la copia descargada el 12 nov 2026.")
