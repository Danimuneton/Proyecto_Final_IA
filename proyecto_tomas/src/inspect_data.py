"""Inspecciona la estructura de las fuentes candidatas (no asume nada)."""
import json
import pathlib
import requests

RAW = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)


def corto(obj, n=600):
    s = json.dumps(obj, ensure_ascii=False, default=str)
    return s[:n] + ("..." if len(s) > n else "")


print("== A. SIATA JSON completo ==")
r = requests.get("https://siata.gov.co/EntregaData1/Datos_SIATA_Aire_pm25.json", timeout=120)
destino = RAW / "siata_pm25.json"
destino.write_bytes(r.content)
print("Guardado:", destino, len(r.content), "bytes")
try:
    data = r.json()
    print("Tipo:", type(data).__name__)
    if isinstance(data, dict):
        print("Claves (primeras 15):", list(data.keys())[:15])
        for k in list(data.keys())[:2]:
            print(f"  {k}:", corto(data[k]))
    elif isinstance(data, list):
        print("Longitud:", len(data))
        for item in data[:2]:
            print("  ", corto(item))
except Exception as e:
    print("No se pudo leer como JSON:", e)

print("\n== B. datos.gov.co g4t8-zkc3 ==")
meta = requests.get("https://www.datos.gov.co/api/views/g4t8-zkc3.json", timeout=60).json()
print("Nombre:", meta.get("name"))
print("Descripción:", (meta.get("description") or "")[:500])
print("Columnas:", [c["fieldName"] for c in meta.get("columns", [])])
muestra = requests.get("https://www.datos.gov.co/resource/g4t8-zkc3.json", params={"$limit": 3}, timeout=60).json()
for fila in muestra:
    print("  ", corto(fila))
