"""Resumen del JSON de SIATA ya descargado + consulta de Antioquia en datos.gov.co."""
import json
import pathlib
import requests
import pandas as pd

RAW = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"

print("== A. SIATA ==")
data = json.load(open(RAW / "siata_pm25.json"))
print("Claves de una estación:", list(data[0].keys()))
filas = []
for est in data:
    for d in est["datos"]:
        filas.append((est["codigoSerial"], est["latitud"], est.get("longitud"),
                      d["fecha"], d["valor"], d["calidad"]))
df = pd.DataFrame(filas, columns=["estacion", "lat", "lon", "fecha", "valor", "calidad"])
df["fecha"] = pd.to_datetime(df["fecha"])
df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
print("Filas:", len(df), "| rango:", df["fecha"].min(), "->", df["fecha"].max())
print("Valores <= -9000:", (df["valor"] <= -9000).mean().round(3), "(fracción)")
df.loc[df["valor"] <= -9000, "valor"] = float("nan")
print("Estadísticos de valor válido:")
print(df["valor"].describe().round(1).to_string())

res = df.groupby("estacion").agg(
    desde=("fecha", "min"), hasta=("fecha", "max"),
    pct_valido=("valor", lambda s: round(s.notna().mean() * 100, 1)))
df["dia"] = df["fecha"].dt.date
dias = (df.dropna(subset=["valor"]).groupby(["estacion", "dia"]).size()
          .loc[lambda s: s >= 18].groupby("estacion").size().rename("dias_ok"))
print(res.join(dias).to_string())
print("Valores de 'calidad' más frecuentes:")
print(df["calidad"].value_counts().head(8).to_string())

print("\n== B. datos.gov.co: Antioquia ==")
r = requests.get("https://www.datos.gov.co/resource/g4t8-zkc3.json", params={
    "$select": "nombre_fgda,municipio,msfl_code,count(*) as n",
    "$where": "departamento='Antioquia'",
    "$group": "nombre_fgda,municipio,msfl_code",
    "$order": "n DESC", "$limit": 30}, timeout=120)
print("Estado:", r.status_code)
for fila in r.json()[:30]:
    print("  ", fila)
