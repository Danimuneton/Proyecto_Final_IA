"""Limpia el JSON horario de SIATA -> CSV diario por estación.
Entrada : data/raw/siata_pm25.json
Salida  : data/processed/pm25_diario.csv  (estacion, nombre, fecha, pm25)
          data/processed/estaciones.csv   (estacion, nombre, nombre_corto, lat, lon)
"""
import json
import pathlib
import pandas as pd

BASE = pathlib.Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw" / "siata_pm25.json"
OUT = BASE / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

MIN_HORAS = 18      # 18 de 24 horas = 75 % de datos válidos para aceptar el promedio diario
MIN_DIAS = 200      # se descartan estaciones con menos días válidos
VALOR_MAX = 500     # por encima se considera dato inválido (hay centinelas como 99999000)

data = json.load(open(RAW))

estaciones = pd.DataFrame([{
    "estacion": e["codigoSerial"], "nombre": e.get("nombre"),
    "nombre_corto": e.get("nombreCorto"), "lat": e.get("latitud"), "lon": e.get("longitud"),
} for e in data])

filas = [(e["codigoSerial"], d["fecha"], d["valor"]) for e in data for d in e["datos"]]
df = pd.DataFrame(filas, columns=["estacion", "fecha", "valor"])
df["fecha"] = pd.to_datetime(df["fecha"])   # se usa tal cual viene en el archivo
df["valor"] = pd.to_numeric(df["valor"], errors="coerce")

invalidos = (df["valor"] < 0) | (df["valor"] > VALOR_MAX)
print(f"Filas horarias: {len(df)} | inválidas (negativas, -9999 o > {VALOR_MAX}): {invalidos.mean():.1%}")
df.loc[invalidos, "valor"] = float("nan")

df["dia"] = df["fecha"].dt.normalize()
diario = (df.groupby(["estacion", "dia"])["valor"]
            .agg(pm25="mean", n="count").reset_index())
diario = diario[diario["n"] >= MIN_HORAS].drop(columns="n")
diario["pm25"] = diario["pm25"].round(2)

dias_por_est = diario.groupby("estacion").size()
buenas = dias_por_est[dias_por_est >= MIN_DIAS].index
descartadas = sorted(set(estaciones["estacion"]) - set(buenas))
print("Estaciones descartadas (pocos datos):", descartadas)
diario = diario[diario["estacion"].isin(buenas)]
estaciones = estaciones[estaciones["estacion"].isin(buenas)]

diario = diario.rename(columns={"dia": "fecha"}).merge(
    estaciones[["estacion", "nombre"]], on="estacion", how="left")
diario = diario[["estacion", "nombre", "fecha", "pm25"]].sort_values(["estacion", "fecha"])

diario.to_csv(OUT / "pm25_diario.csv", index=False)
estaciones.to_csv(OUT / "estaciones.csv", index=False)

print(f"\nEstaciones conservadas: {len(estaciones)} | filas diarias: {len(diario)}")
print("Rango:", diario["fecha"].min().date(), "->", diario["fecha"].max().date())
print("PM2.5 diario (µg/m³):")
print(diario["pm25"].describe().round(1).to_string())
print("\nEstaciones:")
print(estaciones.to_string(index=False))
