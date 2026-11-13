"""Descarga, limpieza y carga de datos reales.

Fuentes:
- SIATA (Sistema de Alerta Temprana del Valle de Aburrá): PM2.5 horario por estación,
  archivo abierto ``Datos_SIATA_Aire_pm25.json``.
- Open-Meteo Historical Weather API (reanálisis ERA5 de Copernicus/ECMWF): variables
  meteorológicas diarias para el centro del valle.

Plan B: si las fuentes no responden, se usan los CSV procesados que vienen en
``data/processed`` (versionados en el repositorio).
"""
from __future__ import annotations

import json
import logging
import unicodedata
from functools import lru_cache

import numpy as np
import pandas as pd
import requests

from . import config as C

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Descarga
# ---------------------------------------------------------------------------
def download_siata_pm25(timeout: int = 180) -> list[dict]:
    """Descarga el JSON de PM2.5 de SIATA y guarda una copia cruda en data/raw."""
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    resp = requests.get(C.SIATA_PM25_URL, timeout=timeout)
    resp.raise_for_status()
    raw = resp.json()
    (C.RAW_DIR / "Datos_SIATA_Aire_pm25.json").write_text(json.dumps(raw), encoding="utf-8")
    return raw


def process_siata_pm25(raw: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convierte el JSON de SIATA en tablas horaria, diaria y de estaciones."""
    rows, stations = [], []
    for s in raw:
        code = int(s["codigoSerial"])
        nombre = s.get("nombre", "").strip()
        activa = not nombre.startswith("_OFF")
        stations.append({
            "codigo": code,
            "nombre": nombre.replace("_OFF-", "").strip(),
            "nombre_corto": s.get("nombreCorto", ""),
            "municipio": C.MUNICIPIO_POR_CODIGO.get(code, "Desconocido"),
            "latitud": float(s["latitud"]),
            "longitud": float(s["longitud"]),
            "activa_en_fuente": activa,
        })
        for x in s.get("datos", []):
            rows.append((code, x["fecha"], float(x.get("calidad", 9)), x.get("valor")))

    hourly = pd.DataFrame(rows, columns=["codigo", "fecha", "calidad", "valor"])
    hourly["fecha"] = pd.to_datetime(hourly["fecha"])
    hourly["valor"] = pd.to_numeric(hourly["valor"], errors="coerce")
    valid = (hourly["calidad"] <= C.MAX_QUALITY_FLAG) & (hourly["valor"] >= 0)
    hourly["pm25"] = hourly["valor"].where(valid)
    hourly = hourly[["codigo", "fecha", "calidad", "pm25"]]

    hourly["dia"] = hourly["fecha"].dt.floor("D")
    daily = (hourly.groupby(["codigo", "dia"])["pm25"]
             .agg(pm25="mean", horas_validas="count", pm25_max_horario="max")
             .reset_index().rename(columns={"dia": "fecha"}))
    # Promedio de 24 h sólo si hay >= 75 % de horas válidas
    daily.loc[daily["horas_validas"] < C.MIN_HOURS_PER_DAY, ["pm25", "pm25_max_horario"]] = np.nan
    daily["pm25"] = daily["pm25"].round(2)

    st = pd.DataFrame(stations)
    n_valid = daily.groupby("codigo")["pm25"].count()
    st["dias_validos"] = st["codigo"].map(n_valid).fillna(0).astype(int)
    hourly = hourly.drop(columns="dia")
    return hourly, daily, st


def download_meteo(start: str, end: str, timeout: int = 120) -> pd.DataFrame:
    """Variables meteorológicas diarias del centro del valle (Open-Meteo / ERA5)."""
    params = {
        "latitude": C.METEO_LAT, "longitude": C.METEO_LON,
        "start_date": start, "end_date": end, "timezone": "America/Bogota",
        "daily": ",".join([
            "precipitation_sum", "wind_speed_10m_mean", "shortwave_radiation_sum",
            "temperature_2m_mean", "relative_humidity_2m_mean", "cloud_cover_mean",
        ]),
        "wind_speed_unit": "ms",
    }
    resp = requests.get(C.OPEN_METEO_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    js = resp.json()
    d = pd.DataFrame(js["daily"]).rename(columns={
        "time": "fecha", "precipitation_sum": "lluvia_mm", "wind_speed_10m_mean": "viento_ms",
        "shortwave_radiation_sum": "radiacion_mj", "temperature_2m_mean": "temp_c",
        "relative_humidity_2m_mean": "humedad_pct", "cloud_cover_mean": "nubosidad_pct",
    })
    d["fecha"] = pd.to_datetime(d["fecha"])
    return d


def build_all(force_download: bool = True) -> dict:
    """Ejecuta el pipeline completo de datos y escribe los CSV procesados."""
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = C.RAW_DIR / "Datos_SIATA_Aire_pm25.json"
    if force_download or not raw_path.exists():
        raw = download_siata_pm25()
    else:
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
    hourly, daily, st = process_siata_pm25(raw)
    hourly.to_csv(C.PM25_HOURLY_FILE, index=False, compression="gzip")
    daily.to_csv(C.PM25_DAILY_FILE, index=False)
    st.to_csv(C.STATIONS_FILE, index=False)

    start = daily["fecha"].min().strftime("%Y-%m-%d")
    end = daily["fecha"].max().strftime("%Y-%m-%d")
    meteo = download_meteo(start, end)
    meteo.to_csv(C.METEO_DAILY_FILE, index=False)
    return {"horas": len(hourly), "dias_estacion": int(daily["pm25"].notna().sum()),
            "estaciones": len(st), "inicio": start, "fin": end, "dias_meteo": len(meteo)}


# ---------------------------------------------------------------------------
# Carga (usada por modelos, herramientas y dashboard)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_stations() -> pd.DataFrame:
    return pd.read_csv(C.STATIONS_FILE)


@lru_cache(maxsize=1)
def load_daily() -> pd.DataFrame:
    df = pd.read_csv(C.PM25_DAILY_FILE, parse_dates=["fecha"])
    return df


@lru_cache(maxsize=1)
def load_hourly() -> pd.DataFrame:
    return pd.read_csv(C.PM25_HOURLY_FILE, parse_dates=["fecha"])


@lru_cache(maxsize=1)
def load_meteo() -> pd.DataFrame:
    return pd.read_csv(C.METEO_DAILY_FILE, parse_dates=["fecha"])


def active_stations() -> pd.DataFrame:
    """Estaciones con al menos 30 días válidos en el periodo."""
    st = load_stations()
    return st[st["dias_validos"] >= 30].sort_values(["municipio", "nombre"]).reset_index(drop=True)


def daily_matrix() -> pd.DataFrame:
    """Tabla fecha x estación con el PM2.5 diario (NaN = sin dato válido)."""
    d = load_daily()
    codes = active_stations()["codigo"].tolist()
    m = d[d["codigo"].isin(codes)].pivot(index="fecha", columns="codigo", values="pm25")
    full = pd.date_range(m.index.min(), m.index.max(), freq="D")
    return m.reindex(full)


def date_range() -> tuple[pd.Timestamp, pd.Timestamp]:
    m = daily_matrix()
    valid = m.dropna(how="all")
    return valid.index.min(), valid.index.max()


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return " ".join(s.replace("-", " ").replace(",", " ").replace(".", " ").split())


def resolve_station(query) -> pd.DataFrame:
    """Busca estaciones por código, nombre corto, nombre o municipio (tolerante a tildes).

    Devuelve un DataFrame (posiblemente vacío) con las estaciones que coinciden.
    """
    st = active_stations()
    if query is None or str(query).strip() == "":
        return st.iloc[0:0]
    q = str(query).strip()
    if q.isdigit():
        return st[st["codigo"] == int(q)]
    qn = _norm(q)
    exact = st[st["nombre_corto"].map(_norm) == qn]
    if len(exact):
        return exact
    by_mun = st[st["municipio"].map(_norm) == qn]
    if len(by_mun):
        return by_mun
    tokens = [t for t in qn.split() if len(t) > 2 and t not in {"estacion", "medellin"}]
    if not tokens:
        tokens = qn.split()
    haystack = (st["nombre"] + " " + st["nombre_corto"] + " " + st["municipio"]).map(_norm)
    mask = haystack.apply(lambda h: all(t in h for t in tokens))
    if mask.any():
        return st[mask]
    mask = haystack.apply(lambda h: any(t in h for t in tokens))
    return st[mask]
