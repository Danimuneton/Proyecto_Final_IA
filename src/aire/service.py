"""Servicio de consulta usado por las herramientas del agente y por el dashboard.

Carga una sola vez los datos, el HMM y el clasificador entrenados (models/) y expone
funciones puras que devuelven diccionarios serializables a JSON.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from . import config as C
from .classifier import SoftmaxRegression
from .data import active_stations, daily_matrix, load_hourly, load_meteo, resolve_station
from .hmm import GaussianHMM
from .ica import categoria_ica, clase_3
from .modeling import DESCRIPCION_FEATURES, FEATURES_COMPLETO, build_table, hmm_filtered

FUENTE_SIATA = ("SIATA - red de calidad del aire del Valle de Aburrá, archivo abierto "
                "Datos_SIATA_Aire_pm25.json (https://siata.gov.co/EntregaData1/Datos_SIATA_Aire_pm25.json)")
FUENTE_METEO = "Open-Meteo Historical Weather API (reanálisis ERA5, Copernicus/ECMWF)"
FUENTE_MODELO = "Modelos del proyecto (HMM y regresión logística propios), ver documento METODO"


class ToolError(Exception):
    """Error esperado de una herramienta; su mensaje se devuelve al agente como observación."""


@lru_cache(maxsize=1)
def models() -> tuple[GaussianHMM, SoftmaxRegression]:
    hmm_path, clf_path = C.MODELS_DIR / "hmm.json", C.MODELS_DIR / "clasificador.json"
    if not hmm_path.exists() or not clf_path.exists():
        raise ToolError("Faltan los modelos entrenados. Ejecute: python scripts/train_models.py")
    return GaussianHMM.load(hmm_path), SoftmaxRegression.load(clf_path)


@lru_cache(maxsize=1)
def matrix() -> pd.DataFrame:
    return daily_matrix()


@lru_cache(maxsize=1)
def filtered_probs() -> dict[int, np.ndarray]:
    hmm, _ = models()
    return hmm_filtered(hmm, np.log(matrix()))


@lru_cache(maxsize=1)
def feature_table() -> pd.DataFrame:
    return build_table(matrix(), filtered_probs())


@lru_cache(maxsize=1)
def viterbi_paths() -> dict[int, np.ndarray]:
    hmm, _ = models()
    m = np.log(matrix())
    return {c: hmm.viterbi(m[c].to_numpy()) for c in m.columns}


def data_range() -> tuple[pd.Timestamp, pd.Timestamp]:
    m = matrix().dropna(how="all")
    return m.index.min(), m.index.max()


# ---------------------------------------------------------------------------
# Utilidades de validación
# ---------------------------------------------------------------------------
def parse_date(fecha, default: pd.Timestamp | None = None) -> pd.Timestamp:
    lo, hi = data_range()
    if fecha in (None, "", "hoy", "today"):
        if default is None:
            return hi
        return pd.Timestamp(default).normalize()
    try:
        ts = pd.Timestamp(str(fecha)).normalize()
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"Fecha inválida '{fecha}'. Use el formato AAAA-MM-DD.") from exc
    if ts < lo or ts > hi:
        raise ToolError(f"La fecha {ts.date()} está fuera del periodo con datos "
                        f"({lo.date()} a {hi.date()}). El archivo abierto de SIATA usado "
                        "solo cubre ese año; para datos actuales consulte siata.gov.co.")
    return ts


def find_stations(estacion) -> pd.DataFrame:
    st = resolve_station(estacion)
    if st.empty:
        opciones = "; ".join(f"{r.nombre_corto} ({r.municipio})" for r in active_stations().itertuples())
        raise ToolError(f"No encontré la estación o municipio '{estacion}'. "
                        f"Estaciones disponibles: {opciones}.")
    return st


def _station_info(row) -> dict:
    return {"codigo": int(row.codigo), "nombre_corto": row.nombre_corto,
            "nombre": row.nombre, "municipio": row.municipio,
            "latitud": float(row.latitud), "longitud": float(row.longitud)}


def _last_valid(code: int, ts: pd.Timestamp) -> tuple[pd.Timestamp | None, float | None]:
    s = matrix()[code]
    s = s[s.index <= ts].dropna()
    if s.empty:
        return None, None
    return s.index[-1], float(s.iloc[-1])


# ---------------------------------------------------------------------------
# Consultas
# ---------------------------------------------------------------------------
def pm25_station(code: int, ts: pd.Timestamp, n_days: int = 7) -> dict:
    st = active_stations().set_index("codigo").loc[code]
    day, val = _last_valid(code, ts)
    if day is None:
        raise ToolError(f"La estación {st.nombre_corto} no tiene datos válidos hasta {ts.date()}.")
    ica = categoria_ica(val)
    s = matrix()[code]
    hist = s[(s.index <= day) & (s.index > day - pd.Timedelta(days=n_days))]
    h = load_hourly()
    hh = h[(h["codigo"] == code) & (h["fecha"].dt.normalize() == day)]
    out = {
        "fecha_dato": str(day.date()),
        "pm25_promedio_24h_ugm3": round(val, 1),
        "ica": ica["ica"], "categoria_ica": ica["categoria"],
        "pm25_max_horario_ugm3": None if hh["pm25"].isna().all() else round(float(hh["pm25"].max()), 1),
        "horas_validas": int(hh["pm25"].notna().sum()),
        "cumple_norma_24h_37": bool(val <= C.NORMA_24H + 0.5),
        "veces_guia_oms_24h_15": round(val / C.OMS_24H, 2),
        "ultimos_dias": {str(d.date()): (None if np.isnan(v) else round(float(v), 1))
                         for d, v in hist.items()},
    }
    if day != ts:
        out["nota"] = (f"No hay dato válido para {ts.date()}; se muestra el último disponible "
                       f"({day.date()}).")
    return out


def get_pm25(estacion: str, fecha=None, default_date=None) -> dict:
    ts = parse_date(fecha, default_date)
    if str(estacion).strip().lower() in ("todas", "todo", "valle", "all", "*"):
        st = active_stations()
    else:
        st = find_stations(estacion)
    resultados = []
    for row in st.itertuples():
        try:
            resultados.append({**_station_info(row), **pm25_station(int(row.codigo), ts)})
        except ToolError as exc:
            resultados.append({**_station_info(row), "error": str(exc)})
    validos = [r for r in resultados if "pm25_promedio_24h_ugm3" in r]
    validos.sort(key=lambda r: -r["pm25_promedio_24h_ugm3"])
    resumen = None
    if len(st) > 1 and validos:
        cats = pd.Series([r["categoria_ica"] for r in validos]).value_counts().to_dict()
        resumen = {"n_estaciones": len(validos), "estaciones_por_categoria": cats,
                   "max": {"estacion": validos[0]["nombre_corto"], "pm25": validos[0]["pm25_promedio_24h_ugm3"]},
                   "min": {"estacion": validos[-1]["nombre_corto"], "pm25": validos[-1]["pm25_promedio_24h_ugm3"]}}
        if len(st) > 6:  # para no saturar el contexto del LLM
            for r in validos:
                r.pop("ultimos_dias", None)
                r.pop("nombre", None)
    return {"fecha_consultada": str(ts.date()), "resumen": resumen,
            "estaciones": validos + [r for r in resultados if "error" in r],
            "unidades": "µg/m3, promedio de 24 h (mínimo 18 horas válidas)",
            "referencia": "Categorías ICA según Resolución 2254 de 2017 (Tabla 6); norma 24 h = 37 µg/m3; guía OMS 2021 24 h = 15 µg/m3",
            "fuente": FUENTE_SIATA}


def _feature_row(code: int, ts: pd.Timestamp) -> pd.Series:
    t = feature_table()
    rows = t[(t["codigo"] == code) & (t["fecha"] <= ts) & t["pm_hoy"].notna()]
    if rows.empty:
        raise ToolError("No hay datos suficientes para predecir en esa estación y fecha.")
    return rows.iloc[-1].copy()


def predict_tomorrow(estacion: str, fecha=None, lluvia_mm=None, viento_ms=None,
                     radiacion_mj=None, nubosidad_pct=None, default_date=None,
                     domingo_o_festivo=None) -> dict:
    ts = parse_date(fecha, default_date)
    st = find_stations(estacion)
    if len(st) > 1:
        # Para un municipio con varias estaciones, se predice para cada una
        preds = [predict_tomorrow(str(int(c)), str(ts.date()), lluvia_mm, viento_ms,
                                  radiacion_mj, nubosidad_pct, domingo_o_festivo=domingo_o_festivo)
                 for c in st["codigo"]]
        return {"municipio_o_consulta": estacion, "predicciones": preds}
    row = st.iloc[0]
    code = int(row.codigo)
    hmm, clf = models()
    r = _feature_row(code, ts)
    escenario = {}
    meteo = load_meteo().set_index("fecha")
    tomorrow = r["fecha"] + pd.Timedelta(days=1)
    base_meteo = meteo.loc[tomorrow] if tomorrow in meteo.index else None
    for name, val, feat, fn in [
        ("lluvia_mm", lluvia_mm, "lluvia_manana_log", np.log1p),
        ("viento_ms", viento_ms, "viento_manana_ms", float),
        ("radiacion_mj", radiacion_mj, "radiacion_manana_mj", float),
        ("nubosidad_pct", nubosidad_pct, "nubosidad_manana_pct", float),
    ]:
        if val is not None:
            try:
                v = float(val)
            except (TypeError, ValueError) as exc:
                raise ToolError(f"Valor inválido para {name}: {val}") from exc
            if v < 0:
                raise ToolError(f"{name} no puede ser negativo.")
            r[feat] = fn(v)
            escenario[name] = v
        elif base_meteo is not None:
            escenario[name] = round(float(base_meteo[name]), 2)
    if domingo_o_festivo is not None:
        r["domingo_manana"] = 1.0 if bool(domingo_o_festivo) else 0.0
    escenario["domingo_o_festivo"] = bool(r["domingo_manana"])
    if r[FEATURES_COMPLETO].isna().any():
        faltan = [f for f in FEATURES_COMPLETO if pd.isna(r[f])]
        raise ToolError(f"Faltan variables para la predicción ({faltan}). Indique el escenario "
                        "meteorológico (lluvia_mm, viento_ms, radiacion_mj, nubosidad_pct).")
    x = r[FEATURES_COMPLETO].to_numpy(dtype=float)[None, :]
    proba = clf.predict_proba(x)[0]
    pred = int(clf.decision_scores(proba[None, :]).argmax())
    # Contribuciones (escala estandarizada) a favor de la clase predicha
    z = ((x - clf.mean_) / clf.std_)[0]
    contrib = z * (clf.W[:, pred] - clf.W.mean(axis=1))
    order = np.argsort(-np.abs(contrib))[:4]
    factores = [{"variable": FEATURES_COMPLETO[i], "descripcion": DESCRIPCION_FEATURES[FEATURES_COMPLETO[i]],
                 "efecto": "aumenta" if contrib[i] > 0 else "disminuye",
                 "peso": round(float(contrib[i]), 3)} for i in order]
    f = filtered_probs()[code][matrix().index.get_loc(r["fecha"])]
    real = r["pm_manana"]
    out = {
        "estacion": _station_info(row),
        "fecha_base": str(r["fecha"].date()),
        "fecha_pronostico": str(tomorrow.date()),
        "pm25_hoy_ugm3": round(float(r["pm_hoy"]), 1),
        "categoria_hoy": C.CLASES[clase_3(r["pm_hoy"])],
        "probabilidades_manana": {C.CLASES[k]: round(float(p), 3) for k, p in enumerate(proba)},
        "categoria_predicha": C.CLASES[pred],
        "linea_base_persistencia": C.CLASES[clase_3(r["pm_hoy"])],
        "estado_hmm_hoy": {C.ESTADOS_HMM[k]: round(float(p), 3) for k, p in enumerate(f)},
        "escenario_meteorologico_manana": escenario,
        "factores_principales": factores,
        "desempeno_historico": performance_summary(),
        "fuente": f"{FUENTE_MODELO}; datos {FUENTE_SIATA}; meteorología {FUENTE_METEO}",
        "advertencia": "Pronóstico académico; no reemplaza al SIATA ni las declaratorias del AMVA.",
    }
    if not np.isnan(real) and not escenario_is_custom(lluvia_mm, viento_ms, radiacion_mj,
                                                      nubosidad_pct, domingo_o_festivo):
        out["valor_real_observado_manana"] = {"pm25": round(float(real), 1),
                                              "categoria": C.CLASES[clase_3(real)]}
    return out


@lru_cache(maxsize=1)
def performance_summary() -> str:
    import json
    p = C.REPORTS_DIR / "metricas_parte1.json"
    if not p.exists():
        return "Métricas no disponibles (ejecute scripts/train_models.py)."
    r = json.loads(p.read_text(encoding="utf-8"))["resultados"]
    f = next(v for k, v in r.items() if "modelo final" in k)
    b = next(v for k, v in r.items() if "persistencia" in k)
    return (f"Backtesting ene-jul 2024 (fuera de muestra): exactitud {f['accuracy']:.2f}, "
            f"F1 macro {f['macro_f1']:.2f}, recall clase Dañina {f['recall_por_clase'][2]:.2f}; "
            f"línea base persistencia: {b['accuracy']:.2f}, {b['macro_f1']:.2f}, "
            f"{b['recall_por_clase'][2]:.2f}.")


def escenario_is_custom(*vals) -> bool:
    return any(v is not None for v in vals)


def episode_state(fecha=None, estacion=None, default_date=None) -> dict:
    """Estado latente del HMM (filtrado) por estación y resumen del valle para una fecha."""
    ts = parse_date(fecha, default_date)
    st = active_stations() if not estacion else find_stations(estacion)
    m = matrix()
    i = m.index.get_loc(ts)
    filt = filtered_probs()
    rows = []
    for row in st.itertuples():
        p = filt[int(row.codigo)][i]
        val = m.at[ts, row.codigo]
        rows.append({"estacion": row.nombre_corto, "municipio": row.municipio,
                     "pm25": None if np.isnan(val) else round(float(val), 1),
                     "categoria_ica": categoria_ica(val)["categoria"] if not np.isnan(val) else "Sin dato",
                     "estado_mas_probable": C.ESTADOS_HMM[int(np.argmax(p))],
                     "probabilidades": {C.ESTADOS_HMM[k]: round(float(v), 3) for k, v in enumerate(p)}})
    estados = pd.Series([r["estado_mas_probable"] for r in rows]).value_counts().to_dict()
    n_prev = sum(1 for r in rows if r["pm25"] is not None and r["pm25"] >= 37.5)
    out = {"fecha": str(ts.date()), "estaciones_por_estado_hmm": estados,
           "estaciones_en_rango_prevencion_o_mas (>=38 µg/m3)": n_prev,
           "n_estaciones": len(rows),
           "detalle": rows,
           "interpretacion": ("Estados del HMM: Normal (media ~12 µg/m3), Alerta (~18), Episodio "
                              "crítico (~30). Son regímenes estadísticos, no declaratorias oficiales "
                              "del AMVA (que usan la Res. 2254: Prevención 38-55, Alerta 56-150)."),
           "fuente": f"{FUENTE_MODELO}; datos {FUENTE_SIATA}"}
    if estacion and len(st) == 1:
        code = int(st.iloc[0].codigo)
        hmm, _ = models()
        # Viterbi solo con datos hasta la fecha consultada (sin mirar el futuro)
        path = hmm.viterbi(np.log(m[code].to_numpy()[: i + 1]))
        lo = max(0, i - 13)
        out["secuencia_viterbi_14_dias"] = {str(m.index[j].date()): C.ESTADOS_HMM[int(path[j])]
                                            for j in range(lo, i + 1)}
    if len(rows) > 6:
        for r in out["detalle"]:
            r.pop("probabilidades")
    return out
