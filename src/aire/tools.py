"""Herramientas (tool calling) del agente: esquemas JSON y despachador con manejo de errores."""
from __future__ import annotations

import json
import time
from typing import Any

from . import service
from .rag import get_retriever

def _fn(name: str, desc: str, props: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": required}}}


_EST = {"type": "string", "description": "Estación (p. ej. MED-BEME), barrio, municipio o 'todas'."}
_FECHA = {"type": "string", "description": "AAAA-MM-DD; por defecto la fecha del dashboard."}

# Descripciones breves a propósito: los esquemas se envían en cada llamada al LLM y la capa
# gratuita de Groq limita los tokens por minuto.
TOOL_SCHEMAS = [
    _fn("get_pm25", "PM2.5 de 24 h medido por SIATA, categoría ICA, comparación con norma (37) y "
        "guía OMS (15). estacion='todas' compara todo el valle.",
        {"estacion": _EST, "fecha": _FECHA}, ["estacion"]),
    _fn("predict_tomorrow", "Probabilidad de la categoría ICA de mañana (modelo del proyecto), con "
        "estado HMM y persistencia. Acepta escenario meteorológico opcional.",
        {"estacion": _EST, "fecha": _FECHA,
         "lluvia_mm": {"type": "number"}, "viento_ms": {"type": "number"},
         "radiacion_mj": {"type": "number"}, "nubosidad_pct": {"type": "number"},
         "domingo_o_festivo": {"type": "boolean"}}, ["estacion"]),
    _fn("get_episode_state", "Estado latente HMM (Normal/Alerta/Episodio crítico) por estación y "
        "resumen del valle: ¿hay episodio de contaminación?",
        {"fecha": _FECHA, "estacion": {"type": "string", "description": "Opcional."}}, []),
    _fn("search_docs", "Busca en documentos: Res. 2254/2017 (norma, ICA, niveles), POECA (episodios "
        "y medidas), salud por color ICA, OMS 2021, SIATA, meteorología, metodología del modelo.",
        {"query": {"type": "string"}, "k": {"type": "integer", "description": "1-8, defecto 4"}},
        ["query"]),
]

TOOL_NAMES = [t["function"]["name"] for t in TOOL_SCHEMAS]


def _search_docs(query: str, k: int = 4) -> dict:
    if not query or not str(query).strip():
        raise service.ToolError("La consulta está vacía.")
    res = get_retriever().search(str(query), k=k or 4)
    return {"consulta": query, "recuperador": get_retriever().backend,
            "resultados": [{"cita": f"[{r['doc_id']}: {r['seccion']}]", **r} for r in res],
            "instruccion": "Cite cada afirmación con la etiqueta 'cita' del fragmento usado."}


def _short_cat(cat: str) -> str:
    return (cat.replace("Dañina a la salud de grupos sensibles", "Dañina GS (naranja)")
               .replace("Dañina (grupos sensibles o peor)", "Dañina GS o peor"))


def compact_for_llm(name: str, obs: dict) -> dict:
    """Versión reducida de la observación para el LLM (ahorra tokens: límites de la capa gratuita).

    La observación completa se conserva en la traza (dashboard y evaluación).
    """
    if not obs.get("ok"):
        return {"ok": False, "error": obs.get("error")}
    r = obs["result"]
    if name == "get_pm25":
        ests = []
        for e in r["estaciones"]:
            if "error" in e:
                ests.append({"estacion": e["nombre_corto"], "error": e["error"][:160]})
                continue
            d = {"estacion": e["nombre_corto"], "municipio": e["municipio"], "fecha": e["fecha_dato"],
                 "pm25": e["pm25_promedio_24h_ugm3"], "ica": e["ica"], "categoria": _short_cat(e["categoria_ica"]),
                 "cumple_norma_24h_37": e["cumple_norma_24h_37"], "veces_guia_oms_15": e["veces_guia_oms_24h_15"]}
            if len(r["estaciones"]) <= 3:
                d["max_horario"] = e["pm25_max_horario_ugm3"]
                d["ultimos_7_dias"] = list(e.get("ultimos_dias", {}).values())
            if "nota" in e:
                d["nota"] = e["nota"]
            ests.append(d)
        out = {"ok": True, "fecha": r["fecha_consultada"], "estaciones": ests, "cita": "[SIATA]"}
        if r.get("resumen"):
            out["resumen"] = r["resumen"]
        return out
    if name == "predict_tomorrow":
        if "predicciones" in r:
            return {"ok": True, "predicciones": [compact_for_llm(name, {"ok": True, "result": p})
                                                 for p in r["predicciones"]]}
        est = max(r["estado_hmm_hoy"], key=r["estado_hmm_hoy"].get)
        out = {"ok": True, "estacion": r["estacion"]["nombre_corto"], "municipio": r["estacion"]["municipio"],
               "hoy": r["fecha_base"], "manana": r["fecha_pronostico"], "pm25_hoy": r["pm25_hoy_ugm3"],
               "categoria_hoy": _short_cat(r["categoria_hoy"]),
               "prob_manana": {_short_cat(k): v for k, v in r["probabilidades_manana"].items()},
               "categoria_predicha": _short_cat(r["categoria_predicha"]),
               "persistencia": _short_cat(r["linea_base_persistencia"]),
               "estado_hmm_hoy": f"{est} ({r['estado_hmm_hoy'][est]:.2f})",
               "escenario": r["escenario_meteorologico_manana"],
               "factores": [f"{f['descripcion']}: {f['efecto']}" for f in r["factores_principales"][:3]],
               "desempeno": r["desempeno_historico"], "cita": "[Modelo del proyecto]"}
        # El valor real de mañana NO se envía al LLM: en la demo "mañana" es futuro (solo el
        # dashboard lo muestra como verificación del backtesting).
        return out
    if name == "get_episode_state":
        out = {"ok": True, "fecha": r["fecha"], "n_estaciones": r["n_estaciones"],
               "por_estado_hmm": r["estaciones_por_estado_hmm"],
               "estaciones_>=38": r["estaciones_en_rango_prevencion_o_mas (>=38 µg/m3)"],
               "detalle": [f"{d['estacion']}: estado HMM {d['estado_mas_probable']}; {d['pm25']} µg/m3, ICA "
                           f"{_short_cat(d['categoria_ica'])}" for d in r["detalle"]],
               "nota": ("Estados HMM: Normal ~12, Alerta ~18, Episodio crítico ~30 µg/m3. Son regímenes estadísticos, "
                        "NO categorías ICA ni declaratorias del AMVA: un día en 'Episodio crítico' puede tener ICA "
                        "Aceptable y cumplir la norma (37 µg/m3)."),
               "cita": "[Modelo del proyecto]"}
        if "secuencia_viterbi_14_dias" in r:
            out["viterbi_14_dias"] = list(r["secuencia_viterbi_14_dias"].values())
        return out
    if name == "search_docs":
        return {"ok": True, "fragmentos": [{"cita": f["cita"], "texto": f["texto"][:650]}
                                           for f in r["resultados"]],
                "instruccion": "Cite con la etiqueta 'cita' exacta."}
    return obs


def _tomorrow_to_base(fecha, default_date):
    """Si el LLM pasa a predict_tomorrow la fecha de MAÑANA (fecha objetivo), se usa hoy como base.

    En la evaluación con Groq el modelo pasaba la fecha de mañana, recibía un error y reintentaba
    con la estación seleccionada en el dashboard en vez de la pedida."""
    if not fecha or default_date is None:
        return fecha
    import pandas as pd
    try:
        f, hoy = pd.Timestamp(str(fecha)).normalize(), pd.Timestamp(str(default_date)).normalize()
    except Exception:  # noqa: BLE001
        return fecha
    return str(hoy.date()) if f == hoy + pd.Timedelta(days=1) else fecha


def _check_not_future(fecha, default_date) -> None:
    """La fecha del dashboard es "hoy": no se permite consultar mediciones posteriores
    (evita que el agente "vea el futuro" en la demostración con datos históricos)."""
    if not fecha or default_date is None or str(fecha).strip().lower() in ("hoy", "today"):
        return
    import pandas as pd
    try:
        f, hoy = pd.Timestamp(str(fecha)).normalize(), pd.Timestamp(str(default_date)).normalize()
    except Exception:  # noqa: BLE001 - el formato lo valida después service.parse_date
        return
    if f > service.data_range()[1]:
        return  # fuera del periodo con datos: service.parse_date da el mensaje adecuado
    if f > hoy:
        raise service.ToolError(
            f"La fecha {f.date()} es posterior a la fecha de consulta ({hoy.date()}), que es 'hoy' en "
            "el dashboard: no hay mediciones del futuro. Para mañana use predict_tomorrow sin fecha y con "
            "LA MISMA estación que pidió el usuario.")


def call_tool(name: str, args: dict | str | None, default_date=None) -> dict[str, Any]:
    """Ejecuta una herramienta. Nunca lanza excepción: los errores vuelven como observación."""
    t0 = time.time()
    try:
        if isinstance(args, str):
            args = json.loads(args) if args.strip() else {}
        args = dict(args or {})
        if name == "predict_tomorrow":
            args["fecha"] = _tomorrow_to_base(args.get("fecha"), default_date)
        _check_not_future(args.get("fecha"), default_date)
        if name == "get_pm25":
            if "estacion" not in args:
                raise service.ToolError("Falta el argumento obligatorio 'estacion'.")
            result = service.get_pm25(args["estacion"], args.get("fecha"), default_date=default_date)
        elif name == "predict_tomorrow":
            if "estacion" not in args:
                raise service.ToolError("Falta el argumento obligatorio 'estacion'.")
            result = service.predict_tomorrow(
                args["estacion"], args.get("fecha"), args.get("lluvia_mm"), args.get("viento_ms"),
                args.get("radiacion_mj"), args.get("nubosidad_pct"), default_date=default_date,
                domingo_o_festivo=args.get("domingo_o_festivo"))
        elif name == "get_episode_state":
            result = service.episode_state(args.get("fecha"), args.get("estacion"),
                                           default_date=default_date)
        elif name == "search_docs":
            result = _search_docs(args.get("query", ""), args.get("k", 4))
        else:
            raise service.ToolError(f"Herramienta desconocida '{name}'. Disponibles: {TOOL_NAMES}.")
        return {"ok": True, "result": result, "segundos": round(time.time() - t0, 3)}
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"Argumentos JSON inválidos: {exc}", "segundos": round(time.time() - t0, 3)}
    except service.ToolError as exc:
        return {"ok": False, "error": str(exc), "segundos": round(time.time() - t0, 3)}
    except Exception as exc:  # noqa: BLE001 - errores inesperados también se devuelven como observación
        return {"ok": False, "error": f"Error interno en {name}: {type(exc).__name__}: {exc}",
                "segundos": round(time.time() - t0, 3)}
