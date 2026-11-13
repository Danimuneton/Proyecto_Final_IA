"""Herramientas (tool calling) del agente: esquemas JSON y despachador con manejo de errores."""
from __future__ import annotations

import json
import time
from typing import Any

from . import service
from .rag import get_retriever

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_pm25",
            "description": (
                "Consulta el PM2.5 promedio de 24 h medido por el SIATA en una estación o municipio "
                "del Valle de Aburrá para una fecha, con su categoría ICA (Res. 2254/2017), máximo "
                "horario, comparación con la norma (37 µg/m3) y la guía OMS (15 µg/m3), y los últimos "
                "7 días. Use estacion='todas' para comparar/ordenar todas las estaciones."),
            "parameters": {
                "type": "object",
                "properties": {
                    "estacion": {"type": "string", "description":
                                 "Nombre corto (p. ej. MED-BEME), código, barrio/nombre o municipio "
                                 "(p. ej. 'Itagüí', 'Belén'), o 'todas'."},
                    "fecha": {"type": "string", "description":
                              "Fecha AAAA-MM-DD. Si se omite se usa la fecha de consulta del dashboard."},
                },
                "required": ["estacion"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_tomorrow",
            "description": (
                "Predice la categoría ICA del PM2.5 del día siguiente (Buena / Aceptable / Dañina para "
                "grupos sensibles o peor) en una estación, con probabilidades, factores principales, "
                "estado latente del HMM y la línea base de persistencia. Permite simular un escenario "
                "meteorológico para mañana."),
            "parameters": {
                "type": "object",
                "properties": {
                    "estacion": {"type": "string", "description": "Estación o municipio."},
                    "fecha": {"type": "string", "description":
                              "Fecha base AAAA-MM-DD (se predice el día siguiente). Opcional."},
                    "lluvia_mm": {"type": "number", "description": "Escenario: lluvia de mañana en mm (opcional)."},
                    "viento_ms": {"type": "number", "description": "Escenario: viento medio de mañana en m/s (opcional)."},
                    "radiacion_mj": {"type": "number", "description": "Escenario: radiación solar de mañana en MJ/m2 (opcional)."},
                    "nubosidad_pct": {"type": "number", "description": "Escenario: nubosidad media de mañana en % (opcional)."},
                    "domingo_o_festivo": {"type": "boolean", "description":
                                          "Escenario: si mañana es domingo o festivo (menos tráfico). Opcional."},
                },
                "required": ["estacion"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_episode_state",
            "description": (
                "Estima con el Modelo Oculto de Markov el estado latente de la calidad del aire "
                "(Normal / Alerta / Episodio crítico) por estación en una fecha, y resume cuántas "
                "estaciones están en cada estado y cuántas superan 38 µg/m3 (rango de Prevención). "
                "Útil para saber si el valle está en un episodio de contaminación."),
            "parameters": {
                "type": "object",
                "properties": {
                    "fecha": {"type": "string", "description": "Fecha AAAA-MM-DD (opcional)."},
                    "estacion": {"type": "string", "description":
                                 "Opcional: una estación o municipio; si se omite, todo el valle."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "Busca en el corpus documental (RAG): Resolución 2254 de 2017 (norma, ICA, niveles de "
                "prevención/alerta/emergencia), POECA del AMVA (episodios y medidas), recomendaciones de "
                "salud por color del ICA, guías OMS 2021, datos del SIATA, meteorología del valle y "
                "metodología de los modelos. Devuelve fragmentos con su fuente para citar."),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Consulta en español."},
                    "k": {"type": "integer", "description": "Número de fragmentos (1-8, por defecto 4)."},
                },
                "required": ["query"],
            },
        },
    },
]

TOOL_NAMES = [t["function"]["name"] for t in TOOL_SCHEMAS]


def _search_docs(query: str, k: int = 4) -> dict:
    if not query or not str(query).strip():
        raise service.ToolError("La consulta está vacía.")
    res = get_retriever().search(str(query), k=k or 4)
    return {"consulta": query, "recuperador": get_retriever().backend,
            "resultados": [{"cita": f"[{r['doc_id']}: {r['seccion']}]", **r} for r in res],
            "instruccion": "Cite cada afirmación con la etiqueta 'cita' del fragmento usado."}


def call_tool(name: str, args: dict | str | None, default_date=None) -> dict[str, Any]:
    """Ejecuta una herramienta. Nunca lanza excepción: los errores vuelven como observación."""
    t0 = time.time()
    try:
        if isinstance(args, str):
            args = json.loads(args) if args.strip() else {}
        args = dict(args or {})
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
