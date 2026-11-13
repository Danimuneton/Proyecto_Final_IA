"""Agente LLM con loop agéntico manual (razonar -> llamar herramienta -> observar -> responder).

- Cliente compatible con OpenAI (Groq por defecto; también NVIDIA NIM, Ollama u otro endpoint).
- Máximo T_MAX iteraciones; si se alcanza, se pide una respuesta final sin herramientas.
- Los errores de las herramientas se devuelven al modelo como observación (no rompen el loop).
- Los errores de la API (llave inválida, límite de uso, red) se devuelven como mensaje claro.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

from . import config as C
from .tools import TOOL_SCHEMAS, call_tool

try:  # carga variables desde .env si existe
    from dotenv import load_dotenv
    load_dotenv(C.ROOT / ".env")
except Exception:  # noqa: BLE001
    pass

PROVIDERS = {
    "groq": {"base_url": "https://api.groq.com/openai/v1", "model": "openai/gpt-oss-120b",
             "key_env": "GROQ_API_KEY"},
    "nvidia": {"base_url": "https://integrate.api.nvidia.com/v1", "model": "meta/llama-3.3-70b-instruct",
               "key_env": "NVIDIA_API_KEY"},
    "ollama": {"base_url": "http://localhost:11434/v1", "model": "qwen2.5:7b", "key_env": None},
    "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini", "key_env": "OPENAI_API_KEY"},
}

MAX_OBS_CHARS = 9000
MAX_NUDGES = 2
DOC_TAGS = ("RES2254", "POECA", "SALUD", "OMS2021", "SIATA", "METEO", "METODO")
CITE_RE = re.compile(r"\[\s*(RES2254|POECA|SALUD|OMS2021|OMS|SIATA|METEO|METODO|Modelo del proyecto)\b[^\]]*\]",
                     re.IGNORECASE)


def retrieved_tags(steps: list["Step"]) -> set[str]:
    """Etiquetas de fuente que el agente realmente obtuvo de sus herramientas."""
    tags = set()
    for s in steps:
        if not s.ok:
            continue
        if s.tool == "search_docs":
            tags |= {r["doc_id"].upper() for r in s.observation["result"]["resultados"]}
        elif s.tool == "get_pm25":
            tags.add("SIATA")
        elif s.tool in ("predict_tomorrow", "get_episode_state"):
            tags |= {"SIATA", "MODELO DEL PROYECTO"}
    return tags


def _grounding_problem(answer: str, steps: list["Step"]) -> str | None:
    """Detecta respuestas no fundamentadas: citas inventadas o cifras sin herramientas."""
    cited = {m.group(1).upper() for m in CITE_RE.finditer(answer)}
    have = retrieved_tags(steps)
    if "[cita]" in answer.lower():
        return ("Usaste la etiqueta genérica [cita]. Reemplázala por la etiqueta 'cita' exacta de cada "
                "fragmento devuelto por search_docs, o por [SIATA] / [Modelo del proyecto].")
    if not steps and cited:
        return ("Respondiste con citas pero no llamaste ninguna herramienta en esta conversación. Llama "
                "primero a la herramienta adecuada (get_pm25, predict_tomorrow, get_episode_state o "
                "search_docs) y responde solo con lo que devuelva.")
    bad = sorted(c for c in cited if c not in have)
    if bad:
        return (f"Citaste {bad} pero esas fuentes no fueron devueltas por tus herramientas. Llama a "
                "search_docs (o a la herramienta de datos) para verificar, o elimina esas citas.")
    return None


def llm_config() -> dict:
    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    base = PROVIDERS.get(provider, PROVIDERS["groq"])
    key = os.getenv("LLM_API_KEY") or (os.getenv(base["key_env"]) if base["key_env"] else None)
    if provider == "ollama" and not key:
        key = "ollama"
    return {"provider": provider,
            "base_url": os.getenv("LLM_BASE_URL", base["base_url"]),
            "model": os.getenv("LLM_MODEL", base["model"]),
            "api_key": key}


SYSTEM_PROMPT = """Eres "AireBot Aburrá", un asistente que explica la calidad del aire (PM2.5) del Valle de Aburrá a la ciudadanía, en español claro y breve.

CONTEXTO DE LA SESIÓN
- Los datos disponibles son del SIATA entre {inicio} y {fin}. La "fecha de consulta" elegida en el dashboard es {fecha} y se considera "hoy". "Mañana" es el día siguiente a esa fecha.
- Estación seleccionada en el dashboard: {estacion}. Úsala si el usuario no menciona otra.

CÓMO TRABAJAR
1. Para cifras de PM2.5, ICA o comparar estaciones usa get_pm25. Para el día siguiente usa predict_tomorrow. Para saber si hay un episodio o el estado latente del valle usa get_episode_state. Para normas, niveles, protocolos (POECA), recomendaciones de salud, guías OMS, meteorología o cómo funciona el modelo usa search_docs.
2. Si la pregunta combina datos y recomendaciones (p. ej. "¿puedo salir a correr mañana en Belén?"), llama a la herramienta de datos y también a search_docs. Para cualquier pregunta de salud, ejercicio o grupos sensibles llama SIEMPRE a search_docs.
3. Nunca inventes cifras, fechas ni normas: usa solo lo que devuelven las herramientas. predict_tomorrow entrega probabilidades de categoría, no una concentración: no inventes un valor de PM2.5 para mañana. Para comparar con la norma (37 µg/m3) usa el campo cumple_norma_24h_37 y para la guía OMS (15 µg/m3) el campo veces_guia_oms_24h_15 (p. ej., "equivale a 1,8 veces la guía de la OMS"). Si una herramienta devuelve un error, explica el problema al usuario y, si es posible, corrige los argumentos y vuelve a intentar (por ejemplo, con una estación de la lista que entrega el error).
4. CITAS OBLIGATORIAS: después de cada afirmación tomada de documentos, escribe la etiqueta 'cita' del fragmento, por ejemplo [RES2254: Puntos de corte del ICA para PM2.5 (Artículo 20, Tabla 6)]. Para datos medidos cita [SIATA] y para pronósticos o estados del HMM cita [Modelo del proyecto]. Cita un documento SOLO si lo recuperaste con search_docs en esta conversación. Termina con una línea "Fuentes:" que liste las fuentes usadas.
5. Si hablas de salud, aclara que es información general y no consejo médico. No presentes el pronóstico como oficial: las declaratorias las hace el AMVA y los pronósticos oficiales el SIATA.
6. Si la pregunta no tiene relación con calidad del aire, contaminación, clima o salud ambiental del Valle de Aburrá, dilo amablemente y no uses herramientas.
7. Responde en máximo unos 180 palabras salvo que el usuario pida detalle."""


@dataclass
class Step:
    tool: str
    args: dict
    ok: bool
    observation: dict
    seconds: float


@dataclass
class AgentResult:
    answer: str
    steps: list[Step] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    iterations: int = 0
    error: str | None = None
    model: str = ""
    seconds: float = 0.0
    verificaciones: int = 0  # veces que el verificador de citas pidió corregir la respuesta

    @property
    def tools_called(self) -> list[str]:
        return [s.tool for s in self.steps]


def _collect_sources(steps: list[Step]) -> list[dict]:
    seen, out = set(), []
    for s in steps:
        if not s.ok:
            continue
        res = s.observation.get("result", {})
        if s.tool == "search_docs":
            for r in res.get("resultados", []):
                key = (r["doc_id"], r["seccion"])
                if key not in seen:
                    seen.add(key)
                    out.append({"tipo": "documento", "cita": r["cita"], "titulo": r["titulo"],
                                "url": r["url"]})
        else:
            key = ("SIATA",) if s.tool == "get_pm25" else ("MODELO",)
            if key not in seen:
                seen.add(key)
                out.append({"tipo": "datos" if s.tool == "get_pm25" else "modelo",
                            "cita": "[SIATA]" if s.tool == "get_pm25" else "[Modelo del proyecto]",
                            "titulo": res.get("fuente", ""),
                            "url": "https://siata.gov.co/EntregaData1/Datos_SIATA_Aire_pm25.json"})
    return out


class AirQualityAgent:
    def __init__(self, client=None, model: str | None = None, t_max: int = C.T_MAX,
                 temperature: float = 0.2):
        cfg = llm_config()
        self.cfg = cfg
        self.model = model or cfg["model"]
        self.t_max = t_max
        self.temperature = temperature
        if client is not None:
            self.client = client
        else:
            if not cfg["api_key"]:
                raise RuntimeError(
                    "No hay llave de API para el LLM. Cree el archivo .env con GROQ_API_KEY=... "
                    "(ver README) o configure LLM_PROVIDER=ollama.")
            from openai import OpenAI
            # Un LLM local en CPU (Ollama) puede tardar varios minutos por respuesta
            default_timeout = 900 if cfg["provider"] == "ollama" else 60
            self.client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"],
                                 timeout=float(os.getenv("LLM_TIMEOUT", default_timeout)), max_retries=0)

    def _system(self, fecha, estacion) -> dict:
        from .service import data_range
        lo, hi = data_range()
        return {"role": "system", "content": SYSTEM_PROMPT.format(
            inicio=lo.date(), fin=hi.date(), fecha=fecha or hi.date(), estacion=estacion or "ninguna")}

    def _complete(self, messages: list[dict], use_tools: bool = True):
        kwargs = dict(model=self.model, messages=messages, temperature=self.temperature,
                      max_tokens=1500)
        if use_tools:
            kwargs.update(tools=TOOL_SCHEMAS, tool_choice="auto")
        last_exc = None
        for attempt in range(3):
            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                name = type(exc).__name__
                status = getattr(exc, "status_code", None)
                if name in ("AuthenticationError", "PermissionDeniedError", "NotFoundError") or status in (401, 403, 404):
                    raise
                # 429 (límite), 5xx, red o tool_use_failed (400): reintento con espera
                time.sleep(2 * (attempt + 1))
        raise last_exc

    def run(self, question: str, history: list[dict] | None = None, fecha=None,
            estacion: str | None = None) -> AgentResult:
        t0 = time.time()
        messages = [self._system(fecha, estacion)]
        for h in (history or [])[-6:]:
            if h.get("role") in ("user", "assistant") and h.get("content"):
                messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": question})
        steps: list[Step] = []
        answer = None
        it = 0
        nudges = 0
        try:
            for it in range(1, self.t_max + 1):
                resp = self._complete(messages)
                msg = resp.choices[0].message
                tool_calls = getattr(msg, "tool_calls", None) or []
                if not tool_calls:
                    answer = (msg.content or "").strip()
                    problem = _grounding_problem(answer, steps)
                    if problem and nudges < MAX_NUDGES and it < self.t_max:
                        # Verificador de citas: se devuelve la observación al modelo para que corrija
                        nudges += 1
                        messages.append({"role": "assistant", "content": answer})
                        messages.append({"role": "user", "content": f"[Verificación automática] {problem}"})
                        answer = None
                        continue
                    break
                messages.append({"role": "assistant", "content": msg.content or "",
                                 "tool_calls": [{"id": tc.id, "type": "function",
                                                 "function": {"name": tc.function.name,
                                                              "arguments": tc.function.arguments or "{}"}}
                                                for tc in tool_calls]})
                for tc in tool_calls:
                    obs = call_tool(tc.function.name, tc.function.arguments, default_date=fecha)
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {"_raw": tc.function.arguments}
                    steps.append(Step(tc.function.name, args, obs["ok"], obs, obs.get("segundos", 0)))
                    content = json.dumps(obs, ensure_ascii=False)
                    if len(content) > MAX_OBS_CHARS:
                        content = content[:MAX_OBS_CHARS] + '... [observación truncada]"}'
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
            if answer is None:
                # Se alcanzó T_MAX: respuesta final con la información reunida, sin más herramientas
                messages.append({"role": "user", "content":
                                 "Se alcanzó el número máximo de pasos. Responde ahora con la información "
                                 "que ya tienes, citando las fuentes, e indica qué no pudiste verificar."})
                resp = self._complete(messages, use_tools=False)
                answer = (resp.choices[0].message.content or "").strip()
            if not answer:
                answer = "No pude generar una respuesta. Intente reformular la pregunta."
            elif _grounding_problem(answer, steps):
                answer += ("\n\n⚠️ *Verificación automática: esta respuesta contiene citas que no "
                           "provienen de las herramientas consultadas. Contrástela con las fuentes oficiales.*")
            return AgentResult(answer, steps, _collect_sources(steps), it, None, self.model,
                               round(time.time() - t0, 2), nudges)
        except Exception as exc:  # noqa: BLE001
            msg = _friendly_api_error(exc)
            return AgentResult(msg, steps, _collect_sources(steps), it, f"{type(exc).__name__}: {exc}",
                               self.model, round(time.time() - t0, 2), nudges)


def _friendly_api_error(exc: Exception) -> str:
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    if name == "AuthenticationError" or status == 401:
        return "⚠️ La llave de API del LLM no es válida. Revise el archivo .env (ver README)."
    if name == "RateLimitError" or status == 429:
        return "⚠️ Se alcanzó el límite de uso gratuito del proveedor del LLM. Espere un minuto e intente de nuevo."
    if name == "NotFoundError" or status == 404:
        return "⚠️ El modelo configurado no existe en el proveedor. Revise LLM_MODEL en .env."
    if name in ("APIConnectionError", "APITimeoutError", "ConnectError"):
        return "⚠️ No hay conexión con el proveedor del LLM. Verifique su internet o el endpoint."
    return f"⚠️ Error al consultar el LLM ({name}). Los datos y gráficas del dashboard siguen disponibles."
