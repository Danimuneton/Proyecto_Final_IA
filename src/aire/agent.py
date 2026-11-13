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
from .tools import TOOL_SCHEMAS, call_tool, compact_for_llm

try:  # carga variables desde .env si existe
    from dotenv import load_dotenv
    load_dotenv(C.ROOT / ".env")
except Exception:  # noqa: BLE001
    pass

PROVIDERS = {
    "groq": {"base_url": "https://api.groq.com/openai/v1", "model": "openai/gpt-oss-120b",
             "key_env": "GROQ_API_KEY", "fallback": "openai/gpt-oss-20b"},
    "nvidia": {"base_url": "https://integrate.api.nvidia.com/v1", "model": "meta/llama-3.3-70b-instruct",
               "key_env": "NVIDIA_API_KEY"},
    "ollama": {"base_url": "http://localhost:11434/v1", "model": "qwen2.5:7b", "key_env": None},
    "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini", "key_env": "OPENAI_API_KEY"},
}

MAX_OBS_CHARS = 3500
MAX_WAIT_RATE_LIMIT = 25  # segundos máximos de espera ante un 429
MAX_NUDGES = 2
DOC_TAGS = ("RES2254", "POECA", "SALUD", "OMS2021", "SIATA", "METEO", "METODO")
CITE_RE = re.compile(r"[\[【]\s*(?:cita:\s*\[)?\s*(RES2254|POECA|SALUD|OMS2021|OMS|SIATA|METEO|METODO|Modelo del proyecto)\b[^\]】]*[\]】]",
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
    called = {s.tool for s in steps}
    if (re.search(r"grupos sensibles|asma|recomendaci|mascarilla|tapabocas|N95|ejercicio", answer, re.I)
            and "search_docs" not in called):
        return ("Diste recomendaciones de salud sin consultar documentos. Llama a search_docs (p. ej. "
                "'recomendaciones de salud ICA <categoría>') y basa las recomendaciones solo en lo que devuelva.")
    bad = sorted(c for c in cited if c not in have)
    if bad:
        return (f"Citaste {bad} pero esas fuentes no fueron devueltas por tus herramientas. Llama a "
                "search_docs (o a la herramienta de datos) para verificar, o elimina esas citas.")
    return None


def llm_config() -> dict:
    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    base = PROVIDERS.get(provider, PROVIDERS["groq"])
    key = os.getenv("LLM_API_KEY") or (os.getenv(base["key_env"]) if base["key_env"] else None)
    if key and ("su_llave" in key or key.strip() in ("", "gsk_...", "nvapi-...")):
        key = None  # valor de ejemplo de .env.example: se trata como "sin llave"
    if provider == "ollama" and not key:
        key = "ollama"
    return {"provider": provider,
            "base_url": os.getenv("LLM_BASE_URL", base["base_url"]),
            "model": os.getenv("LLM_MODEL", base["model"]),
            "fallback_model": os.getenv("LLM_FALLBACK_MODEL", base.get("fallback", "")),
            "api_key": key}


SYSTEM_PROMPT = """Eres "AireBot Aburrá": explicas la calidad del aire (PM2.5) del Valle de Aburrá en español claro y breve.
Contexto: datos SIATA del {inicio} al {fin}. "Hoy" = {fecha} (fecha del dashboard); "mañana" = el día siguiente. Estación seleccionada: {estacion}: úsala SOLO si el usuario no menciona otra estación o municipio. Si menciona un municipio, usa ese municipio aunque una herramienta dé error; nunca respondas con otra estación en su lugar.
Siglas: POECA = Plan Operacional para Enfrentar Episodios de Contaminación Atmosférica; ICA = Índice de Calidad del Aire; AMVA = Área Metropolitana del Valle de Aburrá.
Reglas:
1. Cifras de PM2.5/ICA o comparar estaciones -> get_pm25. Mañana -> predict_tomorrow. ¿Episodio? -> get_episode_state. Norma, niveles, POECA, salud, OMS, meteorología o metodología -> search_docs. Preguntas de salud, ejercicio o grupos sensibles: usa SIEMPRE también search_docs.
2. No inventes cifras, normas ni recomendaciones. Las categorías ICA salen solo de get_pm25; los estados del HMM no son colores del ICA. predict_tomorrow da probabilidades de categoría, no una concentración. Para la norma usa cumple_norma_24h_37 y para la OMS veces_guia_oms_15 (p. ej. "1,8 veces la guía OMS").
3. Si una herramienta da error, explícalo y, si puedes, corrige los argumentos y reintenta.
4. Cita cada afirmación con la etiqueta 'cita' que devuelve la herramienta, p. ej. [RES2254: Puntos de corte del ICA para PM2.5 (Artículo 20, Tabla 6)], [SIATA] o [Modelo del proyecto]. Cita documentos SOLO si los recuperaste con search_docs. Termina con "Fuentes:".
5. En salud aclara que no es consejo médico; el pronóstico no es oficial (SIATA/AMVA).
6. Si la pregunta no trata de aire, clima o salud ambiental del valle, dilo y no uses herramientas.
7. Máximo ~170 palabras."""


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
        """Llama al LLM con reintentos. Ante 429 espera lo que indica 'retry-after' y, si el
        límite persiste, prueba el modelo de respaldo (p. ej. gpt-oss-20b en Groq)."""
        models = [self.model] + ([self.cfg["fallback_model"]] if self.cfg.get("fallback_model")
                                 and self.cfg["fallback_model"] != self.model else [])
        blocked = getattr(self, "_blocked_until", {})
        self._blocked_until = blocked
        available = [m for m in models if blocked.get(m, 0) <= time.time()] or models[-1:]
        last_exc = None
        for model in available:
            kwargs = dict(model=model, messages=messages, temperature=self.temperature, max_tokens=1024)
            if use_tools:
                kwargs.update(tools=TOOL_SCHEMAS, tool_choice="auto")
            if "gpt-oss" in model and self.cfg["provider"] == "groq":
                kwargs["extra_body"] = {"reasoning_effort": "low"}  # menos tokens de razonamiento
            for attempt in range(3):
                try:
                    resp = self.client.chat.completions.create(**kwargs)
                    self.last_model_used = model
                    return resp
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    name = type(exc).__name__
                    status = getattr(exc, "status_code", None)
                    if name in ("AuthenticationError", "PermissionDeniedError") or status in (401, 403):
                        raise
                    if name == "NotFoundError" or status == 404:
                        break  # el modelo no existe: probar el de respaldo
                    if name == "RateLimitError" or status in (413, 429):
                        wait = _retry_after(exc)
                        if wait is None or wait > MAX_WAIT_RATE_LIMIT or attempt == 2:
                            # límite largo (p. ej. tokens por día): no insistir con este modelo
                            blocked[model] = time.time() + (wait if wait else 60)
                            break
                        time.sleep(wait + 0.5)
                        continue
                    # 5xx, red o tool_use_failed (400): reintento con espera corta
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
                    content = json.dumps(compact_for_llm(tc.function.name, obs), ensure_ascii=False,
                                         separators=(",", ":"))
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
            return AgentResult(answer, steps, _collect_sources(steps), it, None, getattr(self, "last_model_used", self.model),
                               round(time.time() - t0, 2), nudges)
        except Exception as exc:  # noqa: BLE001
            msg = _friendly_api_error(exc)
            if steps:
                msg += "\n\n" + _fallback_summary(steps)
            return AgentResult(msg, steps, _collect_sources(steps), it, f"{type(exc).__name__}: {exc}",
                               self.model, round(time.time() - t0, 2), nudges)


def _retry_after(exc: Exception) -> float | None:
    """Segundos de espera sugeridos por el proveedor (encabezado retry-after o texto del error)."""
    try:
        h = exc.response.headers  # type: ignore[attr-defined]
        if h.get("retry-after"):
            return float(h.get("retry-after"))
    except Exception:  # noqa: BLE001
        pass
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", str(exc))
    if m:
        return 60 * float(m.group(1) or 0) + float(m.group(2))
    return None


def _fallback_summary(steps: list[Step]) -> str:
    """Resumen sin LLM de los resultados de las herramientas (para no perder la consulta)."""
    lines = ["**Lo que alcanzaron a devolver las herramientas:**"]
    for s in steps:
        if not s.ok:
            lines.append(f"- {s.tool}: {s.observation.get('error', 'error')}")
            continue
        c = compact_for_llm(s.tool, s.observation)
        if s.tool == "get_pm25":
            for e in c["estaciones"][:5]:
                if "pm25" in e:
                    lines.append(f"- {e['estacion']} ({e['fecha']}): {e['pm25']} µg/m³, ICA {e['ica']} "
                                 f"{e['categoria']} [SIATA]")
        elif s.tool == "predict_tomorrow" and "prob_manana" in c:
            probs = ", ".join(f"{k} {v:.0%}" for k, v in c["prob_manana"].items())
            lines.append(f"- Pronóstico {c['manana']} en {c['estacion']}: {c['categoria_predicha']} "
                         f"({probs}) [Modelo del proyecto]")
        elif s.tool == "get_episode_state":
            lines.append(f"- Estados HMM {c['fecha']}: {c['por_estado_hmm']} [Modelo del proyecto]")
        elif s.tool == "search_docs":
            lines.append("- Documentos: " + "; ".join(f["cita"] for f in c["fragmentos"]))
    return "\n".join(lines)


def _friendly_api_error(exc: Exception) -> str:
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    if name == "AuthenticationError" or status == 401:
        return "⚠️ La llave de API del LLM no es válida. Revise el archivo .env (ver README)."
    if name == "RateLimitError" or status == 429:
        return ("⚠️ Se alcanzó el límite de uso gratuito del proveedor del LLM (tokens por minuto o por día). "
                "Espere un minuto e intente de nuevo; si persiste, el cupo diario se agotó.")
    if name == "NotFoundError" or status == 404:
        return "⚠️ El modelo configurado no existe en el proveedor. Revise LLM_MODEL en .env."
    if name in ("APIConnectionError", "APITimeoutError", "ConnectError"):
        return "⚠️ No hay conexión con el proveedor del LLM. Verifique su internet o el endpoint."
    return f"⚠️ Error al consultar el LLM ({name}). Los datos y gráficas del dashboard siguen disponibles."
