"""Agente LLM con RAG y 3 herramientas: get_pm25, predict_tomorrow, search_docs.

Uso:  python src/agente.py "¿Cómo estará el aire mañana en Bello?"
Config (archivo .env): LLM_BASE_URL, LLM_API_KEY, LLM_MODEL  (endpoint compatible con OpenAI)
      y opcionalmente LLM_REASONING_EFFORT (low/medium/high) para modelos de razonamiento
"""
import json
import os
import pathlib
import re
import sys
import time
import unicodedata

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import modelo as nb          # noqa: E402  (Parte 1: categorías y predicción)
import rag                   # noqa: E402

BASE = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(BASE / ".env")

T_MAX = 6                    # máximo de iteraciones del loop agéntico
_cliente_cache = {}


# ------------------------------------------------------------------ herramientas
def _sin_tildes(s):
    return "".join(c for c in unicodedata.normalize("NFD", str(s)) if unicodedata.category(c) != "Mn").lower()


def _resolver(df, estacion):
    """Devuelve (codigo, nombre, None) o (None, None, dict_error)."""
    est = df.drop_duplicates("estacion")[["estacion", "nombre"]]
    txt = str(estacion).strip()
    if txt.isdigit():
        r = est[est["estacion"] == int(txt)]
    else:
        r = est[est["nombre"].map(_sin_tildes).str.contains(_sin_tildes(txt), regex=False)]
    if len(r) == 1:
        return int(r.iloc[0]["estacion"]), r.iloc[0]["nombre"], None
    if len(r) == 0:
        return None, None, {"error": f"No encontré ninguna estación que coincida con '{estacion}'.",
                            "estaciones_disponibles": est["nombre"].tolist()}
    return None, None, {"error": f"'{estacion}' coincide con varias estaciones; hay que precisar cuál.",
                        "coincidencias": r["nombre"].tolist()}


def get_pm25(estacion, fecha=None):
    """PM2.5 promedio diario de una estación en una fecha (por defecto, el último día con datos)."""
    df = nb.cargar_datos()
    cod, nombre, err = _resolver(df, estacion)
    if err:
        return err
    g = df[df["estacion"] == cod].set_index("fecha")["pm25"]
    rango = [str(g.index.min().date()), str(g.index.max().date())]
    if fecha in (None, ""):
        dia = g.index.max()
    else:
        try:
            dia = pd.Timestamp(fecha)
        except Exception:
            return {"error": f"Fecha '{fecha}' no válida; usa el formato AAAA-MM-DD."}
    if dia not in g.index:
        return {"error": f"No hay dato de PM2.5 para {nombre} el {dia.date()}.", "rango_datos_disponible": rango}
    v = float(g[dia])
    return {"estacion": nombre, "fecha": str(dia.date()), "pm25_promedio_diario_ug_m3": round(v, 1),
            "categoria_ica": nb.CLASES[int(nb.categorizar([v])[0])],
            "promedio_ultimos_7_dias": round(float(g[(g.index <= dia) & (g.index > dia - pd.Timedelta(days=7))].mean()), 1),
            "rango_datos_disponible": rango, "fuente": "SIATA (promedio diario calculado por el proyecto)"}


def predict_tomorrow(estacion, fecha=None):
    """Predicción (Naive Bayes del proyecto) de la categoría de PM2.5 del día siguiente a 'fecha'."""
    df = nb.cargar_datos()
    cod, nombre, err = _resolver(df, estacion)
    if err:
        return err
    r = nb.predecir_manana(cod, fecha if fecha not in (None, "") else None)
    r["fuente"] = "Modelo Naive Bayes del proyecto, entrenado con datos de SIATA"
    return r


GLOSARIO = {
    "poeca": "Plan Operacional para Enfrentar Episodios de Contaminación Atmosférica",
    "ica": "Índice de Calidad del Aire",
    "siata": "Sistema de Alerta Temprana de Medellín y el Valle de Aburrá",
    "amva": "Área Metropolitana del Valle de Aburrá",
    "pigeca": "Plan Integral de Gestión de la Calidad del Aire",
    "pm2.5": "material particulado menor a 2.5 micras",
}


def expandir_siglas(consulta):
    """Añade el significado de las siglas a la consulta: los embeddings manejan mal las siglas sueltas."""
    extra = [v for k, v in GLOSARIO.items() if re.search(rf"(?<!\w){re.escape(k)}(?!\w)", consulta, re.I)]
    return consulta + (" (" + "; ".join(extra) + ")" if extra else "")


def search_docs(consulta):
    """Busca en los documentos del corpus (normativa, ICA, episodios críticos)."""
    res = rag.buscar(expandir_siglas(consulta), k=6)
    nota = None
    if not res or max(r["relevancia"] for r in res) < 0.30:
        nota = "Relevancia baja: los fragmentos podrían no responder la pregunta; si es así, dilo."
    return {"instruccion": "Usa ÚNICAMENTE estos fragmentos. Atribuye cada idea al archivo y página del fragmento que la contiene, "
                           "con una frase textual corta entre «». No añadas información externa ni definiciones propias de siglas.",
            "resultados": res, "nota": nota}


HERRAMIENTAS = {"get_pm25": get_pm25, "predict_tomorrow": predict_tomorrow, "search_docs": search_docs}

ESQUEMAS = [
    {"type": "function", "function": {
        "name": "get_pm25",
        "description": "Consulta el PM2.5 promedio diario (µg/m³) y su categoría ICA en una estación del Valle de Aburrá, para una fecha dada o el último día con datos.",
        "parameters": {"type": "object", "properties": {
            "estacion": {"type": "string", "description": "Nombre (o parte del nombre) de la estación, p. ej. 'Belén', 'Sabaneta', 'Bello'."},
            "fecha": {"type": "string", "description": "Fecha AAAA-MM-DD. Omitir para usar el último día con datos."}},
            "required": ["estacion"]}}},
    {"type": "function", "function": {
        "name": "predict_tomorrow",
        "description": "Predice la categoría de calidad del aire (PM2.5) del día siguiente en una estación, con probabilidades, usando el modelo del proyecto.",
        "parameters": {"type": "object", "properties": {
            "estacion": {"type": "string", "description": "Nombre (o parte del nombre) de la estación."},
            "fecha": {"type": "string", "description": "Fecha base AAAA-MM-DD (opcional; por defecto el último día con datos)."}},
            "required": ["estacion"]}}},
    {"type": "function", "function": {
        "name": "search_docs",
        "description": "Busca en los documentos oficiales (ICA del IDEAM, informes de SIATA, resoluciones y planes del POECA) para explicar categorías, normativa, episodios críticos y recomendaciones.",
        "parameters": {"type": "object", "properties": {
            "consulta": {"type": "string", "description": "Frase a buscar, en español. Si la pregunta es qué hacer o qué recomienda la norma, incluye la palabra 'recomendaciones' o 'medidas'."}},
            "required": ["consulta"]}}},
]


def ejecutar(nombre, args):
    """Ejecuta una herramienta; cualquier fallo se devuelve como observación, nunca rompe el loop."""
    f = HERRAMIENTAS.get(nombre)
    if f is None:
        return {"error": f"Herramienta desconocida: {nombre}", "herramientas_validas": list(HERRAMIENTAS)}
    try:
        return f(**args)
    except TypeError as e:
        return {"error": f"Argumentos inválidos para {nombre}: {e}"}
    except Exception as e:
        return {"error": f"Falló {nombre}: {type(e).__name__}: {e}"}


# ------------------------------------------------------------------ LLM
def _cliente():
    if "c" not in _cliente_cache:
        from openai import OpenAI
        clave = os.getenv("LLM_API_KEY")
        if not clave or clave.startswith("pon_tu"):
            raise RuntimeError("Falta LLM_API_KEY en el archivo .env")
        _cliente_cache["c"] = OpenAI(base_url=os.getenv("LLM_BASE_URL"), api_key=clave)
    return _cliente_cache["c"]


def _llamar(mensajes, con_herramientas=True, intentos=3):
    ultimo = None
    for k in range(intentos):
        try:
            kw = {"tools": ESQUEMAS, "tool_choice": "auto"} if con_herramientas else {}
            esfuerzo = os.getenv("LLM_REASONING_EFFORT")      # opcional: solo para modelos de razonamiento (gpt-oss)
            if esfuerzo:
                kw["extra_body"] = {"reasoning_effort": esfuerzo}
            r = _cliente().chat.completions.create(
                model=os.getenv("LLM_MODEL"), messages=mensajes, temperature=0, max_tokens=2500, **kw)
            return r.choices[0].message
        except Exception as e:        # límites de tasa o llamadas a herramienta mal formadas: reintentar
            ultimo = e
            estado = getattr(e, "status_code", None)
            if estado in (401, 403, 404) or type(e).__name__ in ("AuthenticationError", "PermissionDeniedError",
                                                                    "NotFoundError"):
                break                     # llave inválida o modelo inexistente: reintentar no sirve
            if k < intentos - 1:
                time.sleep(5 * (k + 1) ** 2)   # 5 s, 20 s: deja pasar la ventana del límite de tasa
    raise RuntimeError(f"El LLM no respondió: {ultimo}")


def sistema():
    df = nb.cargar_datos()
    desde, hasta = df["fecha"].min().date(), df["fecha"].max().date()
    glosario = "; ".join(f"{k.upper()} = {v}" for k, v in GLOSARIO.items())
    return f"""Eres un asistente sobre calidad del aire (PM2.5) en el Valle de Aburrá, Colombia.
Herramientas: get_pm25 (valores medidos), predict_tomorrow (predicción del día siguiente) y search_docs (documentos oficiales).

Reglas:
1. Para valores medidos usa get_pm25; para pronósticos usa predict_tomorrow; para el significado de categorías, normativa, episodios críticos o recomendaciones usa search_docs. Si la pregunta combina varias cosas, usa varias herramientas.
2. Nunca inventes cifras, fechas ni citas. Si una herramienta devuelve un error, explícaselo al usuario con claridad y, si sirve, ofrece las opciones (por ejemplo, las estaciones disponibles).
3. Los datos disponibles van del {desde} al {hasta}. El "último día con datos" es el {hasta}; los términos "hoy" y "mañana" se refieren a ese día base y no a la fecha real de hoy. Menciona siempre las fechas explícitas.
4. Termina cada respuesta basada en herramientas con una línea "Fuente: ...". Para datos medidos: "Fuente: SIATA (promedio diario calculado por el proyecto)". Para predicciones: "Fuente: modelo Naive Bayes del proyecto (datos SIATA)". Para documentos: el nombre exacto del archivo y la página, por ejemplo "Fuente: ideam_hoja_metodologica_ICA.pdf, p. 5".
5. Con search_docs, basa cada afirmación ÚNICAMENTE en los fragmentos devueltos. No agregues recomendaciones, síntomas, cifras ni medidas que no aparezcan en ellos, aunque parezcan de sentido común o las conozcas. Para cada afirmación incluye entre comillas angulares «...» una frase corta (máximo 15 palabras) copiada TEXTUALMENTE del fragmento que la respalda, seguida de (archivo.pdf, p. N) con el archivo y la página de ese fragmento. Si no puedes copiar una frase que respalde una idea, no la afirmes. Si distintos documentos aportan ideas distintas, atribuye cada idea a su propio archivo y página. Si los fragmentos no contienen lo pedido, dilo con claridad en lugar de rellenar. Si te preguntan qué hacer o qué recomendaciones hay, busca explícitamente recomendaciones o medidas.
5c. Si la primera búsqueda no trae lo pedido, reformula con otras palabras y busca de nuevo (al menos dos consultas distintas) antes de decir que no está. Cuando informes una ausencia, di "en los fragmentos recuperados no aparece"; nunca afirmes que un documento completo no contiene algo.
5d. El ICA es un índice sin unidades; el PM2.5 se mide en µg/m³. No mezcles las dos escalas: nunca escribas µg/m³ junto a un valor del ICA.
5b. No eres médico. Si das recomendaciones de salud, básate solo en lo que dicen los documentos y aclara que es información orientativa, no consejo médico.
6. Si la pregunta no trata de calidad del aire, di amablemente que solo puedes ayudar con ese tema y no uses herramientas.
7. Responde en español, de forma breve y clara.
8. Glosario (úsalo tal cual; nunca inventes el significado de una sigla): {glosario}."""


# ------------------------------------------------------------------ loop agéntico
def responder(pregunta, historial=None, contexto=None):
    """Devuelve (texto_respuesta, trazas). historial: lista de {'role','content'} (solo texto).

    contexto (opcional, lo envía el dashboard): {'estacion': nombre, 'fecha': 'AAAA-MM-DD'}.
    La evaluación (evaluar_agente.py) no lo usa, así que sus resultados no cambian.
    """
    prompt = sistema()
    if contexto:
        prompt += (f"\n9. En el dashboard el usuario tiene seleccionada la estación «{contexto['estacion']}» y la fecha "
                   f"{contexto['fecha']}. Si pregunta por \"hoy\", \"mañana\" o \"esta estación\" sin dar otra, usa esa "
                   f"estación y toma {contexto['fecha']} como día base (pásalo en el argumento fecha), en lugar de la regla 3.")
    mensajes = [{"role": "system", "content": prompt}] + list(historial or []) + [{"role": "user", "content": pregunta}]
    trazas = []
    for _ in range(T_MAX):
        msg = _llamar(mensajes)
        if not msg.tool_calls:
            return (msg.content or "").strip(), trazas
        mensajes.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
                obs = ejecutar(tc.function.name, args)
            except json.JSONDecodeError:
                args, obs = {}, {"error": "Los argumentos no son un JSON válido."}
            trazas.append({"herramienta": tc.function.name, "argumentos": args, "resultado": obs})
            mensajes.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(obs, ensure_ascii=False)})
    mensajes.append({"role": "user", "content": "Responde ahora con la información que ya tienes, sin llamar más herramientas."})
    return (_llamar(mensajes, con_herramientas=False).content or "").strip(), trazas


if __name__ == "__main__":
    pregunta = " ".join(sys.argv[1:]) or "¿Cómo está el aire en Belén en el último día con datos?"
    texto, trazas = responder(pregunta)
    for t in trazas:
        print(f"[herramienta] {t['herramienta']}({t['argumentos']})")
    print("\n" + texto)