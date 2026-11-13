"""Evalúa el agente con 15 preguntas fijas.

Uso:
    python src/evaluar_agente.py            # llama al agente, guarda las trazas y calcula las métricas
    python src/evaluar_agente.py --rescore  # recalcula las métricas desde las trazas guardadas (sin llamar al LLM)
Salidas: docs/evaluacion_agente.csv, docs/evaluacion_agente.md, docs/evaluacion_trazas.json
"""
import csv
import json
import pathlib
import re
import sys
import time
import unicodedata

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

BASE = pathlib.Path(__file__).resolve().parent.parent
OUT = BASE / "docs"
G, P, S = "get_pm25", "search_docs", "predict_tomorrow"

# (pregunta, herramientas esperadas)
PREGUNTAS = [
    ("¿Cuál fue el PM2.5 promedio en la estación de Belén el 15 de enero de 2024?", {G}),
    ("¿Qué categoría de calidad del aire tuvo Sabaneta el 2 de marzo de 2024?", {G}),
    ("¿Cómo estuvo el aire en Envigado en el último día con datos?", {G}),
    ("¿Cómo estará el aire mañana en Itagüí?", {S}),
    ("Predice la calidad del aire de mañana en Bello.", {S}),
    ("¿Qué significa un ICA naranja y qué debo hacer?", {P}),
    ("¿Cuáles son los niveles de contingencia atmosférica del Valle de Aburrá?", {P}),
    ("¿Qué es el POECA y en qué épocas del año se aplica?", {P}),
    ("¿Cuál es el rango de PM2.5 de la categoría Buena según el ICA?", {P}),
    ("¿Quién declara un episodio de contaminación y cuál es el procedimiento?", {P}),
    ("¿Cómo está el aire en Caldas en el último día con datos y qué debería hacer una persona con asma?", {G, P}),
    ("¿Cómo estará mañana el aire en Copacabana y qué significa esa categoría para los niños?", {S, P}),
    ("¿Cuál es la capital de Francia?", set()),
    ("¿Cuánto fue el PM2.5 en la estación Marte el 1 de enero de 2024?", {G}),
    ("¿Cuál fue el PM2.5 en El Poblado el 10 de noviembre de 2026?", {G}),
]


# ------------------------------------------------------------------ normalización y métricas
def _guiones(t):
    return re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212]", "-", t)


def _norm(t):
    """Minúsculas, sin tildes, guiones unificados y sin espacios (robusto a los cortes de palabra del PDF)."""
    t = _guiones(t).lower()
    t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", "", t)


def _recuperados(trazas):
    return [r for t in trazas if t["herramienta"] == P for r in (t["resultado"].get("resultados") or [])]


def cita_verificable(respuesta, trazas):
    """'sí': todos los (archivo, página) citados estaban entre los fragmentos recuperados;
    'no': se cita algo que no se recuperó; 'sin citas': la respuesta no cita ningún documento."""
    recup = {(r["fuente"], r["pagina"]) for r in _recuperados(trazas)}
    citadas = re.findall(r"([\w\-.]+\.pdf)\s*,?\s*(?:p\.?|pág\.?|página)\s*(\d+)", _guiones(respuesta), flags=re.I)
    if not citadas:
        return "sin citas"
    return "sí" if all((f, int(p)) in recup for f, p in citadas) else "no"


def cita_textual(respuesta, trazas):
    """¿Las frases entre «» o “” están en los fragmentos recuperados? Los '...' dentro de una frase la dividen en partes."""
    texto = _norm(" ".join(r["texto"] for r in _recuperados(trazas)))
    citas = re.findall(r"«([^»]+)»|“([^”]+)”", respuesta)
    citas = [a or b for a, b in citas]
    if not citas:
        return "sin comillas"
    ok = 0
    for c in citas:
        partes = [_norm(x) for x in re.split(r"\.\.\.|…", c)]
        partes = [x for x in partes if len(x) >= 8]
        ok += bool(partes) and all(x in texto for x in partes)
    return "sí" if ok == len(citas) else f"parcial ({ok}/{len(citas)})"


def puntuar(pregunta, esperadas, respuesta, trazas):
    llamadas = {t["herramienta"] for t in trazas}
    correcta = (llamadas >= esperadas) if esperadas else (not llamadas)
    low = _guiones(respuesta).lower()
    if not esperadas:
        cita = "n/a"
    elif P in esperadas:
        cita = "sí" if ".pdf" in low else "no"
    elif trazas and all("error" in t["resultado"] for t in trazas):
        cita = "n/a"                      # la herramienta devolvió un error: no había nada que citar
    else:
        cita = "sí" if ("fuente" in low and ("siata" in low or "naive bayes" in low)) else "no"
    con_doc = P in esperadas
    return {"pregunta": pregunta, "esperadas": ", ".join(sorted(esperadas)) or "ninguna",
            "llamadas": ", ".join(sorted(llamadas)) or "ninguna",
            "herramienta_correcta": "sí" if correcta else "no", "cita_fuente": cita,
            "cita_verificable": cita_verificable(respuesta, trazas) if con_doc else "n/a",
            "cita_textual": cita_textual(respuesta, trazas) if con_doc else "n/a",
            "respuesta": respuesta.replace("\n", " ")}


# ------------------------------------------------------------------ ejecución y reporte
def ejecutar_agente():
    import agente
    registros = []
    for i, (pregunta, esperadas) in enumerate(PREGUNTAS, 1):
        print(f"[{i}/{len(PREGUNTAS)}] {pregunta}")
        try:
            respuesta, trazas = agente.responder(pregunta)
        except Exception as e:
            respuesta, trazas = f"ERROR: {e}", []
        registros.append({"pregunta": pregunta, "respuesta": respuesta, "trazas": trazas})
        time.sleep(8)                     # evita límites de tasa del endpoint gratuito
    OUT.mkdir(exist_ok=True)
    json.dump(registros, open(OUT / "evaluacion_trazas.json", "w"), ensure_ascii=False, indent=1)
    return registros


def escribir(filas):
    OUT.mkdir(exist_ok=True)
    for i, f in enumerate(filas, 1):
        f["n"] = i
    cols = ["n", "pregunta", "esperadas", "llamadas", "herramienta_correcta", "cita_fuente",
            "cita_verificable", "cita_textual", "respuesta"]
    with open(OUT / "evaluacion_agente.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(filas)
    ok = sum(f["herramienta_correcta"] == "sí" for f in filas)
    cf = [f for f in filas if f["cita_fuente"] != "n/a"]
    docs = [f for f in filas if f["cita_verificable"] != "n/a"]
    cuenta = lambda lst, k, v: sum(f[k].startswith(v) for f in lst)
    resumen = (f"Herramienta correcta: {ok}/{len(filas)} | Cita de fuente: {cuenta(cf, 'cita_fuente', 'sí')}/{len(cf)} | "
               f"Cita verificable: sí {cuenta(docs, 'cita_verificable', 'sí')}, no {cuenta(docs, 'cita_verificable', 'no')}, "
               f"sin citas {cuenta(docs, 'cita_verificable', 'sin')} (de {len(docs)}) | "
               f"Frases textuales: sí {cuenta(docs, 'cita_textual', 'sí')}, parcial {cuenta(docs, 'cita_textual', 'parcial')}, "
               f"sin comillas {cuenta(docs, 'cita_textual', 'sin')} (de {len(docs)})")
    with open(OUT / "evaluacion_agente.md", "w", encoding="utf-8") as fh:
        fh.write("# Evaluación del agente\n\n")
        fh.write(f"- Herramienta correcta: **{ok}/{len(filas)}**\n")
        fh.write(f"- Cita de fuente (datos y documentos; excluye la pregunta fuera de tema y los errores de herramienta): **{cuenta(cf, 'cita_fuente', 'sí')}/{len(cf)}**\n")
        fh.write(f"- Cita verificable (archivo y página citados estaban entre los fragmentos recuperados), preguntas sobre documentos: "
                 f"**sí {cuenta(docs, 'cita_verificable', 'sí')}**, no {cuenta(docs, 'cita_verificable', 'no')}, sin citas {cuenta(docs, 'cita_verificable', 'sin')} (de {len(docs)})\n")
        fh.write(f"- Frases entre comillas presentes en los fragmentos recuperados: **sí {cuenta(docs, 'cita_textual', 'sí')}**, "
                 f"parcial {cuenta(docs, 'cita_textual', 'parcial')}, sin comillas {cuenta(docs, 'cita_textual', 'sin')} (de {len(docs)})\n\n")
        fh.write("| # | Pregunta | Esperadas | Llamadas | ¿Correcta? | ¿Citó fuente? | ¿Cita verificable? | ¿Frases textuales? |\n|---|---|---|---|---|---|---|---|\n")
        for f in filas:
            fh.write(f"| {f['n']} | {f['pregunta']} | {f['esperadas']} | {f['llamadas']} | {f['herramienta_correcta']} | {f['cita_fuente']} | {f['cita_verificable']} | {f['cita_textual']} |\n")
        fh.write("\n## Respuestas completas (revisar a mano que no inventen datos)\n\n")
        for f in filas:
            fh.write(f"**{f['n']}. {f['pregunta']}**\n\n{f['respuesta']}\n\n")
    print("\n" + resumen)
    print("Escrito docs/evaluacion_agente.csv, docs/evaluacion_agente.md y docs/evaluacion_trazas.json")


def main():
    if "--rescore" in sys.argv:
        registros = json.load(open(OUT / "evaluacion_trazas.json"))
    else:
        registros = ejecutar_agente()
    esperadas = dict(PREGUNTAS)
    escribir([puntuar(r["pregunta"], esperadas[r["pregunta"]], r["respuesta"], r["trazas"]) for r in registros])


if __name__ == "__main__":
    main()