"""Evaluación automática del agente con preguntas de prueba fijas.

Para cada pregunta de evaluacion/preguntas_agente.json registra:
- herramientas llamadas y si incluyen las esperadas (herramienta_correcta),
- si la respuesta cita fuentes con etiquetas [DOC: sección], [SIATA] o [Modelo del proyecto],
- si las citas a documentos corresponden a fragmentos realmente recuperados (cita_valida),
- si aparece el valor numérico esperado (cuando aplica), errores, iteraciones y tiempo.

Uso (PowerShell):
    python scripts/evaluate_agent.py                 # usa el LLM configurado en .env
    $env:LLM_PROVIDER="ollama"; $env:LLM_MODEL="qwen2.5:7b"; python scripts/evaluate_agent.py

Salidas: evaluacion/resultados_agente_<modelo>.csv / .md y evaluacion/resultados_agente.md
(copia de la última corrida).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from aire import config as C  # noqa: E402
from aire.agent import CITE_RE, AirQualityAgent, llm_config, retrieved_tags  # noqa: E402
from aire.rag import get_retriever  # noqa: E402


def evaluate_answer(q: dict, res) -> dict:
    called = res.tools_called
    expected = q["herramientas_esperadas"]
    if expected:
        tool_ok = all(t in called for t in expected)
    else:
        tool_ok = len(called) == 0
    cites = [m.group(1).upper() for m in CITE_RE.finditer(res.answer)]
    cited = len(cites) > 0
    retrieved = retrieved_tags(res.steps)
    invalid = [c for c in cites if c not in retrieved]
    valor = q.get("valor_esperado")
    valor_ok = None
    if valor:
        valor_ok = (valor in res.answer) or (valor.replace(".", ",") in res.answer)
    return {
        "id": q["id"], "tipo": q["tipo"], "pregunta": q["pregunta"],
        "herramientas_esperadas": ", ".join(expected) or "(ninguna)",
        "herramientas_llamadas": ", ".join(called) or "(ninguna)",
        "herramienta_correcta": tool_ok,
        "requiere_cita": q["requiere_cita"],
        "cito_fuente": cited,
        "cita_valida": cited and not invalid,
        "citas": ", ".join(sorted(set(cites))),
        "citas_no_recuperadas": ", ".join(sorted(set(invalid))),
        "valor_esperado": valor or "",
        "valor_correcto": valor_ok,
        "errores_herramienta": sum(1 for s in res.steps if not s.ok),
        "iteraciones": res.iterations,
        "verificaciones": res.verificaciones,
        "segundos": res.seconds,
        "error_api": res.error or "",
        "respuesta": res.answer,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo", nargs="*", help="IDs de preguntas a correr (p. ej. P01 P05)")
    ap.add_argument("--pausa", type=float, default=2.0, help="segundos entre preguntas (límites de uso)")
    ap.add_argument("--reporte", action="store_true",
                    help="solo regenerar el reporte .md desde el CSV existente (sin llamar al LLM)")
    args = ap.parse_args()

    cfg = llm_config()
    print(f"LLM: {cfg['provider']} / {cfg['model']} @ {cfg['base_url']}")
    preguntas = json.loads((C.EVAL_DIR / "preguntas_agente.json").read_text(encoding="utf-8"))
    if args.reporte:
        args.solo = ["__ninguna__"]
        preguntas = []
    else:
        get_retriever()
        agent = AirQualityAgent()
    if args.solo:
        preguntas = [q for q in preguntas if q["id"] in args.solo]

    rows = []
    for q in preguntas:
        res = agent.run(q["pregunta"], fecha=q["fecha"], estacion=q["estacion"])
        row = evaluate_answer(q, res)
        rows.append(row)
        print(f"{q['id']} herramienta_ok={row['herramienta_correcta']} cita={row['cito_fuente']} "
              f"llamadas=[{row['herramientas_llamadas']}] {row['segundos']}s {row['error_api'][:80]}")
        time.sleep(args.pausa)

    df = pd.DataFrame(rows)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", cfg["model"])
    csv_path = C.EVAL_DIR / f"resultados_agente_{safe}.csv"
    if args.solo and csv_path.exists():
        # Corrida parcial: se actualizan solo las preguntas indicadas y se conservan las demás
        prev = pd.read_csv(csv_path, keep_default_na=False)
        if len(df):
            prev = prev[~prev["id"].isin(df["id"])]
        df = pd.concat([prev, df.astype(str)], ignore_index=True).sort_values("id")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    # Se relee del disco para que el resumen use siempre los mismos tipos de dato
    df = pd.read_csv(csv_path, keep_default_na=False, dtype=str)
    for col in ["herramienta_correcta", "requiere_cita", "cito_fuente", "cita_valida"]:
        df[col] = df[col] == "True"
    df["valor_correcto"] = df["valor_correcto"].map(lambda v: None if v in ("", "None") else v == "True")
    df["segundos"] = df["segundos"].astype(float)
    df["errores_herramienta"] = df["errores_herramienta"].astype(int)
    rows = df.to_dict("records")

    req = df[df["requiere_cita"]]
    resumen = {
        "modelo": cfg["model"], "proveedor": cfg["provider"], "n_preguntas": len(df),
        "herramienta_correcta": f"{df['herramienta_correcta'].sum()}/{len(df)}",
        "cito_fuente (preguntas que lo requieren)": f"{req['cito_fuente'].sum()}/{len(req)}",
        "citas_validas (preguntas que lo requieren)": f"{req['cita_valida'].sum()}/{len(req)}",
        "valor_numerico_correcto": f"{int(df['valor_correcto'].fillna(False).astype(bool).sum())}/"
                                   f"{int(df['valor_correcto'].notna().sum())}",
        "errores_api": int((df["error_api"] != "").sum()),
        "tiempo_medio_s": round(float(df["segundos"].mean()), 1),
    }
    si = lambda b: "✅" if b is True else ("❌" if b is False else "—")  # noqa: E731
    lines = [f"# Evaluación del agente · {cfg['provider']} / `{cfg['model']}`", "",
             f"Fecha de ejecución: {time.strftime('%Y-%m-%d %H:%M')} · T_max = {C.T_MAX}", "",
             "## Resumen", ""] + [f"- **{k}**: {v}" for k, v in resumen.items()] + [
             "", "## Detalle", "",
             "| ID | Tipo | Pregunta | Esperadas | Llamadas | ¿Herramienta correcta? | ¿Citó fuente? | ¿Citas válidas? | Valor | Errores herr. | Verif. | s |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['id']} | {r['tipo']} | {r['pregunta']} | {r['herramientas_esperadas']} | "
                     f"{r['herramientas_llamadas']} | {si(r['herramienta_correcta'])} | "
                     f"{si(r['cito_fuente']) if r['requiere_cita'] else si(r['cito_fuente']) + ' (no requerida)'} | "
                     f"{si(r['cita_valida']) if r['cito_fuente'] else '—'} | {si(r['valor_correcto'])} | "
                     f"{r['errores_herramienta']} | {r['verificaciones']} | {r['segundos']} |")
    lines += ["", "Criterios: *herramienta correcta* = llamó todas las esperadas (o ninguna si la pregunta "
              "está fuera de tema); *citó fuente* = la respuesta contiene etiquetas [DOC: sección], [SIATA] o "
              "[Modelo del proyecto]; *citas válidas* = todas las etiquetas citadas corresponden a fuentes "
              "devueltas por las herramientas en esa conversación; *Verif.* = veces que el verificador de "
              "citas del agente pidió corregir la respuesta."]
    lines += ["", "## Respuestas completas", ""]
    for r in rows:
        lines += [f"### {r['id']} · {r['pregunta']}", "",
                  f"*Herramientas:* {r['herramientas_llamadas']} · *Citas:* {r['citas'] or 'ninguna'}"
                  + (f" · *Error API:* {r['error_api']}" if r['error_api'] else ""), "",
                  r["respuesta"].replace("\n", "\n> ").join(["> ", ""]), ""]
    md = "\n".join(lines) + "\n"
    (C.EVAL_DIR / f"resultados_agente_{safe}.md").write_text(md, encoding="utf-8")
    (C.EVAL_DIR / "resultados_agente.md").write_text(md, encoding="utf-8")
    print(json.dumps(resumen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
