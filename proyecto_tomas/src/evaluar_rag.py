"""Evaluación de la recuperación del RAG con consultas de respuesta conocida.

Cada consulta tiene las (archivo, página) donde está la respuesta (revisadas a mano en los PDF).
Se mide hit@1, hit@6 (el agente recibe 6 fragmentos) y MRR, a nivel de página, comparando:
la búsqueda híbrida que usa el agente (FAISS + TF-IDF con RRF), solo embeddings y solo TF-IDF.

Uso:  python src/evaluar_rag.py      Salida: docs/evaluacion_rag.md y docs/evaluacion_rag.json
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import rag  # noqa: E402
from agente import expandir_siglas  # noqa: E402

BASE = rag.BASE
A, I, L, S, T = ("amva_resolucion_2356_2023.pdf", "ideam_hoja_metodologica_ICA.pdf",
                 "laestrella_plan_accion_poeca.pdf", "siata_informe_diario_amva_2026-04-24.pdf",
                 "siata_poeca_plantilla_ica.pdf")
CONSULTAS = [
    ("¿Cuáles son los puntos de corte del ICA para PM2.5?", {(I, 5)}),
    ("¿Qué efectos en la salud tiene la categoría naranja del ICA?", {(I, 4)}),
    ("¿Cómo se calcula el Índice de Calidad del Aire?", {(I, 5), (I, 6)}),
    ("¿En qué meses se declaran los periodos de gestión de episodios de contaminación?", {(A, 5), (L, 14)}),
    ("¿En qué contaminantes se enfoca el POECA?", {(A, 3)}),
    ("¿Qué información entrega el Comité de Comunicaciones Metropolitano?", {(A, 10)}),
    ("Recomendaciones para establecimientos educativos en alerta roja ambiental", {(L, 59)}),
    ("Signos y síntomas respiratorios de alarma en los niños", {(L, 94)}),
    ("Operativos de vigilancia de quemas e incendios durante los episodios", {(L, 73)}),
    ("¿Qué efecto tuvo la medida de pico y placa en la hora pico de contaminantes?", {(L, 18)}),
    ("¿Cuántas estaciones poblacionales presentan ICA verde y amarillo hoy?", {(S, 1)}),
    ("¿Qué se hace si alguna estación presenta ICA rojo o naranja?", {(T, 1)}),
    ("¿Qué es un episodio o evento de contaminación atmosférica?", {(A, 8)}),
    ("¿Quién es la máxima autoridad ambiental en el perímetro urbano del valle?", {(A, 2)}),
]


def rankings(consulta, n=20):
    rag.buscar(consulta, k=1)  # carga caché
    idx, frag, mod = rag._cargar()
    q = mod.encode([consulta], normalize_embeddings=True).astype("float32")
    _, dense = idx.search(q, n)
    sims = (rag._cache["matriz"] @ rag._cache["tfidf"].transform([consulta]).T).toarray().ravel()
    sparse = [int(i) for i in np.argsort(-sims)[:n] if sims[i] > 0]
    hib = [(r["fuente"], r["pagina"]) for r in rag.buscar(consulta, k=n)]
    to_fp = lambda ids: [(frag[i]["fuente"], frag[i]["pagina"]) for i in ids if i >= 0]  # noqa: E731
    return {"híbrido (agente)": hib, "solo embeddings + FAISS": to_fp(dense[0]), "solo TF-IDF": to_fp(sparse)}


def main():
    res = {}
    detalle = []
    for consulta, relevantes in CONSULTAS:
        r = rankings(expandir_siglas(consulta))
        fila = {"consulta": consulta, "relevantes": sorted(f"{f} p.{p}" for f, p in relevantes)}
        for nombre, rk in r.items():
            pos = next((i + 1 for i, fp in enumerate(rk) if fp in relevantes), None)
            res.setdefault(nombre, []).append(pos)
            fila[nombre] = pos
        detalle.append(fila)
    tabla = {}
    for nombre, pos in res.items():
        tabla[nombre] = {"hit@1": float(np.mean([p == 1 for p in pos])),
                         "hit@6": float(np.mean([p is not None and p <= 6 for p in pos])),
                         "mrr": float(np.mean([1 / p if p else 0 for p in pos]))}
    out = {"n_consultas": len(CONSULTAS), "nivel": "página (archivo + página)", "resultados": tabla, "detalle": detalle}
    (BASE / "docs" / "evaluacion_rag.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [f"# Evaluación de la recuperación (RAG)\n", f"{len(CONSULTAS)} consultas con la página correcta "
             "identificada a mano. Acierto = el fragmento recuperado es de esa página.\n",
             "| Recuperador | hit@1 | hit@6 | MRR |", "|---|---|---|---|"]
    lines += [f"| {n} | {v['hit@1']:.2f} | {v['hit@6']:.2f} | {v['mrr']:.2f} |" for n, v in tabla.items()]
    lines += ["", "| Consulta | Página correcta | Posición (híbrido) | Embeddings | TF-IDF |", "|---|---|---|---|---|"]
    for d in detalle:
        f = lambda v: v if v else ">20"  # noqa: E731
        lines.append(f"| {d['consulta']} | {', '.join(d['relevantes'])} | {f(d['híbrido (agente)'])} | "
                     f"{f(d['solo embeddings + FAISS'])} | {f(d['solo TF-IDF'])} |")
    (BASE / "docs" / "evaluacion_rag.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:7]))


if __name__ == "__main__":
    main()
