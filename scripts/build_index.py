"""Construye (o reconstruye) el índice FAISS del corpus RAG y evalúa la recuperación.

Uso:  python scripts/build_index.py
La evaluación usa preguntas con el documento esperado (evaluacion/rag_consultas.json)
y reporta hit@k y MRR, comparando embeddings+FAISS contra TF-IDF.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aire import config as C  # noqa: E402
from aire.rag import Retriever  # noqa: E402


def evaluate(ret: Retriever, queries: list[dict], k: int = 4) -> dict:
    hits1 = hitk = rr = 0.0
    detalle = []
    for q in queries:
        res = ret.search(q["consulta"], k=k)
        docs = [r["doc_id"] for r in res]
        rank = docs.index(q["doc_esperado"]) + 1 if q["doc_esperado"] in docs else None
        hits1 += rank == 1
        hitk += rank is not None
        rr += 1 / rank if rank else 0
        detalle.append({"consulta": q["consulta"], "esperado": q["doc_esperado"],
                        "recuperados": docs, "rank": rank})
    n = len(queries)
    return {"backend": ret.backend, "n": n, "hit@1": hits1 / n, f"hit@{k}": hitk / n,
            "mrr": rr / n, "detalle": detalle}


def main() -> None:
    ret = Retriever(force_rebuild=True)
    print(f"Índice listo: {len(ret.chunks)} fragmentos, backend = {ret.backend}")
    qpath = C.EVAL_DIR / "rag_consultas.json"
    if not qpath.exists():
        return
    queries = json.loads(qpath.read_text(encoding="utf-8"))
    res = {"hibrido": evaluate(ret, queries)}
    res["embeddings"] = evaluate(Retriever(backend="embeddings"), queries)
    res["tfidf"] = evaluate(Retriever(backend="tfidf"), queries)
    (C.REPORTS_DIR / "evaluacion_rag.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["| Recuperador | hit@1 | hit@4 | MRR |", "|---|---|---|---|"]
    for name, r in res.items():
        lines.append(f"| {r['backend']} | {r['hit@1']:.2f} | {r['hit@4']:.2f} | {r['mrr']:.2f} |")
    (C.REPORTS_DIR / "evaluacion_rag.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    for d in res["hibrido"]["detalle"]:
        if d["rank"] != 1:
            print("  [no top-1]", d["consulta"], "->", d["recuperados"])


if __name__ == "__main__":
    main()
