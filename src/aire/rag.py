"""RAG: fragmentación del corpus, embeddings multilingües e índice FAISS.

- Corpus: archivos Markdown en docs/corpus/ con encabezado de metadatos (id, título,
  fuente, url). Cada sección "## ..." es un fragmento (se subdivide si es muy larga).
- Embeddings: sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2), normalizados.
- Índice: FAISS IndexFlatIP (similitud coseno exacta).
- Respaldo: si el modelo de embeddings no se puede cargar (p. ej., sin internet la
  primera vez), se usa TF-IDF de scikit-learn para que la app no falle.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import config as C

log = logging.getLogger(__name__)
MAX_CHARS = 1200


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    titulo: str
    seccion: str
    fuente: str
    url: str
    texto: str

    def as_text_for_embedding(self) -> str:
        return f"{self.titulo}. {self.seccion}. {self.texto}"


def _parse_front_matter(text: str) -> tuple[dict, str]:
    meta = {}
    if text.startswith("---"):
        end = text.find("\n---", 3)
        header = text[3:end].strip()
        for line in header.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        text = text[end + 4:]
    return meta, text


def _split_long(text: str) -> list[str]:
    if len(text) <= MAX_CHARS:
        return [text]
    parts, cur = [], ""
    for para in re.split(r"\n\s*\n|\n(?=- )", text):
        if len(cur) + len(para) > MAX_CHARS and cur:
            parts.append(cur.strip())
            cur = ""
        cur += para + "\n"
    if cur.strip():
        parts.append(cur.strip())
    return parts


def load_corpus() -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(C.CORPUS_DIR.glob("*.md")):
        meta, body = _parse_front_matter(path.read_text(encoding="utf-8"))
        doc_id = meta.get("id", path.stem)
        sections = re.split(r"\n## ", "\n" + body)
        for i, sec in enumerate(sections):
            sec = sec.strip()
            if not sec or sec.startswith("# "):
                continue
            heading, _, content = sec.partition("\n")
            for j, piece in enumerate(_split_long(content.strip())):
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}-{i:02d}-{j}", doc_id=doc_id,
                    titulo=meta.get("titulo", path.stem), seccion=heading.strip(),
                    fuente=meta.get("fuente", ""), url=meta.get("url", ""), texto=piece,
                ))
    return chunks


def _corpus_hash(chunks: list[Chunk]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.as_text_for_embedding().encode("utf-8"))
    return h.hexdigest()[:16]


class Retriever:
    """Recuperador semántico con FAISS (o TF-IDF como respaldo)."""

    def __init__(self, force_rebuild: bool = False, backend: str = "auto"):
        self.chunks = load_corpus()
        self.backend = None
        self._model = None
        self._index = None
        self._tfidf = None
        self._tfidf_matrix = None
        if backend in ("auto", "embeddings"):
            try:
                self._init_embeddings(force_rebuild)
                self.backend = "embeddings+faiss"
            except Exception as exc:  # noqa: BLE001
                if backend == "embeddings":
                    raise
                log.warning("Embeddings no disponibles (%s). Se usa TF-IDF.", exc)
        self._init_tfidf()
        if self.backend is None:
            self.backend = "tfidf"
        elif backend == "auto":
            # Búsqueda híbrida: fusión por rango recíproco (RRF) de FAISS y TF-IDF
            self.backend = "hibrido"

    # ------------------------------------------------------------- embeddings
    def _init_embeddings(self, force_rebuild: bool) -> None:
        import faiss
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(C.EMBEDDING_MODEL, device="cpu")
        idx_dir = C.RAG_INDEX_DIR
        idx_dir.mkdir(parents=True, exist_ok=True)
        meta_path = idx_dir / "meta.json"
        index_path = idx_dir / "faiss.index"
        h = _corpus_hash(self.chunks)
        if not force_rebuild and meta_path.exists() and index_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("hash") == h and meta.get("modelo") == C.EMBEDDING_MODEL:
                self._index = faiss.read_index(str(index_path))
                return
        emb = self._embed([c.as_text_for_embedding() for c in self.chunks])
        self._index = faiss.IndexFlatIP(emb.shape[1])
        self._index.add(emb)
        faiss.write_index(self._index, str(index_path))
        meta_path.write_text(json.dumps({
            "hash": h, "modelo": C.EMBEDDING_MODEL, "n_fragmentos": len(self.chunks),
            "fragmentos": [c.__dict__ for c in self.chunks]}, ensure_ascii=False, indent=1),
            encoding="utf-8")

    def _embed(self, texts: list[str]) -> np.ndarray:
        e = self._model.encode(texts, batch_size=32, normalize_embeddings=True,
                               show_progress_bar=False)
        return np.asarray(e, dtype="float32")

    # ------------------------------------------------------------------ tfidf
    def _init_tfidf(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._tfidf = TfidfVectorizer(strip_accents="unicode", lowercase=True,
                                      ngram_range=(1, 2), sublinear_tf=True)
        self._tfidf_matrix = self._tfidf.fit_transform(
            [c.as_text_for_embedding() for c in self.chunks])

    # ----------------------------------------------------------------- search
    def _dense_rank(self, query: str, n: int) -> tuple[list[int], list[float]]:
        scores, ids = self._index.search(self._embed([query]), n)
        return ids[0].tolist(), scores[0].tolist()

    def _sparse_rank(self, query: str, n: int) -> tuple[list[int], list[float]]:
        sims = (self._tfidf_matrix @ self._tfidf.transform([query]).T).toarray().ravel()
        top = np.argsort(-sims)[:n]
        return top.tolist(), sims[top].tolist()

    def search(self, query: str, k: int = 4) -> list[dict]:
        k = max(1, min(int(k), 8))
        if self.backend == "embeddings+faiss":
            ids, sc = self._dense_rank(query, k)
            pairs = zip(ids, sc)
        elif self.backend == "tfidf":
            ids, sc = self._sparse_rank(query, k)
            pairs = zip(ids, sc)
        else:
            n = len(self.chunks)
            fused: dict[int, float] = {}
            for ranking in (self._dense_rank(query, n)[0], self._sparse_rank(query, n)[0]):
                for r, i in enumerate(ranking):
                    if i >= 0:
                        fused[i] = fused.get(i, 0.0) + 1.0 / (60 + r + 1)
            top = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
            pairs = [(i, s * 60) for i, s in top]  # escala ~[0, 2] para legibilidad
        out = []
        for i, s in pairs:
            if i < 0:
                continue
            c = self.chunks[i]
            out.append({"fragmento": c.chunk_id, "doc_id": c.doc_id, "titulo": c.titulo,
                        "seccion": c.seccion, "fuente": c.fuente, "url": c.url,
                        "score": round(float(s), 4), "texto": c.texto})
        return out


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever()
