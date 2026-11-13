"""RAG: extrae texto de los PDF de docs/corpus, los trocea, calcula embeddings y los indexa con FAISS.

Uso:
    python src/rag.py                       # construye el índice y muestra un resumen por documento
    python src/rag.py "¿Qué es el POECA?"   # además hace una búsqueda de prueba
"""
import json
import os
import pathlib
import re
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BASE = pathlib.Path(__file__).resolve().parent.parent
CORPUS = BASE / "docs" / "corpus"
INDEX_DIR = BASE / "data" / "index"
MODELO_EMB = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TAM = 900        # caracteres por fragmento
SOLAPE = 150     # caracteres compartidos entre fragmentos consecutivos
MIN_CHARS = 80   # se descartan fragmentos más cortos

_cache = {}


def trocear(texto, tam=TAM, solape=SOLAPE):
    texto = re.sub(r"\s+", " ", texto).strip()
    trozos, i = [], 0
    while i < len(texto):
        fin = min(i + tam, len(texto))
        if fin < len(texto):                                   # intenta cortar en fin de oración o espacio
            corte = max(texto.rfind(". ", i, fin), texto.rfind(" ", i, fin))
            if corte > i + tam // 2:
                fin = corte + 1
        trozos.append(texto[i:fin].strip())
        if fin >= len(texto):
            break
        i = max(fin - solape, i + 1)
    return [t for t in trozos if len(t) >= MIN_CHARS]


def construir_indice():
    import faiss
    from pypdf import PdfReader
    from sentence_transformers import SentenceTransformer

    fragmentos = []
    for pdf in sorted(CORPUS.glob("*.pdf")):
        lector = PdfReader(str(pdf))
        n_chars = n_frag = 0
        for num, pagina in enumerate(lector.pages, 1):
            try:
                txt = pagina.extract_text() or ""
            except Exception:
                txt = ""
            n_chars += len(txt)
            for t in trocear(txt):
                fragmentos.append({"fuente": pdf.name, "pagina": num, "texto": t})
                n_frag += 1
        aviso = "   <-- POCO TEXTO (¿PDF escaneado o de imágenes?)" if n_chars < 500 else ""
        print(f"{pdf.name}: {len(lector.pages)} págs, {n_chars} caracteres, {n_frag} fragmentos{aviso}")
    if not fragmentos:
        raise RuntimeError(f"No se extrajo texto de ningún PDF en {CORPUS}")

    print(f"\nTotal: {len(fragmentos)} fragmentos. Calculando embeddings (la primera vez descarga el modelo)...")
    modelo = SentenceTransformer(MODELO_EMB)
    emb = modelo.encode([f["texto"] for f in fragmentos], batch_size=32,
                        show_progress_bar=True, normalize_embeddings=True).astype("float32")
    indice = faiss.IndexFlatIP(emb.shape[1])                    # producto interno = coseno (vectores normalizados)
    indice.add(emb)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(indice, str(INDEX_DIR / "faiss.index"))
    json.dump(fragmentos, open(INDEX_DIR / "fragmentos.json", "w"), ensure_ascii=False)
    print("Índice guardado en", INDEX_DIR)


def _cargar():
    if not _cache:
        import faiss
        from sentence_transformers import SentenceTransformer
        if not (INDEX_DIR / "faiss.index").exists():
            construir_indice()
        _cache["indice"] = faiss.read_index(str(INDEX_DIR / "faiss.index"))
        _cache["fragmentos"] = json.load(open(INDEX_DIR / "fragmentos.json"))
        _cache["modelo"] = SentenceTransformer(MODELO_EMB)
        from sklearn.feature_extraction.text import TfidfVectorizer
        _cache["tfidf"] = TfidfVectorizer(strip_accents="unicode", lowercase=True, sublinear_tf=True, ngram_range=(1, 2))
        _cache["matriz"] = _cache["tfidf"].fit_transform([f["texto"] for f in _cache["fragmentos"]])
    return _cache["indice"], _cache["fragmentos"], _cache["modelo"]


def buscar(consulta, k=6, n_candidatos=20, c_rrf=60):
    """Búsqueda híbrida: embeddings (FAISS) + palabras clave (TF-IDF), fusionadas con Reciprocal Rank Fusion."""
    import numpy as np
    indice, fragmentos, modelo = _cargar()
    q = modelo.encode([consulta], normalize_embeddings=True).astype("float32")
    _, pos_emb = indice.search(q, min(n_candidatos, len(fragmentos)))
    sims = (_cache["matriz"] @ _cache["tfidf"].transform([consulta]).T).toarray().ravel()
    pos_lex = [int(i) for i in np.argsort(-sims)[:n_candidatos] if sims[i] > 0]
    puntaje = {}
    for rango, i in enumerate(pos_emb[0]):
        if i >= 0:
            puntaje[int(i)] = puntaje.get(int(i), 0.0) + 1.0 / (c_rrf + rango)
    for rango, i in enumerate(pos_lex):
        puntaje[i] = puntaje.get(i, 0.0) + 1.0 / (c_rrf + rango)
    mejores = sorted(puntaje, key=puntaje.get, reverse=True)[:k]
    return [{"fuente": fragmentos[i]["fuente"], "pagina": fragmentos[i]["pagina"], "texto": fragmentos[i]["texto"],
             "relevancia": round(float(indice.reconstruct(i) @ q[0]), 3)} for i in mejores]


if __name__ == "__main__":
    construir_indice()
    if len(sys.argv) > 1:
        print(f"\nBúsqueda de prueba: {sys.argv[1]}")
        for r in buscar(sys.argv[1]):
            print(f"\n[{r['relevancia']}] {r['fuente']} p.{r['pagina']}\n{r['texto'][:400]}")