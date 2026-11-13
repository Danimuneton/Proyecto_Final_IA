"""Genera reports/figures/arquitectura.png (diagrama de arquitectura del README)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures" / "arquitectura.png"

# nombre: (x, y, ancho, alto, texto, color)
CAJAS = {
    "siata": (0.2, 6.6, 2.6, 1.0, "SIATA\nPM2.5 horario (JSON abierto)", "#dbe9f6"),
    "meteo": (0.2, 5.1, 2.6, 1.0, "Open-Meteo / ERA5\nmeteorología diaria", "#dbe9f6"),
    "data": (3.6, 5.8, 2.6, 1.1, "data.py\nlimpieza y promedios 24 h", "#eeeeee"),
    "hmm": (7.0, 6.6, 3.0, 1.1, "PARTE 1 · HMM propio\nBaum-Welch, forward, Viterbi\nestado: Normal / Alerta / Crítico", "#fde2c8"),
    "rl": (7.0, 4.9, 3.0, 1.1, "PARTE 1 · Regresión logística\npropia: clase ICA de mañana", "#fde2c8"),
    "corpus": (0.2, 2.4, 2.6, 1.2, "docs/corpus\nRes. 2254, POECA, salud,\nOMS, SIATA, meteorología", "#dbe9f6"),
    "rag": (3.6, 2.4, 2.6, 1.2, "RAG híbrido\nMiniLM multilingüe\n+ FAISS + TF-IDF", "#d9f2d9"),
    "tools": (7.0, 2.4, 3.0, 1.4, "Herramientas\nget_pm25 · predict_tomorrow\nget_episode_state\nsearch_docs", "#d9f2d9"),
    "agent": (11.4, 2.4, 2.8, 1.4, "PARTE 2 · Agente LLM\nloop T_max = 6\nverificador de citas", "#d9f2d9"),
    "ui": (11.4, 5.6, 2.8, 1.5, "Dashboard Streamlit\nmapa · series · predicción\nevaluación · chat", "#e6dcf5"),
}
FLECHAS = [
    ("siata", "data", ""), ("meteo", "data", ""), ("data", "hmm", ""), ("data", "rl", ""),
    ("hmm", "rl", "P(estado de hoy)"), ("corpus", "rag", ""), ("rag", "tools", ""),
    ("rl", "tools", "HMM + regresión logística"), ("tools", "agent", "llamada /\nobservación"),
    ("agent", "ui", "chat"), ("rl", "ui", ""), ("hmm", "ui", ""),
]


def centro(n):
    x, y, w, h, *_ = CAJAS[n]
    return x + w / 2, y + h / 2


def borde(n, hacia):
    """Punto del borde de la caja n en dirección a la caja 'hacia'."""
    x, y, w, h, *_ = CAJAS[n]
    cx, cy = centro(n)
    tx, ty = centro(hacia)
    dx, dy = tx - cx, ty - cy
    if dx == 0 and dy == 0:
        return cx, cy
    sx = (w / 2) / abs(dx) if dx else float("inf")
    sy = (h / 2) / abs(dy) if dy else float("inf")
    s = min(sx, sy)
    return cx + dx * s, cy + dy * s


def main() -> None:
    fig, ax = plt.subplots(figsize=(13, 6.4))
    ax.set_xlim(0, 14.4)
    ax.set_ylim(1.9, 8.0)
    ax.axis("off")
    for n, (x, y, w, h, txt, col) in CAJAS.items():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                                    fc=col, ec="#333333", lw=1.2))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9.2)
    for a, b, lab in FLECHAS:
        p, q = borde(a, b), borde(b, a)
        doble = lab.startswith("llamada")
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="<|-|>" if doble else "-|>", mutation_scale=13,
                                     lw=1.2, color="#444444"))
        if lab:
            mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
            centrado = doble
            ax.text(mx if centrado else mx + 0.1, my + (0.12 if not centrado else 0.18), lab, fontsize=8,
                    color="#8a3b00", ha="center" if centrado else "left", va="bottom")
    ax.set_title("Arquitectura: datos reales → modelos de la Parte 1 → herramientas → agente LLM con RAG → dashboard",
                 fontsize=11)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, facecolor="white")
    print("Guardado", OUT)


if __name__ == "__main__":
    sys.exit(main())
