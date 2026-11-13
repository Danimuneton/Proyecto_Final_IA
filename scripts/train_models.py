"""Entrena y evalúa los modelos de la Parte 1 (HMM + regresión logística propios).

Salidas:
- reports/metricas_parte1.json        métricas de todos los modelos y líneas base
- reports/metricas_parte1.md          tabla en Markdown (se copia al README/informe)
- reports/backtest_predicciones.csv   predicciones fuera de muestra (rolling origin)
- reports/figures/*.png               figuras para el informe
- models/hmm.json, models/clasificador.json   modelos finales usados por la app

Uso:  python scripts/train_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from aire import config as C  # noqa: E402
from aire.classifier import metrics  # noqa: E402
from aire.modeling import (MODELOS, rolling_origin_evaluation, save_json,  # noqa: E402
                           train_final)

RNG = np.random.default_rng(42)


def bootstrap_diff(P: pd.DataFrame, col_a: str, col_b: str, n: int = 2000) -> dict:
    """IC 95 % de la diferencia de métricas (a - b) remuestreando DÍAS completos.

    Remuestrear por día (no por fila) respeta la correlación entre estaciones del mismo día.
    """
    days = P["fecha"].unique()
    groups = {d: idx for d, idx in P.groupby("fecha").indices.items()}
    y = P["clase_manana"].to_numpy()
    a = P[col_a].to_numpy()
    b = P[col_b].to_numpy()
    out = {"accuracy": [], "macro_f1": [], "balanced_accuracy": []}
    for _ in range(n):
        sample = RNG.choice(days, size=len(days), replace=True)
        idx = np.concatenate([groups[d] for d in sample])
        ma, mb = metrics(y[idx], a[idx]), metrics(y[idx], b[idx])
        for k in out:
            out[k].append(ma[k] - mb[k])
    return {k: {"media": float(np.mean(v)), "ic95": [float(np.percentile(v, 2.5)),
                                                     float(np.percentile(v, 97.5))]}
            for k, v in out.items()}


def bootstrap_proba_diff(P: pd.DataFrame, pref_a: str, pref_b: str, n: int = 2000) -> dict:
    """IC 95 % de la diferencia de log-loss y Brier (a - b; negativo = a es mejor)."""
    y = P["clase_manana"].to_numpy()
    Y = np.eye(3)[y]
    pa = np.clip(P[[f"proba_{pref_a}_{k}" for k in range(3)]].to_numpy(), 1e-12, 1)
    pb = np.clip(P[[f"proba_{pref_b}_{k}" for k in range(3)]].to_numpy(), 1e-12, 1)
    ll = -np.log((pa * Y).sum(1)) + np.log((pb * Y).sum(1))
    br = ((pa - Y) ** 2).sum(1) - ((pb - Y) ** 2).sum(1)
    days = P["fecha"].unique()
    groups = P.groupby("fecha").indices
    res = {"log_loss": [], "brier": []}
    for _ in range(n):
        idx = np.concatenate([groups[d] for d in RNG.choice(days, size=len(days), replace=True)])
        res["log_loss"].append(ll[idx].mean())
        res["brier"].append(br[idx].mean())
    return {k: {"media": float(np.mean(v)), "ic95": [float(np.percentile(v, 2.5)),
                                                     float(np.percentile(v, 97.5))]}
            for k, v in res.items()}


def main() -> None:
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("1) Evaluación temporal con origen móvil (re-entrena por mes)...")
    r = rolling_origin_evaluation()
    P = r["predicciones"]
    P.to_csv(C.REPORTS_DIR / "backtest_predicciones.csv", index=False)

    print("2) Bootstrap por días: modelo final vs persistencia...")
    boot = bootstrap_diff(P, "pred_logreg_completo", "pred_persistencia")
    boot_hmm = bootstrap_diff(P, "pred_logreg_completo", "pred_logreg_sin_hmm")
    boot_prob = bootstrap_proba_diff(P, "logreg_completo", "markov")
    boot_prob_hmm = bootstrap_proba_diff(P, "logreg_completo", "logreg_sin_hmm")

    print("3) Entrenamiento final con todos los datos...")
    final = train_final()

    out = {
        "descripcion": ("Predicción de la clase ICA de PM2.5 del día siguiente por estación. "
                        "Validación con origen móvil mensual (ene-jul 2024), re-entrenando "
                        "con datos anteriores a cada mes."),
        "meses_prueba": r["meses_prueba"],
        "n_predicciones": int(len(P)),
        "distribucion_clases_prueba": np.bincount(P["clase_manana"], minlength=3).tolist(),
        "resultados": r["resultados"],
        "bootstrap_final_vs_persistencia": boot,
        "bootstrap_final_vs_sin_hmm": boot_hmm,
        "bootstrap_probabilistico_final_vs_markov": boot_prob,
        "bootstrap_probabilistico_final_vs_sin_hmm": boot_prob_hmm,
        "hmm_loglik_predictiva": r["hmm_loglik"],
        "verificacion_sklearn": r["verificacion_sklearn"],
        "alpha_por_mes": r["alpha_por_mes"],
        "modelo_final": final,
    }
    save_json(out, C.REPORTS_DIR / "metricas_parte1.json")

    # Tabla Markdown
    lines = ["| Modelo | Exactitud | Exactitud balanceada | F1 macro | Recall Dañina | Log-loss | Brier |",
             "|---|---|---|---|---|---|---|"]
    for name, m in r["resultados"].items():
        ll = f"{m['log_loss']:.3f}" if "log_loss" in m else "—"
        br = f"{m['brier']:.3f}" if "brier" in m else "—"
        lines.append(f"| {name} | {m['accuracy']:.3f} | {m['balanced_accuracy']:.3f} | "
                     f"{m['macro_f1']:.3f} | {m['recall_por_clase'][2]:.3f} | {ll} | {br} |")
    hl = pd.DataFrame(r["hmm_loglik"])
    lines += ["", "Log-verosimilitud predictiva por observación (mayor es mejor):", "",
              "| Mes | HMM (3 estados) | Gaussiana i.i.d. |", "|---|---|---|"]
    for _, row in hl.iterrows():
        lines.append(f"| {row['mes']} | {row['ll_hmm_por_obs']:.3f} | {row['ll_iid_por_obs']:.3f} |")
    lines += ["", "Diferencia (modelo final − persistencia), IC 95 % por bootstrap de días:", ""]
    for k, v in boot.items():
        lines.append(f"- {k}: {v['media']:+.3f} [{v['ic95'][0]:+.3f}, {v['ic95'][1]:+.3f}]")
    lines += ["", "Diferencia (modelo final − mismo modelo sin variables del HMM):", ""]
    for k, v in boot_hmm.items():
        lines.append(f"- {k}: {v['media']:+.3f} [{v['ic95'][0]:+.3f}, {v['ic95'][1]:+.3f}]")
    lines += ["", "Métricas probabilísticas (negativo = el modelo final es mejor), IC 95 %:", ""]
    for k, v in boot_prob.items():
        lines.append(f"- {k}, final − cadena de Markov: {v['media']:+.4f} "
                     f"[{v['ic95'][0]:+.4f}, {v['ic95'][1]:+.4f}]")
    for k, v in boot_prob_hmm.items():
        lines.append(f"- {k}, final − sin HMM: {v['media']:+.4f} "
                     f"[{v['ic95'][0]:+.4f}, {v['ic95'][1]:+.4f}]")
    v = r["verificacion_sklearn"]
    lines += ["", f"Verificación contra scikit-learn (mismo objetivo): acuerdo en argmax = "
              f"{v['acuerdo_argmax']:.3%}, diferencia máxima de probabilidad = {v['max_dif_probabilidad']:.4f}."]
    (C.REPORTS_DIR / "metricas_parte1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Figuras
    names = list(MODELOS.keys())
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(names))
    for i, (k, lab) in enumerate([("accuracy", "Exactitud"), ("balanced_accuracy", "Exact. balanceada"),
                                  ("macro_f1", "F1 macro")]):
        ax.bar(x + (i - 1) * 0.27, [r["resultados"][n][k] for n in names], 0.27, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("Reg. logística propia: ", "RL: ").replace("Línea base: ", "Base: ")
                        .replace(" (modelo final)", "*") for n in names], rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title("Predicción de la clase ICA del día siguiente (ene–jul 2024, fuera de muestra)")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "comparacion_modelos.png", dpi=150)
    plt.close(fig)

    M = np.array(r["resultados"][names[-1]]["matriz_confusion"])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(M, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, M[i, j], ha="center", va="center",
                    color="white" if M[i, j] > M.max() / 2 else "black")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(C.CLASES_CORTAS); ax.set_yticklabels(C.CLASES_CORTAS)
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión (modelo final)", fontsize=10)
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "matriz_confusion.png", dpi=150)
    plt.close(fig)

    print((C.REPORTS_DIR / "metricas_parte1.md").read_text(encoding="utf-8"))
    print("Modelos guardados en", C.MODELS_DIR)


if __name__ == "__main__":
    main()
