"""Evaluación complementaria de la Parte 1 (no modifica el modelo ni models/metricas.json).

1. Verificación de corrección: el Naive Bayes propio se compara con sklearn.naive_bayes.CategoricalNB
   configurado con el mismo suavizado y las mismas probabilidades a priori.
2. Validación con origen móvil mensual (enero a julio de 2024): para cada mes se entrena SOLO con
   días anteriores y se predice cada día del mes. A diferencia de la partición única
   (prueba desde 2024-04-30, sin días "Dañina+"), aquí sí entran los meses de la temporada crítica.
3. Intervalo de confianza del 95 % de la diferencia Naive Bayes - persistencia, por bootstrap de días.

Uso:  python src/evaluacion_temporal.py
Salidas: models/metricas_origen_movil.json, docs/figuras/*.png
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import modelo as nb  # noqa: E402

BASE = nb.BASE
SALIDA = BASE / "models" / "metricas_origen_movil.json"
FIG = BASE / "docs" / "figuras"
MESES = [f"2024-{m:02d}" for m in range(1, 8)]
RNG = np.random.default_rng(42)


def verificar_con_sklearn(t):
    from sklearn.naive_bayes import CategoricalNB
    train, test, _ = nb.split_temporal(t)
    cols = nb.CONJUNTOS["A_basico"]
    propio = nb.entrenar(train, cols)
    sk = CategoricalNB(alpha=1.0, class_prior=np.exp(propio.log_prior),
                       min_categories=[nb.COLUMNAS[c] for c in cols])
    sk.fit(train[cols].values.astype(int), train["y"].values.astype(int))
    p1 = propio.predict_proba(test[cols].values)
    p2 = sk.predict_proba(test[cols].values.astype(int))
    return {"acuerdo_prediccion": float((p1.argmax(1) == p2.argmax(1)).mean()),
            "max_diferencia_probabilidad": float(np.abs(p1 - p2).max()), "n": int(len(test))}


def recall_danina(y, yhat):
    m = y == 2
    return float((yhat[m] == 2).mean()) if m.any() else None


def origen_movil(t):
    filas = []
    d1 = pd.Timedelta(days=1)
    for mes in MESES:
        ini = pd.Timestamp(mes + "-01")
        fin = ini + pd.offsets.MonthBegin(1)
        # 'fecha' es el día base; el objetivo es el día siguiente
        tr = t[t["fecha"] < ini - d1]
        te = t[(t["fecha"] >= ini - d1) & (t["fecha"] < fin - d1)].copy()
        te["mes"] = mes
        for nombre, cols in nb.CONJUNTOS.items():
            te[f"pred_{nombre}"] = nb.entrenar(tr, cols).predict(te[cols].values)
        te["pred_persistencia"] = te["categoria_hoy"].astype(int)
        te["pred_mayoritaria"] = int(np.bincount(tr["y"].astype(int)).argmax())
        filas.append(te)
    return pd.concat(filas, ignore_index=True)


def bootstrap(P, col_a, col_b, n=2000):
    """IC 95 % de (métrica a - métrica b) remuestreando días completos."""
    grupos = P.groupby("fecha").indices
    dias = list(grupos)
    y = P["y"].astype(int).values
    a, b = P[col_a].astype(int).values, P[col_b].astype(int).values
    acc, f1 = [], []
    for _ in range(n):
        idx = np.concatenate([grupos[d] for d in RNG.choice(len(dias), len(dias))
                              for d in [dias[d]]])
        ma, mb = nb.metricas(y[idx], a[idx]), nb.metricas(y[idx], b[idx])
        acc.append(ma["accuracy"] - mb["accuracy"])
        f1.append(ma["f1_macro"] - mb["f1_macro"])
    ic = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    return {"accuracy": {"media": float(np.mean(acc)), "ic95": ic(acc)},
            "f1_macro": {"media": float(np.mean(f1)), "ic95": ic(f1)}}


def figuras(P, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    # 1) exactitud por mes
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    for col, lab, est in [("pred_A_basico", "Naive Bayes (A_basico)", "-o"),
                          ("pred_persistencia", "Persistencia (mañana = hoy)", "--s"),
                          ("pred_mayoritaria", "Clase mayoritaria", ":^")]:
        acc = P.groupby("mes").apply(lambda g: (g[col] == g["y"]).mean(), include_groups=False)
        ax.plot(acc.index, acc.values, est, label=lab)
    dan = P.groupby("mes")["y"].apply(lambda s: int((s == 2).sum()))
    for i, (m, v) in enumerate(dan.items()):
        ax.annotate(f"{v} D+", (i, 0.42), ha="center", fontsize=7, color="#c0392b")
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("Exactitud")
    ax.set_title("Origen móvil mensual (D+ = días Dañina+ en el mes)", fontsize=10)
    ax.legend(fontsize=7, loc="upper right", ncol=3)
    fig.tight_layout()
    fig.savefig(FIG / "origen_movil_por_mes.png", dpi=150)
    plt.close(fig)
    # 2) matriz de confusión
    M = np.array(res["modelos"]["Naive Bayes A_basico"]["matriz_confusion"])
    fig, ax = plt.subplots(figsize=(3.6, 3.1))
    ax.imshow(M, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, M[i, j], ha="center", va="center", color="white" if M[i, j] > M.max() / 2 else "black")
    et = ["Buena", "Aceptable", "Dañina+"]
    ax.set_xticks(range(3), et, fontsize=8)
    ax.set_yticks(range(3), et, fontsize=8)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Naive Bayes, ene–jul 2024", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "matriz_confusion_origen_movil.png", dpi=150)
    plt.close(fig)


def main():
    t = nb.construir_dataset(nb.cargar_datos())
    verif = verificar_con_sklearn(t)
    print("Verificación contra sklearn CategoricalNB:", verif)

    P = origen_movil(t)
    y = P["y"].astype(int).values
    modelos = {f"Naive Bayes {n}": f"pred_{n}" for n in nb.CONJUNTOS}
    modelos["Línea base: persistencia (mañana = hoy)"] = "pred_persistencia"
    modelos["Línea base: clase mayoritaria"] = "pred_mayoritaria"
    res = {"descripcion": "Origen móvil mensual ene-jul 2024: se re-entrena con días anteriores a cada mes.",
           "n_predicciones": int(len(P)),
           "distribucion_clases": np.bincount(y, minlength=3).tolist(),
           "verificacion_sklearn": verif, "modelos": {}}
    print(f"\n{len(P)} predicciones | clases (Buena, Aceptable, Dañina+): {res['distribucion_clases']}")
    print(f"{'Modelo':45s} {'Acc':>6s} {'F1mac':>6s} {'RecD+':>6s}")
    for nombre, col in modelos.items():
        m = nb.metricas(y, P[col].astype(int).values)
        m["recall_danina"] = recall_danina(y, P[col].astype(int).values)
        res["modelos"][nombre] = m
        print(f"{nombre:45s} {m['accuracy']:6.3f} {m['f1_macro']:6.3f} {m['recall_danina']:6.3f}")
    res["por_mes"] = {
        mes: {"n": int(len(g)), "danina": int((g["y"] == 2).sum()),
              "acc_nb": float((g["pred_A_basico"] == g["y"]).mean()),
              "acc_persistencia": float((g["pred_persistencia"] == g["y"]).mean())}
        for mes, g in P.groupby("mes")}
    res["bootstrap_nb_menos_persistencia"] = bootstrap(P, "pred_A_basico", "pred_persistencia")
    print("\nNB - persistencia (IC 95 %, bootstrap de días):", res["bootstrap_nb_menos_persistencia"])
    SALIDA.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    figuras(P, res)
    print("Guardado:", SALIDA.relative_to(BASE), "y figuras en", FIG.relative_to(BASE))


if __name__ == "__main__":
    main()
