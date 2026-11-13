"""Parte 1: Naive Bayes categórico implementado desde cero con NumPy.

Tarea: dada la situación de hoy y ayer en una estación, predecir la categoría de
PM2.5 (promedio 24 h) de MAÑANA.

Uso:
    python src/modelo.py          # entrena, evalúa y guarda models/*.json
    from modelo import predecir_manana   # lo usará el agente
"""
import json
import pathlib

import numpy as np
import pandas as pd

BASE = pathlib.Path(__file__).resolve().parent.parent
CSV = BASE / "data" / "processed" / "pm25_diario.csv"
MODELO = BASE / "models" / "nb_pm25.json"
METRICAS = BASE / "models" / "metricas.json"

CLASES = ["Buena", "Aceptable", "Dañina a sensibles o peor"]
# Puntos de corte de PM2.5 (24 h, µg/m³) según el ICA de la Res. 2254 de 2017:
# Buena 0-12, Aceptable 13-37, desde 38 dañina a grupos sensibles o peor.
CORTES = [12.0, 37.0]
CORTES_FINOS = [8.0, 12.0, 16.0, 20.0, 25.0, 37.0]   # nivel de PM2.5 en 7 tramos (incluye 12 y 37)
UMBRAL_TENDENCIA = 3.0                  # µg/m³ para decir que "sube" o "baja"
MESES_LLUVIOSOS = [3, 4, 5, 9, 10, 11]  # supuesto: temporadas de lluvia del Valle de Aburrá

# Variables posibles y cuántos valores toma cada una
COLUMNAS = {"categoria_hoy": 3, "categoria_ayer": 3, "nivel_hoy": 7, "tendencia": 3,
            "fin_de_semana_manana": 2, "temporada_lluviosa_manana": 2}
# Conjuntos de variables que se comparan (la elección se hace con VALIDACIÓN, no con la prueba)
CONJUNTOS = {
    "A_basico": ["categoria_hoy", "categoria_ayer", "tendencia",
                 "fin_de_semana_manana", "temporada_lluviosa_manana"],
    "B_nivel_fino": ["nivel_hoy", "tendencia", "fin_de_semana_manana", "temporada_lluviosa_manana"],
    "C_nivel_fino_sin_calendario": ["nivel_hoy", "tendencia"],
}


# ---------------------------------------------------------------- datos
def cargar_datos():
    df = pd.read_csv(CSV, parse_dates=["fecha"])
    return df[~df["nombre"].str.startswith("_OFF")]   # estación marcada como apagada


def categorizar(pm, cortes=CORTES):
    pm = np.asarray(pm, dtype=float)
    cat = np.digitize(pm, cortes, right=True).astype(float)   # con CORTES: <=12 ->0, <=37 ->1, >37 ->2
    cat[np.isnan(pm)] = np.nan
    return cat


def construir_features(hoy, ayer, fecha_manana):
    hoy, ayer = np.asarray(hoy, float), np.asarray(ayer, float)
    dif = hoy - ayer
    tendencia = np.where(dif < -UMBRAL_TENDENCIA, 0, np.where(dif > UMBRAL_TENDENCIA, 2, 1)).astype(float)
    tendencia[np.isnan(dif)] = np.nan
    fm = pd.DatetimeIndex(fecha_manana)
    return pd.DataFrame({
        "categoria_hoy": categorizar(hoy),
        "categoria_ayer": categorizar(ayer),
        "nivel_hoy": categorizar(hoy, CORTES_FINOS),
        "tendencia": tendencia,
        "fin_de_semana_manana": (fm.day_of_week >= 5).astype(float),
        "temporada_lluviosa_manana": np.isin(fm.month, MESES_LLUVIOSOS).astype(float),
    })


def construir_dataset(df):
    partes = []
    for est, g in df.groupby("estacion"):
        s = g.set_index("fecha")["pm25"].asfreq("D")          # calendario diario completo
        t = construir_features(s.values, s.shift(1).values, s.index + pd.Timedelta(days=1))
        t["y"] = categorizar(s.shift(-1).values)
        t["fecha"], t["estacion"] = s.index, est
        partes.append(t)
    return pd.concat(partes, ignore_index=True).dropna()


def split_temporal(t, frac=0.75):
    """Entrena con el pasado y prueba con lo más reciente (nunca al azar)."""
    fechas = np.sort(t["fecha"].unique())
    corte = pd.Timestamp(fechas[int(len(fechas) * frac)])
    return t[t["fecha"] < corte], t[t["fecha"] >= corte], corte


# ---------------------------------------------------------------- modelo
class NaiveBayesCategorico:
    def __init__(self, alpha=1.0):
        self.alpha = alpha            # suavizado de Laplace
        self.columnas = None          # nombres de las variables usadas

    def fit(self, X, y, n_valores, n_clases):
        X, y = X.astype(int), y.astype(int)
        self.n_valores, self.n_clases = list(n_valores), n_clases
        cont_c = np.bincount(y, minlength=n_clases).astype(float)
        self.log_prior = np.log((cont_c + self.alpha) / (len(y) + self.alpha * n_clases))
        self.log_lik = []
        for j, nv in enumerate(self.n_valores):
            cont = np.zeros((n_clases, nv))
            np.add.at(cont, (y, X[:, j]), 1)                   # conteos clase x valor
            self.log_lik.append(np.log((cont + self.alpha) / (cont_c[:, None] + self.alpha * nv)))
        return self

    def log_proba(self, X):
        X = X.astype(int)
        lp = np.tile(self.log_prior, (len(X), 1))
        for j in range(X.shape[1]):
            lp += self.log_lik[j][:, X[:, j]].T                # suma de log-verosimilitudes
        return lp

    def predict_proba(self, X):
        lp = self.log_proba(X)
        lp -= lp.max(axis=1, keepdims=True)
        p = np.exp(lp)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.log_proba(X).argmax(axis=1)

    def to_dict(self):
        return {"alpha": self.alpha, "columnas": self.columnas, "n_valores": self.n_valores, "n_clases": self.n_clases,
                "log_prior": self.log_prior.tolist(), "log_lik": [m.tolist() for m in self.log_lik]}

    @classmethod
    def from_dict(cls, d):
        m = cls(d["alpha"])
        m.columnas = d.get("columnas")
        m.n_valores, m.n_clases = d["n_valores"], d["n_clases"]
        m.log_prior = np.array(d["log_prior"])
        m.log_lik = [np.array(a) for a in d["log_lik"]]
        return m


# ---------------------------------------------------------------- métricas (a mano)
def metricas(y, yhat, k=len(CLASES)):
    y, yhat = np.asarray(y).astype(int), np.asarray(yhat).astype(int)
    m = np.zeros((k, k), int)
    np.add.at(m, (y, yhat), 1)                                 # filas = real, columnas = predicho
    f1 = []
    for c in range(k):
        tp = m[c, c]
        fp, fn = m[:, c].sum() - tp, m[c, :].sum() - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    presentes = [c for c in range(k) if m[c, :].sum() > 0]     # clases con soporte en la prueba
    return {"accuracy": float(np.trace(m) / m.sum()),
            "f1_macro": float(np.mean([f1[c] for c in presentes])),
            "f1_por_clase": [float(x) for x in f1], "matriz_confusion": m.tolist()}


# ---------------------------------------------------------------- uso por el agente
def cargar_modelo():
    return NaiveBayesCategorico.from_dict(json.load(open(MODELO)))


def resolver_estacion(df, estacion):
    est = df.drop_duplicates("estacion")[["estacion", "nombre"]]
    if str(estacion).strip().isdigit():
        r = est[est["estacion"] == int(estacion)]
    else:
        r = est[est["nombre"].str.contains(str(estacion), case=False, regex=False)]
    return r


def predecir_manana(estacion, fecha=None):
    """Devuelve un dict con la predicción de mañana, o {'error': ...} si no se puede."""
    df = cargar_datos()
    r = resolver_estacion(df, estacion)
    if len(r) == 0:
        return {"error": f"No encontré la estación '{estacion}'.",
                "estaciones_disponibles": df.drop_duplicates("estacion")["nombre"].tolist()}
    if len(r) > 1:
        return {"error": f"'{estacion}' es ambiguo; coincide con varias estaciones.",
                "coincidencias": r["nombre"].tolist()}
    cod, nombre = int(r.iloc[0]["estacion"]), r.iloc[0]["nombre"]
    s = df[df["estacion"] == cod].set_index("fecha")["pm25"].asfreq("D")
    validos = s.notna() & s.shift(1).notna()
    if fecha is None:
        if not validos.any():
            return {"error": "No hay dos días consecutivos con datos para esta estación."}
        base = validos[validos].index[-1]
    else:
        base = pd.Timestamp(fecha)
        if base not in s.index or not validos.loc[base]:
            return {"error": f"No hay datos de hoy y ayer para {base.date()} en {nombre}.",
                    "rango_datos": [str(s.index.min().date()), str(s.index.max().date())]}
    modelo = cargar_modelo()
    X = construir_features([s[base]], [s.shift(1)[base]], [base + pd.Timedelta(days=1)])[modelo.columnas].values
    p = modelo.predict_proba(X)[0]
    return {"estacion": cod, "nombre": nombre,
            "fecha_base": str(base.date()), "fecha_prediccion": str((base + pd.Timedelta(days=1)).date()),
            "pm25_base": round(float(s[base]), 1),
            "categoria_base": CLASES[int(categorizar([s[base]])[0])],
            "categoria_predicha": CLASES[int(p.argmax())],
            "probabilidades": {c: round(float(x), 3) for c, x in zip(CLASES, p)}}


# ---------------------------------------------------------------- entrenamiento
def entrenar(datos, columnas):
    nb = NaiveBayesCategorico(alpha=1.0).fit(
        datos[columnas].values, datos["y"].values, [COLUMNAS[c] for c in columnas], len(CLASES))
    nb.columnas = list(columnas)
    return nb


def conteo(d):
    c = np.bincount(d["y"].astype(int), minlength=3)
    return ", ".join(f"{n}={k}" for n, k in zip(["Buena", "Aceptable", "Dañina+"], c))


if __name__ == "__main__":
    t = construir_dataset(cargar_datos())
    train, test, corte = split_temporal(t)          # prueba: lo más reciente, se toca UNA vez
    tr, val, corte_val = split_temporal(train)      # validación: para elegir variables
    print(f"Ejemplos: {len(t)} | train final: {len(train)} (< {corte.date()}) | prueba: {len(test)} (>= {corte.date()})")
    print(f"Dentro del train: sub-train {len(tr)} (< {corte_val.date()}) | validación {len(val)}")
    print("Clases  sub-train:", conteo(tr), "\n        validación:", conteo(val), "\n        prueba    :", conteo(test))

    # 1) Elegir el conjunto de variables con VALIDACIÓN
    val_res = {n: metricas(val["y"].values, entrenar(tr, c).predict(val[c].values)) for n, c in CONJUNTOS.items()}
    mejor = max(val_res, key=lambda n: (val_res[n]["f1_macro"], val_res[n]["accuracy"]))
    print(f"\n{'Conjunto de variables (validación)':40s} {'Accuracy':>9s} {'F1 macro':>9s}")
    for n, m in val_res.items():
        print(f"{n + (' <- elegido' if n == mejor else ''):40s} {m['accuracy']:9.3f} {m['f1_macro']:9.3f}")

    # 2) Reentrenar con todo el train y evaluar en la prueba
    y_te = test["y"].values
    y_tr = train["y"].values
    prueba = {}
    for n, c in CONJUNTOS.items():
        prueba[f"naive_bayes {n}" + (" (elegido)" if n == mejor else "")] = metricas(
            y_te, entrenar(train, c).predict(test[c].values))
    prueba["linea_base: mañana = hoy"] = metricas(y_te, test["categoria_hoy"].values)
    prueba["linea_base: clase mayoritaria"] = metricas(y_te, np.full(len(y_te), np.bincount(y_tr.astype(int)).argmax()))

    print(f"\n{'PRUEBA (datos más recientes)':48s} {'Accuracy':>9s} {'F1 macro':>9s}")
    for n, m in prueba.items():
        print(f"{n:48s} {m['accuracy']:9.3f} {m['f1_macro']:9.3f}")

    final = entrenar(train, CONJUNTOS[mejor])
    print(f"\nMatriz de confusión del modelo elegido en prueba (filas = real, columnas = predicho):")
    print(np.array(metricas(y_te, final.predict(test[final.columnas].values))["matriz_confusion"]))

    MODELO.parent.mkdir(exist_ok=True)
    json.dump(final.to_dict(), open(MODELO, "w"))
    json.dump({"conjunto_elegido": mejor, "corte_prueba": str(corte.date()), "corte_validacion": str(corte_val.date()),
               "n_train": len(train), "n_test": len(test), "validacion": val_res, "prueba": prueba},
              open(METRICAS, "w"), ensure_ascii=False, indent=2)
    print("\nGuardado:", MODELO.name, "y", METRICAS.name)
    print("Ejemplo:", predecir_manana(cargar_datos()["estacion"].iloc[0]))