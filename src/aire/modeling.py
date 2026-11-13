"""Construcción de variables, entrenamiento y evaluación temporal de los modelos de la Parte 1.

Problema de predicción: dado lo observado hasta el día t en una estación, predecir la
clase ICA del PM2.5 promedio del día t+1 (Buena / Aceptable / Dañina GS o peor).

Validación: origen móvil (rolling origin) mensual. Para cada mes de prueba M se
re-entrenan el HMM y el clasificador SÓLO con días anteriores a M, y se evalúa en M.
El hiperparámetro alpha de la regla de decisión se elige con una validación interna
(últimos 30 días del entrenamiento). Así no hay fuga de información del futuro.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C
from .classifier import SoftmaxRegression, metrics
from .data import daily_matrix, load_meteo
from .hmm import GaussianHMM, iid_gaussian_loglik
from .ica import LIMITES_CLASES, clase_3_array

LOG_CUTS = [float(np.log(c)) for c in LIMITES_CLASES]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]

FEATURES_HOY = [
    "log_pm_hoy", "log_pm_ayer", "delta_1d", "log_media_3d", "log_pm_valle_hoy",
    "hoy_buena", "hoy_danina", "p_alerta_hoy", "p_critico_hoy", "lluvia_hoy_log",
]
FEATURES_MANANA = [
    "lluvia_manana_log", "viento_manana_ms", "radiacion_manana_mj",
    "nubosidad_manana_pct", "domingo_manana",
]
FEATURES_HMM = ["p_alerta_hoy", "p_critico_hoy"]
FEATURES_COMPLETO = FEATURES_HOY + FEATURES_MANANA

DESCRIPCION_FEATURES = {
    "log_pm_hoy": "log del PM2.5 promedio de hoy en la estación",
    "log_pm_ayer": "log del PM2.5 de ayer",
    "delta_1d": "cambio de log PM2.5 entre ayer y hoy",
    "log_media_3d": "log de la media móvil de 3 días",
    "log_pm_valle_hoy": "log del PM2.5 promedio de todas las estaciones hoy",
    "hoy_buena": "1 si hoy la estación estuvo en ICA Buena",
    "hoy_danina": "1 si hoy la estación estuvo en ICA Dañina GS o peor",
    "p_alerta_hoy": "P(estado HMM = Alerta | datos hasta hoy)",
    "p_critico_hoy": "P(estado HMM = Episodio crítico | datos hasta hoy)",
    "lluvia_hoy_log": "log(1 + lluvia de hoy en mm)",
    "lluvia_manana_log": "log(1 + lluvia de mañana en mm) (pronóstico/escenario)",
    "viento_manana_ms": "viento medio de mañana a 10 m (m/s)",
    "radiacion_manana_mj": "radiación solar de mañana (MJ/m2)",
    "nubosidad_manana_pct": "nubosidad media de mañana (%)",
    "domingo_manana": "1 si mañana es domingo (menos tráfico)",
}


def _sequences(logm: pd.DataFrame, end: pd.Timestamp | None = None) -> list[np.ndarray]:
    sub = logm if end is None else logm[logm.index < end]
    return [sub[c].to_numpy() for c in sub.columns]


def fit_hmm(logm: pd.DataFrame, end: pd.Timestamp | None = None) -> GaussianHMM:
    return GaussianHMM(n_states=3).fit(_sequences(logm, end))


def hmm_filtered(hmm: GaussianHMM, logm: pd.DataFrame) -> dict[int, np.ndarray]:
    """Probabilidades filtradas P(z_t | x_1:t) por estación (T x 3)."""
    return {c: hmm.filter(logm[c].to_numpy()) for c in logm.columns}


def build_table(matrix: pd.DataFrame, filtered: dict[int, np.ndarray]) -> pd.DataFrame:
    """Una fila por (estación, día t) con variables de t, pronóstico de t+1 y objetivo."""
    meteo = load_meteo().set_index("fecha").reindex(matrix.index)
    valle = np.log(matrix.mean(axis=1, skipna=True))
    rows = []
    for c in matrix.columns:
        pm = matrix[c]
        lp = np.log(pm)
        f = filtered[c]
        df = pd.DataFrame(index=matrix.index)
        df["codigo"] = c
        df["pm_hoy"] = pm
        df["log_pm_hoy"] = lp
        df["log_pm_ayer"] = lp.shift(1).fillna(lp)
        df["delta_1d"] = df["log_pm_hoy"] - df["log_pm_ayer"]
        df["log_media_3d"] = np.log(pm.rolling(3, min_periods=1).mean())
        df["log_pm_valle_hoy"] = valle
        df["p_normal_hoy"] = f[:, 0]
        df["p_alerta_hoy"] = f[:, 1]
        df["p_critico_hoy"] = f[:, 2]
        df["lluvia_hoy_log"] = np.log1p(meteo["lluvia_mm"])
        df["lluvia_manana_log"] = np.log1p(meteo["lluvia_mm"].shift(-1))
        df["viento_manana_ms"] = meteo["viento_ms"].shift(-1)
        df["radiacion_manana_mj"] = meteo["radiacion_mj"].shift(-1)
        df["nubosidad_manana_pct"] = meteo["nubosidad_pct"].shift(-1)
        df["domingo_manana"] = ((matrix.index + pd.Timedelta(days=1)).day_of_week == 6).astype(float)
        df["pm_manana"] = pm.shift(-1)
        rows.append(df)
    t = pd.concat(rows).rename_axis("fecha").reset_index()
    t["clase_hoy"] = np.where(t["pm_hoy"].notna(), clase_3_array(t["pm_hoy"].fillna(0)), -1)
    t["clase_manana"] = np.where(t["pm_manana"].notna(),
                                 clase_3_array(t["pm_manana"].fillna(0)), -1)
    t["hoy_buena"] = (t["clase_hoy"] == 0).astype(float)
    t["hoy_danina"] = (t["clase_hoy"] == 2).astype(float)
    return t


def fit_classifier(tr: pd.DataFrame, feats: list[str], val_days: int = 30) -> SoftmaxRegression:
    """Ajusta la regresión logística y elige alpha por validación interna temporal."""
    cut = tr["fecha"].max() - pd.Timedelta(days=val_days)
    inner, val = tr[tr["fecha"] <= cut], tr[tr["fecha"] > cut]
    best_alpha, best_f1 = 0.0, -1.0
    if len(inner) > 100 and len(val) > 30:
        clf = SoftmaxRegression(feature_names=feats).fit(inner[feats].to_numpy(),
                                                          inner["clase_manana"].to_numpy())
        P = clf.predict_proba(val[feats].to_numpy())
        for a in ALPHAS:
            clf.alpha_ = a
            f1 = metrics(val["clase_manana"], clf.decision_scores(P).argmax(axis=1))["macro_f1"]
            if f1 > best_f1 + 1e-9:
                best_alpha, best_f1 = a, f1
    clf = SoftmaxRegression(feature_names=feats).fit(tr[feats].to_numpy(), tr["clase_manana"].to_numpy())
    clf.alpha_ = best_alpha
    return clf


def _hmm_next_class_proba(hmm: GaussianHMM, tab: pd.DataFrame) -> np.ndarray:
    alpha = tab[["p_normal_hoy", "p_alerta_hoy", "p_critico_hoy"]].to_numpy()
    return (alpha @ hmm.A) @ hmm.class_probs_given_state(LOG_CUTS)


def _markov_baseline(tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    """Línea base probabilística: cadena de Markov sobre las clases (con suavizado de Laplace)."""
    T = np.ones((3, 3))
    np.add.at(T, (tr["clase_hoy"].to_numpy(), tr["clase_manana"].to_numpy()), 1)
    T /= T.sum(axis=1, keepdims=True)
    return T[te["clase_hoy"].to_numpy()]


MODELOS = {
    "Línea base: clase mayoritaria": ("pred_mayoritaria", None),
    "Línea base: persistencia (mañana = hoy)": ("pred_persistencia", None),
    "Línea base: cadena de Markov de clases": ("pred_markov", "markov"),
    "HMM propio (predicción a un paso)": ("pred_hmm", "hmm"),
    "Reg. logística propia: sólo info de hoy": ("pred_logreg_hoy", "logreg_hoy"),
    "Reg. logística propia: info de hoy + domingo (sin meteorología)": ("pred_logreg_hoy_domingo", "logreg_hoy_domingo"),
    "Reg. logística propia: completa sin HMM": ("pred_logreg_sin_hmm", "logreg_sin_hmm"),
    "Reg. logística propia: completa + HMM (modelo final)": ("pred_logreg_completo", "logreg_completo"),
}


def rolling_origin_evaluation(test_months: list[str] | None = None) -> dict:
    """Backtesting mensual con re-entrenamiento. Devuelve métricas y predicciones."""
    matrix = daily_matrix()
    logm = np.log(matrix)
    if test_months is None:
        test_months = [f"2024-{m:02d}" for m in range(1, 8)]

    preds, hmm_ll, alphas = [], [], []
    for month in test_months:
        start = pd.Timestamp(month + "-01")
        end = start + pd.offsets.MonthBegin(1)
        hmm = fit_hmm(logm, end=start)
        filt = hmm_filtered(hmm, logm)  # filtrado causal: en t sólo usa x_1..x_t
        tab = build_table(matrix, filt)
        ok = (tab["clase_hoy"] >= 0) & (tab["clase_manana"] >= 0)
        d1 = pd.Timedelta(days=1)
        tr = tab[ok & (tab["fecha"] < start - d1)].dropna(subset=FEATURES_COMPLETO)
        te = tab[ok & (tab["fecha"] >= start - d1) & (tab["fecha"] < end - d1)].dropna(
            subset=FEATURES_COMPLETO)

        out = te[["fecha", "codigo", "pm_hoy", "pm_manana", "clase_hoy", "clase_manana"]].copy()
        out["mes"] = month
        out["pred_mayoritaria"] = int(np.bincount(tr["clase_manana"]).argmax())
        out["pred_persistencia"] = te["clase_hoy"].to_numpy()
        pm_ = _markov_baseline(tr, te)
        out["pred_markov"] = pm_.argmax(axis=1)
        out[[f"proba_markov_{k}" for k in range(3)]] = pm_
        ph = _hmm_next_class_proba(hmm, te)
        out["pred_hmm"] = ph.argmax(axis=1)
        out[[f"proba_hmm_{k}" for k in range(3)]] = ph

        variantes = {
            "logreg_hoy": FEATURES_HOY,
            "logreg_hoy_domingo": FEATURES_HOY + ["domingo_manana"],
            "logreg_completo": FEATURES_COMPLETO,
            "logreg_sin_hmm": [f for f in FEATURES_COMPLETO if f not in FEATURES_HMM],
        }
        for name, feats in variantes.items():
            clf = fit_classifier(tr, feats)
            p = clf.predict_proba(te[feats].to_numpy())
            out[f"pred_{name}"] = clf.decision_scores(p).argmax(axis=1)
            out[[f"proba_{name}_{k}" for k in range(3)]] = p
            if name == "logreg_completo":
                alphas.append({"mes": month, "alpha": clf.alpha_})
                ours_argmax = p.argmax(axis=1)
                # Verificación de corrección: mismo problema de optimización en scikit-learn
                try:
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.preprocessing import StandardScaler
                    sc = StandardScaler().fit(tr[feats])
                    sk = LogisticRegression(C=1.0 / (clf.l2 * len(tr)), max_iter=10000)
                    sk.fit(sc.transform(tr[feats]), tr["clase_manana"])
                    psk = sk.predict_proba(sc.transform(te[feats]))
                    out["sk_igual_argmax"] = (psk.argmax(axis=1) == ours_argmax)
                    out["sk_dif_proba"] = np.abs(psk - p).max(axis=1)
                except Exception:  # noqa: BLE001
                    out["sk_igual_argmax"] = np.nan
                    out["sk_dif_proba"] = np.nan
        preds.append(out)

        # HMM vs gaussiana i.i.d.: log-verosimilitud predictiva del mes de prueba
        seq_all = _sequences(logm, end)
        seq_tr = _sequences(logm, start)
        n_test = int(np.isfinite(logm[(logm.index >= start) & (logm.index < end)].to_numpy()).sum())
        ll_hmm = sum(hmm.loglik(a) - hmm.loglik(b) for a, b in zip(seq_all, seq_tr))
        ll_iid = iid_gaussian_loglik(seq_tr, [s[len(seq_tr[0]):] for s in seq_all])
        hmm_ll.append({"mes": month, "n_obs": n_test, "ll_hmm_por_obs": ll_hmm / n_test,
                       "ll_iid_por_obs": ll_iid / n_test})

    P = pd.concat(preds, ignore_index=True)
    y = P["clase_manana"].to_numpy()
    res = {}
    for name, (col, pcol) in MODELOS.items():
        proba = P[[f"proba_{pcol}_{k}" for k in range(3)]].to_numpy() if pcol else None
        res[name] = metrics(y, P[col].to_numpy(), proba)
    verif = {"acuerdo_argmax": float(P["sk_igual_argmax"].mean()),
             "max_dif_probabilidad": float(P["sk_dif_proba"].max()),
             "media_dif_probabilidad": float(P["sk_dif_proba"].mean())}
    return {"resultados": res, "predicciones": P, "hmm_loglik": hmm_ll,
            "verificacion_sklearn": verif, "alpha_por_mes": alphas, "meses_prueba": test_months}


def train_final() -> dict:
    """Entrena HMM y clasificador con todos los datos y los guarda en models/."""
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    matrix = daily_matrix()
    logm = np.log(matrix)
    hmm = fit_hmm(logm)
    hmm.save(C.MODELS_DIR / "hmm.json")
    tab = build_table(matrix, hmm_filtered(hmm, logm))
    ok = (tab["clase_hoy"] >= 0) & (tab["clase_manana"] >= 0)
    tr = tab[ok].dropna(subset=FEATURES_COMPLETO)
    clf = fit_classifier(tr, FEATURES_COMPLETO)
    clf.save(C.MODELS_DIR / "clasificador.json")
    return {"hmm": hmm.to_dict(), "n_entrenamiento": int(len(tr)), "alpha": clf.alpha_,
            "coeficientes": clf.coef_table()}


def save_json(obj, path) -> None:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=default), encoding="utf-8")
