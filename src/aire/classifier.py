"""Regresión logística multinomial (softmax) implementada desde cero con NumPy.

Minimiza la entropía cruzada ponderada por clase + penalización L2 mediante
descenso de gradiente por lotes completos con Adam. Incluye la estandarización
de variables (ajustada sólo con datos de entrenamiento).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@dataclass
class SoftmaxRegression:
    n_classes: int = 3
    l2: float = 1e-2
    lr: float = 0.05
    n_iter: int = 3000
    tol: float = 1e-7
    class_weight: str | None = None
    alpha_: float = 0.0
    prior_: np.ndarray = field(default=None, repr=False)
    feature_names: list = field(default_factory=list)
    W: np.ndarray = field(default=None, repr=False)
    b: np.ndarray = field(default=None, repr=False)
    mean_: np.ndarray = field(default=None, repr=False)
    std_: np.ndarray = field(default=None, repr=False)
    loss_history: list = field(default_factory=list, repr=False)

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SoftmaxRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n, d = X.shape
        K = self.n_classes
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        Xs = self._scale(X)
        Y = np.eye(K)[y]
        counts = np.bincount(y, minlength=K).astype(float)
        self.prior_ = counts / counts.sum()
        if self.class_weight == "balanced":
            cw = np.where(counts > 0, n / (K * np.maximum(counts, 1)), 0.0)
        else:
            cw = np.ones(K)
        sw = cw[y]
        sw = sw / sw.sum()

        self.W = np.zeros((d, K))
        self.b = np.zeros(K)
        # Adam
        mW = np.zeros_like(self.W); vW = np.zeros_like(self.W)
        mb = np.zeros_like(self.b); vb = np.zeros_like(self.b)
        b1, b2, eps = 0.9, 0.999, 1e-8
        prev = np.inf
        self.loss_history = []
        for t in range(1, self.n_iter + 1):
            P = softmax(Xs @ self.W + self.b)
            loss = -(sw * np.log(np.clip((P * Y).sum(axis=1), 1e-12, None))).sum() \
                + 0.5 * self.l2 * (self.W ** 2).sum()
            G = (P - Y) * sw[:, None]           # dL/dz
            gW = Xs.T @ G + self.l2 * self.W
            gb = G.sum(axis=0)
            mW = b1 * mW + (1 - b1) * gW; vW = b2 * vW + (1 - b2) * gW ** 2
            mb = b1 * mb + (1 - b1) * gb; vb = b2 * vb + (1 - b2) * gb ** 2
            self.W -= self.lr * (mW / (1 - b1 ** t)) / (np.sqrt(vW / (1 - b2 ** t)) + eps)
            self.b -= self.lr * (mb / (1 - b1 ** t)) / (np.sqrt(vb / (1 - b2 ** t)) + eps)
            self.loss_history.append(float(loss))
            if abs(prev - loss) < self.tol:
                break
            prev = loss
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(self._scale(np.asarray(X, dtype=float)) @ self.W + self.b)

    def decision_scores(self, P: np.ndarray) -> np.ndarray:
        """Regla de decisión sensible al desbalance: P(c|x) / prior(c)^alpha.

        alpha = 0 es el argmax usual (maximiza exactitud); alpha = 1 equivale a
        clases balanceadas. alpha se elige por validación interna (ver modeling.py).
        """
        if self.prior_ is None or self.alpha_ == 0:
            return P
        return P / np.maximum(self.prior_, 1e-3) ** self.alpha_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.decision_scores(self.predict_proba(X)).argmax(axis=1)

    def coef_table(self) -> list[dict]:
        """Coeficientes en escala estandarizada (interpretables como importancia)."""
        rows = []
        for i, name in enumerate(self.feature_names):
            rows.append({"variable": name, **{f"clase_{k}": float(self.W[i, k])
                                               for k in range(self.n_classes)}})
        return rows

    def to_dict(self) -> dict:
        return {"n_classes": self.n_classes, "l2": self.l2, "feature_names": self.feature_names,
                "W": self.W.tolist(), "b": self.b.tolist(), "mean": self.mean_.tolist(),
                "std": self.std_.tolist(), "prior": self.prior_.tolist(), "alpha": self.alpha_}

    @classmethod
    def from_dict(cls, d: dict) -> "SoftmaxRegression":
        m = cls(n_classes=d["n_classes"], l2=d["l2"], feature_names=d["feature_names"])
        m.W, m.b = np.array(d["W"]), np.array(d["b"])
        m.mean_, m.std_ = np.array(d["mean"]), np.array(d["std"])
        m.prior_, m.alpha_ = np.array(d["prior"]), float(d["alpha"])
        return m

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SoftmaxRegression":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Métricas (implementadas a mano; sklearn se usa sólo para verificarlas)
# ---------------------------------------------------------------------------
def confusion(y_true, y_pred, k: int = 3) -> np.ndarray:
    M = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        M[int(t), int(p)] += 1
    return M


def metrics(y_true, y_pred, proba: np.ndarray | None = None, k: int = 3) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    M = confusion(y_true, y_pred, k)
    tp = np.diag(M).astype(float)
    support = M.sum(axis=1).astype(float)
    predicted = M.sum(axis=0).astype(float)
    recall = np.divide(tp, support, out=np.zeros(k), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros(k), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(k), where=(precision + recall) > 0)
    present = support > 0
    out = {
        "n": int(len(y_true)),
        "accuracy": float(tp.sum() / M.sum()),
        "balanced_accuracy": float(recall[present].mean()),
        "macro_f1": float(f1[present].mean()),
        "recall_por_clase": recall.tolist(),
        "precision_por_clase": precision.tolist(),
        "f1_por_clase": f1.tolist(),
        "soporte": support.astype(int).tolist(),
        "matriz_confusion": M.tolist(),
    }
    if proba is not None:
        P = np.clip(np.asarray(proba, dtype=float), 1e-12, 1)
        P = P / P.sum(axis=1, keepdims=True)
        Y = np.eye(k)[y_true]
        out["log_loss"] = float(-np.log((P * Y).sum(axis=1)).mean())
        out["brier"] = float(((P - Y) ** 2).sum(axis=1).mean())
    return out
