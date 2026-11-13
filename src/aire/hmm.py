"""Modelo Oculto de Markov (HMM) gaussiano implementado desde cero con NumPy.

Uso en el proyecto: estimar el estado latente diario de la calidad del aire
(Normal, Alerta, Episodio crítico) a partir de la observación ruidosa
x_t = log(PM2.5 diario) de cada estación.

Implementa:
- Algoritmo forward-backward con escalamiento (Rabiner, 1989) -> filtrado y suavizado.
- Baum-Welch (EM) con múltiples secuencias (una por estación) y datos faltantes:
  un día sin dato tiene verosimilitud de emisión 1 (no aporta evidencia), así la
  cadena oculta sigue evolucionando sólo con la matriz de transición.
- Viterbi en espacio logarítmico.
- Predicción a un paso: P(z_{t+1} | x_{1:t}) = alpha_t A, y la probabilidad de cada
  categoría normativa del día siguiente integrando la emisión gaussiana.

No usa hmmlearn ni ninguna librería que resuelva el HMM.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_LOG2PI = math.log(2 * math.pi)


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


@dataclass
class GaussianHMM:
    n_states: int = 3
    n_iter: int = 200
    tol: float = 1e-5
    min_var: float = 1e-3
    pi: np.ndarray = field(default=None, repr=False)
    A: np.ndarray = field(default=None, repr=False)
    mu: np.ndarray = field(default=None, repr=False)
    var: np.ndarray = field(default=None, repr=False)
    history: list = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------ utils
    def _emission(self, x: np.ndarray) -> np.ndarray:
        """Matriz T x K de verosimilitudes p(x_t | z_t = k). NaN -> 1."""
        x = np.asarray(x, dtype=float)
        B = np.ones((len(x), self.n_states))
        obs = ~np.isnan(x)
        if obs.any():
            d = x[obs, None] - self.mu[None, :]
            logp = -0.5 * (_LOG2PI + np.log(self.var)[None, :] + d * d / self.var[None, :])
            B[obs] = np.exp(logp)
        return np.maximum(B, 1e-300)

    def _forward(self, B: np.ndarray):
        T, K = B.shape
        alpha = np.zeros((T, K))
        c = np.zeros(T)
        a = self.pi * B[0]
        c[0] = a.sum()
        alpha[0] = a / c[0]
        for t in range(1, T):
            a = (alpha[t - 1] @ self.A) * B[t]
            c[t] = a.sum()
            alpha[t] = a / c[t]
        return alpha, c

    def _backward(self, B: np.ndarray, c: np.ndarray):
        T, K = B.shape
        beta = np.ones((T, K))
        for t in range(T - 2, -1, -1):
            beta[t] = (self.A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
        return beta

    # ------------------------------------------------------------- inicializar
    def _init_params(self, seqs: list[np.ndarray]) -> None:
        allx = np.concatenate([s[~np.isnan(s)] for s in seqs])
        K = self.n_states
        # Inicialización determinista por cuantiles (reproducible)
        qs = np.quantile(allx, np.linspace(0, 1, K + 1))
        mu, var = [], []
        for k in range(K):
            m = allx[(allx >= qs[k]) & (allx <= qs[k + 1])]
            mu.append(m.mean())
            var.append(max(m.var(), self.min_var))
        self.mu = np.array(mu)
        self.var = np.array(var)
        self.pi = np.full(K, 1.0 / K)
        self.A = np.full((K, K), 0.1 / (K - 1))
        np.fill_diagonal(self.A, 0.9)

    # -------------------------------------------------------------- Baum-Welch
    def fit(self, seqs: list[np.ndarray]) -> "GaussianHMM":
        seqs = [np.asarray(s, dtype=float) for s in seqs if np.isfinite(s).sum() > 1]
        self._init_params(seqs)
        prev = -np.inf
        self.history = []
        for _ in range(self.n_iter):
            K = self.n_states
            pi_acc = np.zeros(K)
            xi_acc = np.zeros((K, K))
            g_acc = np.zeros(K)
            gx_acc = np.zeros(K)
            gxx_acc = np.zeros(K)
            ll = 0.0
            for x in seqs:
                B = self._emission(x)
                alpha, c = self._forward(B)
                beta = self._backward(B, c)
                gamma = alpha * beta
                gamma /= gamma.sum(axis=1, keepdims=True)
                ll += np.log(c).sum()
                pi_acc += gamma[0]
                # xi_t(i,j) = alpha_t(i) A_ij B_{t+1}(j) beta_{t+1}(j) / c_{t+1}
                xi = (alpha[:-1, :, None] * self.A[None, :, :]
                      * (B[1:] * beta[1:])[:, None, :]) / c[1:, None, None]
                xi_acc += xi.sum(axis=0)
                obs = ~np.isnan(x)
                go = gamma[obs]
                xo = x[obs]
                g_acc += go.sum(axis=0)
                gx_acc += (go * xo[:, None]).sum(axis=0)
                gxx_acc += (go * (xo[:, None] ** 2)).sum(axis=0)
            # M-step
            self.pi = pi_acc / pi_acc.sum()
            self.A = xi_acc / xi_acc.sum(axis=1, keepdims=True)
            self.mu = gx_acc / g_acc
            self.var = np.maximum(gxx_acc / g_acc - self.mu ** 2, self.min_var)
            self.history.append(ll)
            if ll - prev < self.tol * abs(ll):
                break
            prev = ll
        self._sort_states()
        return self

    def _sort_states(self) -> None:
        order = np.argsort(self.mu)
        self.mu, self.var, self.pi = self.mu[order], self.var[order], self.pi[order]
        self.A = self.A[np.ix_(order, order)]

    # ------------------------------------------------------------- inferencia
    def loglik(self, x: np.ndarray) -> float:
        _, c = self._forward(self._emission(x))
        return float(np.log(c).sum())

    def filter(self, x: np.ndarray) -> np.ndarray:
        """P(z_t | x_{1:t}) para cada t (filtrado, causal)."""
        alpha, _ = self._forward(self._emission(x))
        return alpha

    def smooth(self, x: np.ndarray) -> np.ndarray:
        """P(z_t | x_{1:T}) (suavizado, usa toda la secuencia)."""
        B = self._emission(x)
        alpha, c = self._forward(B)
        beta = self._backward(B, c)
        g = alpha * beta
        return g / g.sum(axis=1, keepdims=True)

    def viterbi(self, x: np.ndarray) -> np.ndarray:
        B = self._emission(x)
        T, K = B.shape
        logA = np.log(self.A)
        delta = np.log(self.pi) + np.log(B[0])
        psi = np.zeros((T, K), dtype=int)
        for t in range(1, T):
            scores = delta[:, None] + logA
            psi[t] = scores.argmax(axis=0)
            delta = scores.max(axis=0) + np.log(B[t])
        path = np.zeros(T, dtype=int)
        path[-1] = delta.argmax()
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    def predict_next_state(self, alpha_t: np.ndarray) -> np.ndarray:
        return alpha_t @ self.A

    def class_probs_given_state(self, cuts_log: list[float]) -> np.ndarray:
        """Matriz K x C: P(clase | estado) integrando la gaussiana entre los cortes."""
        sd = np.sqrt(self.var)
        edges = [-np.inf] + list(cuts_log) + [np.inf]
        P = np.zeros((self.n_states, len(edges) - 1))
        for j in range(len(edges) - 1):
            lo = _norm_cdf((edges[j] - self.mu) / sd) if np.isfinite(edges[j]) else 0.0
            hi = _norm_cdf((edges[j + 1] - self.mu) / sd) if np.isfinite(edges[j + 1]) else 1.0
            P[:, j] = hi - lo
        return P

    # ---------------------------------------------------------- persistencia
    def to_dict(self) -> dict:
        return {"n_states": self.n_states, "pi": self.pi.tolist(), "A": self.A.tolist(),
                "mu": self.mu.tolist(), "var": self.var.tolist(), "history": self.history}

    @classmethod
    def from_dict(cls, d: dict) -> "GaussianHMM":
        m = cls(n_states=d["n_states"])
        m.pi, m.A = np.array(d["pi"]), np.array(d["A"])
        m.mu, m.var = np.array(d["mu"]), np.array(d["var"])
        m.history = d.get("history", [])
        return m

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "GaussianHMM":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def iid_gaussian_loglik(train: list[np.ndarray], test: list[np.ndarray]) -> float:
    """Línea base sin dinámica temporal: una gaussiana i.i.d. ajustada en train."""
    x = np.concatenate([s[~np.isnan(s)] for s in train])
    mu, var = x.mean(), x.var()
    ll = 0.0
    for s in test:
        o = s[~np.isnan(s)]
        ll += float((-0.5 * (_LOG2PI + np.log(var) + (o - mu) ** 2 / var)).sum())
    return ll
