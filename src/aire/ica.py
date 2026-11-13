"""Índice de Calidad del Aire (ICA) para PM2.5 según la Resolución 2254 de 2017."""
from __future__ import annotations

import math

import numpy as np

from .config import CLASES, ICA_PM25


def _redondear(conc: float) -> int:
    # Los puntos de corte de la Tabla 6 son enteros; se redondea al entero más cercano.
    return int(math.floor(conc + 0.5))


def categoria_ica(conc: float) -> dict:
    """Devuelve ICA, categoría y color para una concentración de PM2.5 de 24 h (ug/m3).

    ICA_p = (I_alto - I_bajo) / (PC_alto - PC_bajo) * (C_p - PC_bajo) + I_bajo   (Art. 21)
    """
    if conc is None or (isinstance(conc, float) and math.isnan(conc)):
        return {"ica": None, "categoria": "Sin dato", "color": "#BBBBBB"}
    c = max(0, _redondear(conc))
    for lo, hi, ilo, ihi, cat, color in ICA_PM25:
        if lo <= c <= hi:
            ica = (ihi - ilo) / (hi - lo) * (c - lo) + ilo
            return {"ica": round(ica), "categoria": cat, "color": color}
    return {"ica": 500, "categoria": "Peligrosa", "color": "#7E0023"}


def clase_3(conc: float) -> int:
    """Agrupa en 3 clases: 0 Buena (0-12), 1 Aceptable (13-37), 2 Dañina GS o peor (>=38)."""
    c = _redondear(conc)
    if c <= 12:
        return 0
    if c <= 37:
        return 1
    return 2


def clase_3_array(conc: np.ndarray) -> np.ndarray:
    c = np.floor(np.asarray(conc, dtype=float) + 0.5)
    return np.where(c <= 12, 0, np.where(c <= 37, 1, 2))


def nombre_clase(k: int) -> str:
    return CLASES[int(k)]


# Límites en ug/m3 de las 3 clases sobre la escala continua (útil para el HMM)
LIMITES_CLASES = [12.5, 37.5]
