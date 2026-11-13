| Modelo | Exactitud | Exactitud balanceada | F1 macro | Recall Dañina | Log-loss | Brier |
|---|---|---|---|---|---|---|
| Línea base: clase mayoritaria | 0.757 | 0.333 | 0.287 | 0.000 | — | — |
| Línea base: persistencia (mañana = hoy) | 0.799 | 0.685 | 0.684 | 0.601 | — | — |
| Línea base: cadena de Markov de clases | 0.796 | 0.654 | 0.664 | 0.506 | 0.536 | 0.300 |
| HMM propio (predicción a un paso) | 0.758 | 0.339 | 0.300 | 0.000 | 0.522 | 0.313 |
| Reg. logística propia: sólo info de hoy | 0.795 | 0.669 | 0.661 | 0.565 | 0.448 | 0.275 |
| Reg. logística propia: info de hoy + domingo (sin meteorología) | 0.813 | 0.674 | 0.684 | 0.524 | 0.434 | 0.262 |
| Reg. logística propia: completa sin HMM | 0.811 | 0.681 | 0.686 | 0.571 | 0.445 | 0.267 |
| Reg. logística propia: completa + HMM (modelo final) | 0.808 | 0.703 | 0.686 | 0.631 | 0.439 | 0.265 |

Log-verosimilitud predictiva por observación (mayor es mejor):

| Mes | HMM (3 estados) | Gaussiana i.i.d. |
|---|---|---|
| 2024-01 | -0.346 | -0.580 |
| 2024-02 | -0.236 | -0.533 |
| 2024-03 | -0.255 | -1.557 |
| 2024-04 | -0.228 | -0.654 |
| 2024-05 | -0.133 | -0.498 |
| 2024-06 | -0.213 | -0.690 |
| 2024-07 | -0.018 | -0.487 |

Diferencia (modelo final − persistencia), IC 95 % por bootstrap de días:

- accuracy: +0.009 [-0.008, +0.024]
- macro_f1: +0.002 [-0.031, +0.038]
- balanced_accuracy: +0.017 [-0.018, +0.061]

Diferencia (modelo final − mismo modelo sin variables del HMM):

- accuracy: -0.003 [-0.016, +0.009]
- macro_f1: -0.000 [-0.027, +0.028]
- balanced_accuracy: +0.022 [+0.001, +0.053]

Métricas probabilísticas (negativo = el modelo final es mejor), IC 95 %:

- log_loss, final − cadena de Markov: -0.0976 [-0.1329, -0.0644]
- brier, final − cadena de Markov: -0.0357 [-0.0509, -0.0212]
- log_loss, final − sin HMM: -0.0054 [-0.0090, -0.0017]
- brier, final − sin HMM: -0.0020 [-0.0039, -0.0001]

Verificación contra scikit-learn (mismo objetivo): acuerdo en argmax = 100.000%, diferencia máxima de probabilidad = 0.0059.
