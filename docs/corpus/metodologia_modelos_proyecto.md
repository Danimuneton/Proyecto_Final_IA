---
id: METODO
titulo: Metodología de los modelos del proyecto (HMM de estado latente y regresión logística para el día siguiente)
fuente: Documento técnico del equipo del proyecto final SI3003 (EAFIT, 2026). Resultados en reports/metricas_parte1.md.
url: https://github.com/Danimuneton/Proyecto_Final_IA
---

# Cómo funcionan los modelos de este proyecto

## Estado latente con un Modelo Oculto de Markov (HMM)
La calidad del aire de cada estación se modela con un HMM de 3 estados ocultos: Normal, Alerta y Episodio crítico. La observación de cada día es el logaritmo del PM2.5 promedio diario, con emisión gaussiana por estado. Los parámetros (probabilidades iniciales, matriz de transición, medias y varianzas) se estiman con el algoritmo de Baum-Welch (EM) implementado por el equipo, usando a la vez las secuencias de las 17 estaciones. Los días sin dato válido no aportan evidencia pero la cadena oculta sigue evolucionando. Con el modelo entrenado sobre el año completo, las medias de los estados equivalen a unos 12, 18 y 30 µg/m3, y cada estado tiende a persistir (probabilidad de quedarse en el mismo estado de un día al siguiente cercana a 0,94-0,95).

El estado del HMM es un régimen estadístico de la serie, no una declaratoria oficial. Por ejemplo, el estado "Episodio crítico" (media cercana a 30 µg/m3) indica un periodo sostenido de concentraciones altas para el valle y aumento del riesgo de días Naranja, aunque muchos de sus días sigan en ICA Amarillo. Las declaratorias oficiales de Prevención, Alerta o Emergencia las hace el AMVA según la Resolución 2254 de 2017.

Se usa el filtrado (algoritmo forward), que estima P(estado de hoy | datos hasta hoy) sin mirar el futuro, y Viterbi para la secuencia de estados más probable en las gráficas históricas.

## Predicción de la categoría ICA de mañana
Un clasificador de regresión logística multinomial (softmax), implementado por el equipo con descenso de gradiente (Adam) y regularización L2, predice la categoría del PM2.5 de mañana en una estación: Buena (0-12 µg/m3), Aceptable (13-37) o Dañina para grupos sensibles o peor (38 o más, que coincide con el nivel de Prevención). Variables: PM2.5 de hoy y de ayer (en logaritmo), cambio diario, media de 3 días, promedio del valle, categoría de hoy, probabilidades filtradas del HMM, lluvia de hoy, meteorología de mañana (lluvia, viento, radiación, nubosidad; tomadas del reanálisis ERA5 como si fueran un pronóstico perfecto, o fijadas por el usuario como escenario) y si mañana es domingo.

Como la clase Dañina es poco frecuente (3 % de los días estación del año y 5 % del periodo de prueba), la regla de decisión divide la probabilidad de cada clase por su frecuencia elevada a un exponente alpha. Alpha se elige con una validación interna (últimos 30 días de cada entrenamiento) y resultó 0,5 en la mayoría de meses. Las probabilidades que se muestran al usuario son las del modelo, sin ese ajuste.

## Evaluación sin mirar el futuro
Se usó validación con origen móvil mensual: para cada mes de enero a julio de 2024 se re-entrenan el HMM y el clasificador solo con datos anteriores y se predice cada día del mes (3.309 predicciones estación-día). Líneas base: clase mayoritaria, persistencia (mañana igual a hoy) y cadena de Markov sobre las clases.

## Resultados principales
- Persistencia: exactitud 0,799, exactitud balanceada 0,685, F1 macro 0,684, recall de la clase Dañina 0,60.
- Modelo final (regresión logística con variables del HMM): exactitud 0,808, exactitud balanceada 0,703, F1 macro 0,686, recall de Dañina 0,63.
- En probabilidades, el modelo final reduce la pérdida logarítmica de 0,536 (cadena de Markov) a 0,439 y el puntaje de Brier de 0,300 a 0,265; ambas mejoras tienen intervalos de confianza del 95 % que excluyen el cero.
- Frente a la persistencia, las mejoras en exactitud y F1 macro son pequeñas y sus intervalos de confianza incluyen el cero: la persistencia es una línea base muy fuerte porque el PM2.5 cambia lentamente.
- El HMM describe mucho mejor los datos que una gaussiana sin dinámica temporal (log-verosimilitud predictiva por observación de -0,20 frente a -0,71 en promedio de los 7 meses) y sus variables mejoran la exactitud balanceada y la pérdida logarítmica del clasificador.

## Limitaciones
- Solo hay un año de datos (julio 2023 a julio 2024) y pocos días Dañinos (172 estación-días); el desempeño en temporadas más severas no está probado.
- La meteorología de mañana proviene de un reanálisis para un solo punto, usado como pronóstico perfecto; con un pronóstico real el desempeño sería menor.
- El modelo no conoce incendios forestales, eventos especiales ni medidas de restricción vigentes.
- Es una herramienta académica de apoyo: no reemplaza los pronósticos del SIATA, las declaratorias del AMVA ni el consejo médico.
