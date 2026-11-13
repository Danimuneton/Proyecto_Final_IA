---
id: METEO
titulo: Meteorología y contaminación en el Valle de Aburrá - por qué se acumula el PM2.5
fuente: Síntesis elaborada por el equipo a partir de comunicados del AMVA y del SIATA (temporadas críticas), prensa regional (El Colombiano, Caracol Radio, El Tiempo, Minuto30) y del análisis exploratorio de los datos SIATA 2023-2024 y Open-Meteo (reanálisis ERA5) realizado en este proyecto.
url: https://siata.gov.co/siata_nuevo/
referencias: Open-Meteo Historical Weather API https://open-meteo.com/en/docs/historical-weather-api (datos ERA5, Copernicus/ECMWF, licencia CC BY 4.0)
---

# Meteorología y contaminación del aire en el Valle de Aburrá

## Un valle estrecho y semicerrado
El Valle de Aburrá es una cuenca angosta y semicerrada, rodeada de montañas, de unos 60 km de largo, con el fondo del valle a unos 1.500 metros sobre el nivel del mar en Medellín. Allí viven cerca de cuatro millones de personas en 10 municipios. La topografía limita la ventilación: cuando la atmósfera está estable y hay poco viento, los contaminantes emitidos por vehículos e industrias se quedan dentro de la cuenca y se acumulan día tras día.

## Estabilidad atmosférica, nubes y radiación
En las mañanas, la capa de aire cercana al suelo suele estar fría y estable (inversión térmica), lo que impide que los contaminantes asciendan. La radiación solar calienta el suelo, rompe esa estabilidad y hace crecer la capa de mezcla, que diluye los contaminantes. Cuando hay nubosidad persistente, llega menos radiación, la capa de mezcla crece poco y el PM2.5 se acumula. Esta situación fue la descrita por las autoridades en el estado de alerta de marzo de 2019: la cobertura parcial de nubes impidió el ingreso eficiente de radiación solar y favoreció la acumulación de contaminantes.

## Temporadas críticas: transición seca-lluviosa
El valle tiene un régimen bimodal de lluvias. Las concentraciones de PM2.5 aumentan en dos periodos al año que coinciden con la transición entre la temporada seca y la de lluvias: el primero entre febrero y abril (normalmente el más crítico, con mayor frecuencia en marzo) y el segundo entre octubre y noviembre. En esos meses hay más nubosidad baja, menos radiación en las mañanas y una atmósfera más estable. El SIATA emite pronósticos para estos periodos, por ejemplo de la probabilidad de que cinco o más estaciones pasen a ICA Naranja.

## Papel de la lluvia y del viento
La lluvia remueve partículas de la atmósfera (lavado húmedo) y suele asociarse con mejor calidad del aire en las horas y días siguientes, aunque los días nublados previos a la lluvia pueden ser críticos. El viento ayuda a renovar el aire de la cuenca; en el fondo del valle el viento medio es débil (en el periodo analizado, entre 0,5 y 2,1 m/s de promedio diario según el reanálisis ERA5). El fenómeno de El Niño reduce las lluvias y puede agravar las temporadas críticas, como ocurrió en 2026.

## Ciclo semanal y diario
Las emisiones dependen del tráfico: los domingos y festivos suelen tener menos tráfico pesado y menores concentraciones. En el ciclo diario, las concentraciones suelen ser más altas en la mañana, cuando coinciden la hora pico de tráfico y la atmósfera estable, y bajan en la tarde cuando la capa de mezcla es más profunda.

## Lo que muestran los datos 2023-2024 de este proyecto
- Marzo de 2024 fue el mes más contaminado del periodo (promedio de 30,9 µg/m3 entre estaciones) y concentró la mayoría de días en ICA Naranja; enero, febrero y abril de 2024 también tuvieron días Naranja.
- La segunda temporada de 2023 (octubre-noviembre) fue moderada: casi todos los días estuvieron en Buena o Aceptable.
- Mayo a julio de 2024, con más lluvias, fueron los meses más limpios.
- En el modelo de predicción del día siguiente, las variables más influyentes son la concentración de hoy, su media de 3 días y su tendencia (alta persistencia), y la probabilidad de que el HMM esté en estado "Episodio crítico". La lluvia de hoy y que mañana sea domingo reducen la probabilidad de que mañana sea Dañina. Las variables meteorológicas de mañana tomadas del reanálisis tienen efectos pequeños y no siempre en la dirección esperada (por ejemplo, la lluvia de mañana tiene un coeficiente levemente positivo para la clase Dañina, probablemente porque los días de transición son a la vez lluviosos y contaminados), así que deben interpretarse con cautela.
- El viento medio diario en el punto del reanálisis varió entre 0,5 y 2,1 m/s; la lluvia diaria, entre 0 y 28 mm.

## Advertencia
Las variables meteorológicas usadas por el modelo provienen de un reanálisis global (ERA5) para un único punto del centro del valle; no reemplazan las mediciones locales del SIATA ni sus pronósticos oficiales.
