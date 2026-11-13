---
id: SIATA
titulo: SIATA y la red de monitoreo de calidad del aire del Valle de Aburrá - datos usados en este proyecto
fuente: Sistema de Alerta Temprana de Medellín y el Valle de Aburrá (SIATA), proyecto del AMVA, la Alcaldía de Medellín y EPM. Datos abiertos de PM2.5 (archivo Datos_SIATA_Aire_pm25.json). Descripción del procesamiento elaborada por el equipo.
url: https://siata.gov.co/siata_nuevo/
datos: https://siata.gov.co/EntregaData1/Datos_SIATA_Aire_pm25.json
---

# SIATA y la red de monitoreo de PM2.5

## Qué es el SIATA
El SIATA (Sistema de Alerta Temprana de Medellín y el Valle de Aburrá) es un proyecto de ciencia y tecnología del Área Metropolitana del Valle de Aburrá, la Alcaldía de Medellín y EPM. Opera redes de monitoreo hidrometeorológico y de calidad del aire, produce pronósticos y publica en tiempo real el Índice de Calidad del Aire (ICA) por estación. La red de monitoreo de calidad del aire del valle está acreditada por el IDEAM para producir información ambiental oficial, y sus datos son la base para que el AMVA declare los niveles del POECA.

## Datos usados en este proyecto
Se usa el archivo abierto de PM2.5 del SIATA, con mediciones horarias por estación entre el 31 de julio de 2023 y el 31 de julio de 2024. El archivo contiene 21 estaciones con su código, nombre, nombre corto, latitud y longitud; 17 tienen datos válidos en el periodo (cuatro de las marcadas como "_OFF" no reportaron ningún dato válido y la estación MED-LAYE tiene datos parciales). Cada dato horario incluye una bandera de calidad: SIATA indica que solo los datos con bandera menor o igual a 2,5 son de buena calidad; las demás banderas o el valor -9999 indican dato faltante o no validado.

## Procesamiento y control de calidad
1. Se descartan las horas con bandera de calidad mayor a 2,5 o con valor negativo.
2. El promedio diario de 24 horas se calcula solo si hay al menos 18 horas válidas (75 % del día), criterio de representatividad usual en la normatividad colombiana de monitoreo.
3. Se trabaja con 17 estaciones que tienen al menos 30 días válidos; en total 5.870 promedios diarios estación-día.
4. La categoría ICA de cada día se calcula con los puntos de corte de la Resolución 2254 de 2017 (redondeando la concentración al entero más cercano).

## Resumen de los datos del periodo 2023-2024
- PM2.5 diario medio de todas las estaciones: alrededor de 18,7 µg/m3 (mediana 17 µg/m3).
- Distribución de categorías diarias estación-día: Buena (0-12) 1.158 días; Aceptable (13-37) 4.540 días; Dañina para grupos sensibles (38-55) 171 días; Dañina (56-150) 1 día.
- Los días en categoría Naranja o peor se concentraron entre enero y abril de 2024, con el máximo en marzo de 2024 (promedio mensual de 30,9 µg/m3 y 91 estación-días en Naranja o peor), coincidiendo con la primera temporada crítica del año. En octubre a diciembre de 2023 solo hubo 2 estación-días en Naranja.
- El valor diario más alto observado fue 59 µg/m3 (marzo de 2024).
- Los meses más limpios fueron mayo a julio de 2024, con promedios mensuales cercanos a 14 µg/m3.

## Estaciones con datos válidos (código - nombre corto - municipio)
12 CEN-TRAF Estación Tráfico Centro (Medellín); 28 ITA-CJUS Casa de Justicia (Itagüí); 38 ITA-CONC I.E. Concejo Municipal (Itagüí); 44 MED-LAYE Tanques La Ye EPM, El Poblado (Medellín, datos parciales); 69 CAL-JOAR E.U. Joaquín Aristizábal (Caldas); 78 EST-HOSP Hospital (La Estrella); 79 MED-ALTA Altavista (Medellín); 80 MED-VILL Villahermosa (Medellín); 81 BAR-TORR Torre Social (Barbosa); 82 COP-CVID Ciudadela Educativa La Vida (Copacabana); 83 MED-BEME Belén (Medellín); 84 MED-TESO I.E. INEM sede Santa Catalina, El Poblado (Medellín); 85 MED-SCRI San Cristóbal (Medellín); 86 MED-ARAN Aranjuez (Medellín); 87 BEL-FEVE I.E. Fernando Vélez (Bello); 88 ENV-HOSP E.S.E. Santa Gertrudis (Envigado); 90 SAB-RAME I.E. Rafael J. Mejía (Sabaneta).

## Limitaciones de los datos
- El archivo abierto cubre un año (2023-2024); no incluye datos posteriores, por lo que la aplicación trabaja con una "fecha de consulta" dentro de ese periodo y no informa la calidad del aire actual.
- Las estaciones miden en un punto: su valor representa el entorno de la estación, no todo el municipio.
- Los datos horarios publicados en tiempo real pueden ser preliminares; la validación final la hace la autoridad.
- Para la calidad del aire del día de hoy se debe consultar directamente el SIATA (siata.gov.co) o los canales del AMVA.
