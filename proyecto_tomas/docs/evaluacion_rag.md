# Evaluación de la recuperación (RAG)

14 consultas con la página correcta identificada a mano. Acierto = el fragmento recuperado es de esa página.

| Recuperador | hit@1 | hit@6 | MRR |
|---|---|---|---|
| híbrido (agente) | 0.71 | 0.93 | 0.80 |
| solo embeddings + FAISS | 0.50 | 0.93 | 0.69 |
| solo TF-IDF | 0.57 | 1.00 | 0.71 |

| Consulta | Página correcta | Posición (híbrido) | Embeddings | TF-IDF |
|---|---|---|---|---|
| ¿Cuáles son los puntos de corte del ICA para PM2.5? | ideam_hoja_metodologica_ICA.pdf p.5 | 3 | 6 | 5 |
| ¿Qué efectos en la salud tiene la categoría naranja del ICA? | ideam_hoja_metodologica_ICA.pdf p.4 | 13 | 3 | 3 |
| ¿Cómo se calcula el Índice de Calidad del Aire? | ideam_hoja_metodologica_ICA.pdf p.5, ideam_hoja_metodologica_ICA.pdf p.6 | 1 | 2 | 2 |
| ¿En qué meses se declaran los periodos de gestión de episodios de contaminación? | amva_resolucion_2356_2023.pdf p.5, laestrella_plan_accion_poeca.pdf p.14 | 1 | 1 | 6 |
| ¿En qué contaminantes se enfoca el POECA? | amva_resolucion_2356_2023.pdf p.3 | 3 | 1 | 4 |
| ¿Qué información entrega el Comité de Comunicaciones Metropolitano? | amva_resolucion_2356_2023.pdf p.10 | 1 | 1 | 1 |
| Recomendaciones para establecimientos educativos en alerta roja ambiental | laestrella_plan_accion_poeca.pdf p.59 | 1 | 2 | 1 |
| Signos y síntomas respiratorios de alarma en los niños | laestrella_plan_accion_poeca.pdf p.94 | 1 | 1 | 1 |
| Operativos de vigilancia de quemas e incendios durante los episodios | laestrella_plan_accion_poeca.pdf p.73 | 1 | 2 | 2 |
| ¿Qué efecto tuvo la medida de pico y placa en la hora pico de contaminantes? | laestrella_plan_accion_poeca.pdf p.18 | 1 | 9 | 1 |
| ¿Cuántas estaciones poblacionales presentan ICA verde y amarillo hoy? | siata_informe_diario_amva_2026-04-24.pdf p.1 | 2 | 1 | 1 |
| ¿Qué se hace si alguna estación presenta ICA rojo o naranja? | siata_poeca_plantilla_ica.pdf p.1 | 1 | 2 | 1 |
| ¿Qué es un episodio o evento de contaminación atmosférica? | amva_resolucion_2356_2023.pdf p.8 | 1 | 1 | 1 |
| ¿Quién es la máxima autoridad ambiental en el perímetro urbano del valle? | amva_resolucion_2356_2023.pdf p.2 | 1 | 1 | 1 |
