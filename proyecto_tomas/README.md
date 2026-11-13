# Agente de calidad del aire del Valle de Aburrá

Proyecto final de **SI3003 – Inteligencia Artificial** (EAFIT, 2026-1). MVP que combina:

- **Parte 1 (IA clásica):** clasificador **Naive Bayes categórico escrito desde cero con NumPy** que predice la categoría de PM2.5 del día siguiente, evaluado con partición temporal frente a dos líneas base.
- **Parte 2 (IA moderna):** **agente LLM** con loop agéntico propio, **RAG híbrido** (embeddings + FAISS + TF-IDF) sobre documentos oficiales y **3 herramientas**: `get_pm25`, `predict_tomorrow`, `search_docs`.
- **Dashboard** en Streamlit: serie de tiempo, mapa y comparación entre estaciones, predicción y chat con el agente.

> **Alcance y advertencias.** Los datos son un histórico de SIATA del **31-jul-2023 al 30-jul-2024**: "mañana" significa el día siguiente a la fecha elegida, no la fecha real de hoy. La herramienta es analítica y orientativa: **no es consejo médico, legal ni de seguridad**.

## Integrantes
Daniel Muñeton · Tomas Vera · Grupo jueves

**Informe (≤ 4 páginas):** [`informe/informe.pdf`](informe/informe.pdf) · **Declaración de uso de IA:** [`DECLARACION_USO_IA.md`](DECLARACION_USO_IA.md)

## Estructura
```
app/app.py              Dashboard Streamlit
src/modelo.py           Naive Bayes propio, métricas, partición temporal, predecir_manana()
src/rag.py              Extracción de PDF, fragmentos, embeddings, FAISS y búsqueda híbrida
src/agente.py           Loop agéntico (T_max = 6), herramientas, prompt
src/evaluar_agente.py   Evaluación con 15 preguntas fijas
src/evaluacion_temporal.py  Origen móvil mensual, verificación vs scikit-learn e IC bootstrap (Parte 1)
src/evaluar_rag.py      Evaluación de la recuperación (hit@k, MRR) con 14 consultas
notebooks/resultados.ipynb  Reproduce todos los resultados sin llave de API
informe/                Informe (≤ 4 páginas) en .md y .pdf
src/clean_data.py       JSON horario -> CSV diario por estación
src/download_data.py    (opcional) descarga el JSON crudo de SIATA
src/probe_data.py, inspect_data.py, analyze_siata.py   exploración inicial de fuentes
data/processed/         CSV limpios (incluidos en el repo)
models/                 Modelo entrenado y métricas (JSON)
docs/corpus/            PDF del RAG (fuentes abajo)
docs/evaluacion_*       Resultados de la evaluación del agente y del RAG
docs/figuras/           Figuras del informe
```

## Reproducir (desde cero)
Requiere Python 3.10+ y una API key de un endpoint compatible con OpenAI (se usó **Groq**, capa gratuita: https://console.groq.com/keys). Este proyecto está en la carpeta `proyecto_tomas/` del repositorio.

**Windows (PowerShell):**
```powershell
git clone https://github.com/Danimuneton/Proyecto_Final_IA.git
cd Proyecto_Final_IA\proyecto_tomas
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env        # luego: notepad .env  y poner la llave en LLM_API_KEY
python src\rag.py             # construye el índice (la 1.ª vez descarga ~470 MB del modelo de embeddings)
streamlit run app\app.py      # abre http://localhost:8501
```

**macOS / Linux:**
```bash
git clone https://github.com/Danimuneton/Proyecto_Final_IA.git && cd Proyecto_Final_IA/proyecto_tomas
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # editar LLM_API_KEY
python src/rag.py && streamlit run app/app.py
```

Pasos opcionales:
```bash
python src/download_data.py && python src/clean_data.py   # regenerar los CSV desde SIATA
python src/modelo.py                                      # reentrenar y re-evaluar el modelo
python src/evaluar_agente.py                              # re-ejecutar la evaluación (≈15 llamadas al LLM, ~5 min)
python src/evaluar_agente.py --rescore                    # recalcular métricas desde docs/evaluacion_trazas.json sin llamar al LLM
python src/evaluacion_temporal.py                         # origen móvil, verificación vs sklearn e IC bootstrap (~5 s)
python src/evaluar_rag.py                                 # evaluación de la recuperación del RAG
```
Si Groq cambia sus modelos, lista los disponibles con `python -c "from dotenv import load_dotenv; import os; load_dotenv(); from openai import OpenAI; c=OpenAI(base_url=os.getenv('LLM_BASE_URL'), api_key=os.getenv('LLM_API_KEY')); print([m.id for m in c.models.list().data])"` y ajusta `LLM_MODEL` en `.env`. El modelo usado fue `openai/gpt-oss-120b` con `LLM_REASONING_EFFORT=low`.

## Datos
- **PM2.5 horario por estación:** SIATA (Sistema de Alerta Temprana de Medellín y el Valle de Aburrá), `https://siata.gov.co/EntregaData1/Datos_SIATA_Aire_pm25.json`, descargado el 12-nov-2026. 21 estaciones; se descartaron 4 sin datos válidos (3, 25, 31, 48) y la 44 (marcada `_OFF`), quedando **16 estaciones y 5.535 ejemplos** de entrenamiento/prueba.
- **Limpieza:** se descartan valores negativos (incluye el centinela -9999) y mayores de 500 µg/m³ (hay centinelas como 99.999.000); el promedio diario exige ≥ 18 de 24 horas válidas (75 %).
- **Uso responsable:** solo datos ambientales agregados y públicos; no hay datos personales.
- **Puntos de corte (PM2.5, 24 h):** Buena ≤ 12, Aceptable 13–37, Dañina a sensibles o peor > 37 µg/m³, según el ICA (Resolución 2254 de 2017; hoja metodológica del IDEAM).

## Corpus del RAG (docs/corpus)
| Archivo | Fuente |
|---|---|
| `ideam_hoja_metodologica_ICA.pdf` | IDEAM, hoja metodológica del Índice de Calidad del Aire (`bart.ideam.gov.co`) |
| `siata_informe_diario_amva_2026-04-24.pdf` | SIATA, informe diario AMVA del 24-abr-2026 |
| `siata_poeca_plantilla_ica.pdf` | SIATA, plantilla de reporte ICA / POECA (1 página) |
| `amva_resolucion_2356_2023.pdf` | Resolución metropolitana 2356 de 2023 (copia alojada en colmayor.edu.co) |
| `laestrella_plan_accion_poeca.pdf` | Plan de acción del protocolo POECA, municipio de La Estrella |

## Resultados

**Modelo (Parte 1).** Partición temporal: entrenamiento hasta 2024-04-29 (4.150 ejemplos), prueba desde 2024-04-30 (1.385). El conjunto de variables se eligió con una validación interna, no con la prueba.

| Modelo | Accuracy | F1 macro |
|---|---|---|
| Naive Bayes (variables `A_basico`) | 0,737 | 0,693 |
| Línea base: mañana = hoy | 0,746 | 0,719 |
| Línea base: clase mayoritaria | 0,653 | 0,395 |

El Naive Bayes **no supera** a "mañana = hoy".

**Evaluación complementaria con origen móvil** (`src/evaluacion_temporal.py`): para cada mes de enero a julio de 2024 se re-entrena con los días anteriores y se predice el mes (3.249 predicciones, 182 días Dañina+). Naive Bayes: exactitud 0,815, F1 macro 0,666, recall Dañina+ 0,49; persistencia: 0,811, 0,692, 0,61. Diferencia NB − persistencia (IC 95 %, bootstrap de días): exactitud +0,004 [−0,007; +0,014], F1 macro −0,026 [−0,058; +0,000]: no hay diferencia significativa. La implementación coincide con `sklearn.naive_bayes.CategoricalNB` en el 100 % de las predicciones (diferencia máxima de probabilidad 1e-15).

**RAG** (`src/evaluar_rag.py`, 14 consultas con la página correcta identificada a mano): híbrido hit@1 0,71, hit@6 0,93, MRR 0,80; solo embeddings 0,50 / 0,93 / 0,69; solo TF-IDF 0,57 / 1,00 / 0,71. En la prueba no hay días "Dañina a sensibles o peor" (0 de 1.385; en validación hubo 170), por lo que esa clase no se puede evaluar: es un desplazamiento de distribución (los episodios críticos ocurren sobre todo entre febrero y abril).

**Agente (Parte 2).** Última corrida sobre 15 preguntas fijas (`docs/evaluacion_agente.md`):
- Herramienta correcta: 14/15 (la falla es la pregunta 15: respondió que no hay datos de 2026 usando el rango de fechas de su prompt, sin llamar a la herramienta).
- Cita de fuente: 11/12 (excluye la pregunta fuera de tema y los errores de herramienta).
- Cita verificable (archivo y página estaban entre los fragmentos recuperados): 7/7 preguntas documentales.
- Frases entre comillas presentes en los fragmentos: 5 sí, 2 parciales, 0 sin comillas (de 7).
- **Revisión manual de las 7 respuestas documentales (6 a 12):** 2 correctas y fundamentadas (8, 9); 2 correctas con alguna afirmación sin frase de respaldo (6, 10); 3 con errores de razonamiento (7: equipara colores del ICA con niveles de contingencia; 11: dice que no hay recomendaciones para asma y pone entre «» un texto propio; 12: dice que no hay efectos para niños, aunque la hoja del IDEAM p. 4 menciona síntomas en grupos sensibles).

## Limitaciones
- Datos históricos de un solo año (hasta jul-2024); no hay predicción en tiempo real.
- La limpieza no usa la bandera de calidad de SIATA: el 2,3 % de las horas conservadas (3.082) tienen bandera > 2,5, que SIATA no considera de buena calidad.
- El modelo no supera la línea base "mañana = hoy"; usa solo PM2.5 pasado y calendario (sin lluvia ni viento).
- La clase "Dañina+" no se puede evaluar en la prueba; la variable de temporada lluviosa es un supuesto nuestro.
- Las métricas automáticas del agente **no detectan razonamientos errados con citas válidas**. Ejemplo: en la pregunta 7 el agente equiparó los colores del ICA con los "niveles de contingencia" sin que ningún fragmento lo diga; en la 11 puso entre «» un texto que no está en los fragmentos.
- El prompt del agente se ajustó mirando las mismas 15 preguntas de la evaluación, por lo que los resultados son optimistas.
- El texto de algunos PDF se extrae con espacios cortados o tildes mal codificadas ("e n", "Dańina").

## Declaración de uso de IA generativa
Ver [`DECLARACION_USO_IA.md`](DECLARACION_USO_IA.md).
