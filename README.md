# 🌫️ Agente de calidad del aire del Valle de Aburrá

**Proyecto final integrador · SI3003 Inteligencia Artificial · Universidad EAFIT · 2026-1**
Profesor: Jorge Iván Padilla Buriticá

**Integrantes:** Daniel Muñeton · _[NOMBRE DEL COMPAÑERO]_ · **Grupo:** _[N.º]_

> **Problema.** Los ciudadanos no comprenden qué significa el PM2.5 del día ni qué hacer al respecto.
> **Solución.** Un dashboard con datos reales del SIATA que (1) estima el **estado latente** de la calidad del aire con un **Modelo Oculto de Markov** propio, (2) **predice la categoría ICA de mañana** con una **regresión logística** propia que usa el estado del HMM, y (3) ofrece un **agente LLM con RAG** que consulta datos, predice y explica la norma, los protocolos de episodios (POECA) y las recomendaciones de salud **citando las fuentes**.

⚠️ Herramienta académica. No es un pronóstico oficial (ver [SIATA](https://siata.gov.co)) ni consejo médico, legal o de seguridad.

---

## Contenido

1. [Cumplimiento de los requisitos y la rúbrica](#1-cumplimiento-de-los-requisitos-y-la-rúbrica)
2. [Instalación y ejecución (Windows / PowerShell)](#2-instalación-y-ejecución-windows--powershell)
3. [Reproducir todos los resultados](#3-reproducir-todos-los-resultados)
4. [Datos y uso responsable](#4-datos-y-uso-responsable)
5. [Arquitectura](#5-arquitectura)
6. [Resultados](#6-resultados)
7. [Limitaciones](#7-limitaciones)
8. [Estructura del repositorio](#8-estructura-del-repositorio)
9. [Entregables](#9-entregables)

---

## 1. Cumplimiento de los requisitos y la rúbrica

| Requisito del enunciado | Dónde se cumple |
|---|---|
| 1. Datos reales, fuente identificada y citada, plan B | SIATA `Datos_SIATA_Aire_pm25.json` (PM2.5 horario de 21 estaciones, 31-jul-2023 a 31-jul-2024) + Open-Meteo/ERA5. Plan B: CSV procesados versionados en `data/processed/`. Ver [§4](#4-datos-y-uso-responsable) |
| 2. Modelo de la Parte 1 implementado por el equipo, con métrica frente a línea base | **HMM gaussiano** (forward-backward, Baum-Welch, Viterbi) en [`src/aire/hmm.py`](src/aire/hmm.py) y **regresión logística multinomial** (Adam + L2) en [`src/aire/classifier.py`](src/aire/classifier.py), solo con NumPy. Comparación con 3 líneas base (mayoritaria, persistencia, cadena de Markov) en [`reports/metricas_parte1.md`](reports/metricas_parte1.md) |
| 3. Agente LLM con RAG y ≥ 2 herramientas | Loop agéntico manual con `T_MAX = 6` en [`src/aire/agent.py`](src/aire/agent.py); 4 herramientas: `get_pm25`, `predict_tomorrow`, `get_episode_state`, `search_docs` ([`src/aire/tools.py`](src/aire/tools.py)); RAG híbrido FAISS + TF-IDF sobre 7 documentos ([`docs/corpus/`](docs/corpus)) |
| 4. ≥ 2 visualizaciones significativas | Mapa de estaciones por color ICA, serie de tiempo con bandas ICA y estado HMM, probabilidades de estados, perfil horario, probabilidades de mañana, backtesting, matriz de confusión y matriz de transición |
| 5. Interacción: seleccionar parámetros y conversar | Barra lateral: municipio, estación, fecha, ventana, **escenario meteorológico de mañana** y domingo/festivo. Pestaña de chat con el agente |
| 6. Funcionalidad reproducible siguiendo el README | [§2](#2-instalación-y-ejecución-windows--powershell). Los modelos entrenados y los datos procesados vienen en el repo: la app corre sin reentrenar |
| 7. Repositorio con código, notebooks, README, requirements y datos | [`notebooks/`](notebooks), [`src/aire/`](src/aire), [`scripts/`](scripts), [`requirements.txt`](requirements.txt), [`data/`](data) |
| 8. Declaración de uso de IA generativa | [`DECLARACION_USO_IA.md`](DECLARACION_USO_IA.md) |
| Entregable: informe ≤ 4 páginas | [`informe/informe.md`](informe/informe.md) (y [`informe/informe.pdf`](informe/informe.pdf)) |
| Entregable: evaluación del agente con 10–15 preguntas | 14 preguntas fijas en [`evaluacion/preguntas_agente.json`](evaluacion/preguntas_agente.json); script [`scripts/evaluate_agent.py`](scripts/evaluate_agent.py); resultados en [`evaluacion/`](evaluacion) |

| Criterio de la rúbrica | Evidencia |
|---|---|
| Técnica Parte 1 (20 %) | Implementación propia verificada: forward = enumeración exhaustiva, Baum-Welch recupera parámetros simulados y su log-verosimilitud es monótona; la regresión logística coincide con scikit-learn en 100 % de las predicciones ([notebook 02](notebooks/02_modelos_parte1_hmm_logreg.ipynb)) |
| Agente LLM (20 %) | Loop con `T_MAX`, 4 herramientas, errores de herramienta devueltos como observación, errores de API con mensajes claros, **verificador de citas** que rechaza citas a documentos no recuperados |
| Dashboard (10 %) | `app.py` (Streamlit + Plotly) con 5 pestañas |
| Interacción y funcionalidad (15 %) | Parámetros + escenarios + chat; probado de punta a punta con `streamlit.testing` y un LLM real |
| Rigor experimental (15 %) | Validación temporal con origen móvil, 3 líneas base, ablaciones, IC 95 % por bootstrap de días, evaluación del RAG (hit@k, MRR) y del agente (herramienta correcta, citas, validez de citas) |
| Reproducibilidad (10 %) | Scripts en orden, semillas fijas, dependencias, datos procesados versionados, uso responsable de datos |
| Declaración ética de IA (5 %) | [`DECLARACION_USO_IA.md`](DECLARACION_USO_IA.md) |

---

## 2. Instalación y ejecución (Windows / PowerShell)

**Requisitos:** Python 3.10, 3.11 o 3.12 ([python.org](https://www.python.org/downloads/), marque *Add Python to PATH*), Git y conexión a internet la primera vez (se descargan las librerías y el modelo de embeddings, ~0,5 GB).

```powershell
# 1) Clonar
git clone https://github.com/Danimuneton/Proyecto_Final_IA.git
cd Proyecto_Final_IA

# 2) Entorno virtual e instalación (5-10 minutos la primera vez)
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3) Llave gratuita del LLM (Groq): crear en https://console.groq.com/keys
copy .env.example .env
notepad .env        # reemplace gsk_su_llave_aqui por su llave y guarde

# 4) Ejecutar el dashboard (se abre en http://localhost:8501)
streamlit run app.py
```

Si PowerShell bloquea la activación del entorno, ejecute una vez `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` y repita `venv\Scripts\activate`.

<details>
<summary>macOS / Linux</summary>

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # editar la llave
streamlit run app.py
```
</details>

**Proveedores de LLM alternativos** (todos con API compatible con OpenAI, se configuran en `.env`):

| Proveedor | Variables en `.env` | Modelo por defecto |
|---|---|---|
| Groq (por defecto, capa gratuita) | `LLM_PROVIDER=groq`, `GROQ_API_KEY=...` | `openai/gpt-oss-120b` |
| NVIDIA NIM | `LLM_PROVIDER=nvidia`, `NVIDIA_API_KEY=...` | `meta/llama-3.3-70b-instruct` |
| Ollama local, sin llave | `LLM_PROVIDER=ollama` (antes: `ollama pull qwen2.5:7b`) | `qwen2.5:7b` |

Cualquier modelo se cambia con `LLM_MODEL=...`. Sin llave, el dashboard funciona completo (mapa, series, predicción, evaluación) y el chat muestra un mensaje claro de configuración.

**Uso sugerido para la demostración:** fecha de consulta 2024-03-15 (en plena temporada crítica de marzo de 2024), estación MED-BEME (Belén). Pruebe las preguntas sugeridas del chat, cambie el municipio y mueva el escenario de mañana.

---

## 3. Reproducir todos los resultados

Los artefactos ya están en el repositorio; estos comandos los regeneran desde cero (con el entorno activado):

```powershell
python scripts\download_data.py       # 1) Descarga SIATA + Open-Meteo y genera data\processed (plan B: CSV ya incluidos)
python scripts\train_models.py        # 2) Evalúa HMM + regresión logística (origen móvil) y entrena los modelos finales (~1 min)
python scripts\build_index.py         # 3) Construye el índice FAISS y evalúa la recuperación del RAG
python scripts\evaluate_agent.py      # 4) Evalúa el agente con las 14 preguntas (requiere llave del LLM, ~3 min con Groq)
jupyter notebook notebooks            # 5) Notebooks 01 (EDA), 02 (Parte 1) y 03 (RAG/agente)
```

Para evaluar con otro modelo sin tocar `.env`: `$env:LLM_PROVIDER="ollama"; $env:LLM_MODEL="qwen2.5:7b"; python scripts\evaluate_agent.py`.
Cada corrida guarda `evaluacion\resultados_agente_<modelo>.md/.csv`.

---

## 4. Datos y uso responsable

| Fuente | Contenido | Acceso |
|---|---|---|
| **SIATA** (AMVA, Alcaldía de Medellín, EPM) | PM2.5 horario por estación con bandera de calidad, 21 estaciones (17 con datos válidos), 31-jul-2023 a 31-jul-2024 | [Datos_SIATA_Aire_pm25.json](https://siata.gov.co/EntregaData1/Datos_SIATA_Aire_pm25.json) (abierto) |
| **Open-Meteo Historical Weather API** (reanálisis ERA5, Copernicus/ECMWF) | Lluvia, viento, radiación, nubosidad, temperatura y humedad diarias en el centro del valle (6,25 N; −75,57 O) | [open-meteo.com](https://open-meteo.com/en/docs/historical-weather-api), licencia CC BY 4.0 |
| **Corpus RAG** | Res. 2254/2017 (extracto del texto oficial), POECA/AMVA, recomendaciones de salud por ICA (Res. 2254 + US EPA AirNow), guías OMS 2021, SIATA, meteorología del valle, metodología del proyecto | [`docs/corpus/`](docs/corpus); cada archivo indica su fuente y URL |

**Procesamiento:** se descartan las horas con bandera de calidad SIATA > 2,5 o valor negativo; el promedio de 24 h exige ≥ 18 horas válidas (75 %); ICA según los puntos de corte de la Tabla 6 de la Res. 2254/2017. Resultado: 5.870 promedios diarios estación-día.

**Verificación de acceso y plan B:** el acceso a las fuentes se verificó al inicio. Si SIATA u Open-Meteo no responden, el proyecto usa los CSV versionados en `data/processed/` (no se requiere descarga para correr la app).

**Uso responsable:** solo datos ambientales agregados por estación, sin datos personales. Los resúmenes del corpus elaborados por el equipo se marcan como tales y citan la fuente original. Los textos normativos son públicos. La app advierte que no es pronóstico oficial ni consejo médico y remite al SIATA y al AMVA.

---

## 5. Arquitectura

```mermaid
flowchart LR
    A[SIATA JSON<br/>PM2.5 horario] --> P[data.py<br/>limpieza y promedios 24 h]
    B[Open-Meteo ERA5] --> P
    P --> H[HMM propio<br/>Baum-Welch / forward / Viterbi]
    H -->|P(estado hoy)| L[Regresión logística propia<br/>clase ICA de mañana]
    P --> L
    D[docs/corpus<br/>Res. 2254, POECA, salud, OMS...] --> R[RAG híbrido<br/>MiniLM multilingüe + FAISS + TF-IDF]
    H & L & P --> T[Herramientas<br/>get_pm25 · predict_tomorrow<br/>get_episode_state · search_docs]
    R --> T
    T <--> G[Agente LLM<br/>loop T_max=6 + verificador de citas]
    G --> UI[Dashboard Streamlit<br/>mapa · series · predicción · evaluación · chat]
    H & L & P --> UI
```

**Integración no trivial Parte 1 ↔ Parte 2:** el HMM entrega las probabilidades filtradas del estado latente, que son variables del clasificador; ambos modelos son herramientas que el agente invoca, y el agente combina sus salidas con los documentos recuperados (p. ej., "¿puedo trotar mañana si tengo asma?" → `predict_tomorrow` + `search_docs` → respuesta con citas).

---

## 6. Resultados

### Parte 1 — predicción de la categoría ICA del día siguiente
Validación con **origen móvil mensual** (ene–jul 2024; para cada mes se re-entrena solo con datos anteriores), 3.309 predicciones estación-día. Clases: Buena (0–12 µg/m³), Aceptable (13–37), Dañina para grupos sensibles o peor (≥ 38, nivel de Prevención).

| Modelo | Exactitud | Exact. balanceada | F1 macro | Recall Dañina | Log-loss | Brier |
|---|---|---|---|---|---|---|
| Base: clase mayoritaria | 0.757 | 0.333 | 0.287 | 0.000 | — | — |
| Base: persistencia (mañana = hoy) | 0.799 | 0.685 | 0.684 | 0.601 | — | — |
| Base: cadena de Markov de clases | 0.796 | 0.654 | 0.664 | 0.506 | 0.536 | 0.300 |
| HMM propio (predicción a un paso) | 0.758 | 0.339 | 0.300 | 0.000 | 0.522 | 0.313 |
| RL propia: solo información de hoy | 0.795 | 0.669 | 0.661 | 0.565 | 0.448 | 0.275 |
| RL propia: hoy + domingo (sin meteorología) | 0.813 | 0.674 | 0.684 | 0.524 | **0.434** | **0.262** |
| RL propia: completa sin HMM | **0.811** | 0.681 | **0.686** | 0.571 | 0.445 | 0.267 |
| **RL propia: completa + HMM (modelo final)** | 0.808 | **0.703** | **0.686** | **0.631** | 0.439 | 0.265 |

- Frente a la cadena de Markov, el modelo final mejora la log-loss en −0,098 (IC 95 % [−0,133; −0,064]) y el Brier en −0,036 (IC [−0,051; −0,021]).
- Frente a la persistencia, las mejoras en exactitud (+0,009) y F1 macro (+0,002) **no son significativas** (los IC incluyen 0): el PM2.5 cambia lentamente y la persistencia es una línea base muy fuerte. El modelo sí entrega probabilidades calibradas y mayor recall de la clase Dañina.
- Las variables del HMM mejoran la exactitud balanceada (+0,022, IC [+0,001; +0,053]) y la log-loss (−0,005, IC [−0,009; −0,002]).
- El HMM describe la serie mucho mejor que una gaussiana sin dinámica (log-verosimilitud predictiva media −0,20 vs −0,71 por observación). Estados: Normal ≈ 12, Alerta ≈ 18, Episodio crítico ≈ 30 µg/m³, con permanencia diaria ≈ 0,94.

Detalle completo: [`reports/metricas_parte1.md`](reports/metricas_parte1.md) · Figuras: [`reports/figures/`](reports/figures).

### Parte 2 — RAG y agente
**Recuperación** (18 consultas con documento esperado, [`evaluacion/rag_consultas.json`](evaluacion/rag_consultas.json)):

| Recuperador | hit@1 | hit@4 | MRR |
|---|---|---|---|
| **Híbrido FAISS + TF-IDF (RRF, usado por el agente)** | 0.72 | **1.00** | 0.84 |
| Solo embeddings + FAISS | 0.78 | 0.94 | 0.85 |
| Solo TF-IDF | 0.72 | 0.94 | 0.80 |

**Agente** (14 preguntas fijas: datos, predicción, escenario, episodio, documentales, combinadas, errores de estación y de fecha, y una fuera de tema). Resultados por modelo en [`evaluacion/`](evaluacion):

| LLM | Herramienta correcta | Citó fuente (11 que lo requieren) | Citas válidas | Valor numérico correcto |
|---|---|---|---|---|
| `qwen2.5:7b` local (Ollama, CPU) | 11/14 | 11/11 | 10/11 | 3/3 |
| `openai/gpt-oss-120b` (Groq) | ver `evaluacion/resultados_agente_openai_gpt-oss-120b.md` | | | |

---

## 7. Limitaciones

- **Un solo año de datos** (el archivo abierto de SIATA cubre jul-2023 a jul-2024): la app trabaja con una "fecha de consulta" histórica y no informa la calidad del aire actual.
- **Pocos días Dañinos** (172 de 5.870 estación-días) y ninguno peor que Rojo: no se ha probado en episodios severos.
- La **meteorología de mañana** proviene de un reanálisis global en un solo punto y se usa como "pronóstico perfecto"; sus coeficientes son pequeños y algunos tienen signos poco intuitivos (confusión con la estacionalidad). Con un pronóstico real el desempeño sería menor.
- Los **estados del HMM** son regímenes estadísticos (el más alto tiene media ≈ 30 µg/m³), no declaratorias oficiales del AMVA.
- El HMM final se entrenó con todo el año, por lo que en la demo el filtrado de fechas pasadas usa parámetros que "vieron" el futuro (la evaluación de §6 sí es estrictamente temporal).
- El corpus es pequeño (7 documentos, 56 fragmentos) y las consultas de evaluación del RAG fueron escritas por el equipo.
- El agente depende del LLM: con modelos pequeños omite herramientas o inventa citas; el verificador de citas mitiga, pero no elimina, este riesgo.
- **No es consejo médico, legal ni de seguridad**, ni reemplaza al SIATA o al AMVA.

---

## 8. Estructura del repositorio

```
├── app.py                        # Dashboard Streamlit
├── src/aire/
│   ├── config.py                 # rutas, ICA (Res. 2254), constantes
│   ├── data.py                   # descarga y limpieza SIATA / Open-Meteo
│   ├── ica.py                    # cálculo del ICA y clases
│   ├── hmm.py                    # HMM gaussiano propio (Baum-Welch, forward, Viterbi)
│   ├── classifier.py             # regresión logística multinomial propia + métricas
│   ├── modeling.py               # variables, origen móvil, líneas base, entrenamiento
│   ├── service.py                # consultas usadas por herramientas y dashboard
│   ├── rag.py                    # corpus, embeddings, FAISS, TF-IDF, RRF
│   ├── tools.py                  # esquemas y despachador de herramientas
│   └── agent.py                  # loop agéntico, verificador de citas, proveedores LLM
├── scripts/                      # download_data, train_models, build_index, evaluate_agent
├── notebooks/                    # 01 EDA · 02 Parte 1 · 03 RAG y agente (ejecutados)
├── data/processed/               # datos procesados (plan B)
├── docs/corpus/                  # corpus del RAG con fuentes
├── models/                       # HMM, clasificador e índice FAISS entrenados
├── reports/                      # métricas, backtesting y figuras
├── evaluacion/                   # preguntas y resultados del agente y del RAG
├── informe/                      # informe (≤ 4 páginas)
└── DECLARACION_USO_IA.md
```

---

## 9. Entregables

1. Repositorio: este repositorio.
2. Informe breve: [`informe/informe.md`](informe/informe.md) / [`informe/informe.pdf`](informe/informe.pdf).
3. Demostración: `streamlit run app.py` (video de respaldo: _[ENLACE AL VIDEO]_).
4. Evaluación del agente: [`evaluacion/`](evaluacion).
5. Declaración de uso de IA generativa: [`DECLARACION_USO_IA.md`](DECLARACION_USO_IA.md).
