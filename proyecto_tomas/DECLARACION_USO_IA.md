# Declaración de uso de IA generativa

**Proyecto:** Agente de calidad del aire del Valle de Aburrá (versión en `proyecto_tomas/`) · SI3003 Inteligencia Artificial · EAFIT 2026-1
**Integrantes:** Daniel Muñeton · Tomas Vera · Grupo jueves

## 1. Herramientas usadas para desarrollar el proyecto

| Herramienta | Propósito | Alcance |
|---|---|---|
| **Claude (Anthropic)**, asistente de desarrollo | Diseño y escritura de código: limpieza de datos, Naive Bayes, RAG, agente, dashboard, evaluación del agente y README | _[COMPLETAR: qué partes generó y cuáles escribieron o modificaron ustedes]_ |
| **Kiro** (agente de programación, modelo Claude de Anthropic) | Revisión final contra la rúbrica: ejecutó el proyecto de punta a punta; agregó `src/evaluacion_temporal.py` (origen móvil, verificación contra scikit-learn, bootstrap), `src/evaluar_rag.py` (evaluación de la recuperación) y `notebooks/resultados.ipynb`; envió al agente la estación y fecha del dashboard; evitó reintentos inútiles ante una llave inválida; hizo la revisión manual de las 7 respuestas documentales; completó el README y redactó el informe | Alto en esas piezas; las cifras salen de ejecutar los scripts del repositorio |

## 2. IA generativa que forma parte del producto

| Componente | Modelo | Uso |
|---|---|---|
| LLM del agente | `openai/gpt-oss-120b` vía Groq (`LLM_REASONING_EFFORT=low`) | Elige herramientas y redacta las respuestas con citas |
| Embeddings del RAG | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Búsqueda semántica en FAISS |

Ninguno de estos modelos genera datos ni métricas: las cifras vienen de SIATA y del Naive Bayes implementado con NumPy.

## 3. Comprensión y verificación por parte del equipo (completar con honestidad)

- Qué ejecutaron ustedes mismos: _[p. ej., `python src/modelo.py`, `python src/evaluar_agente.py` con Groq, el dashboard]_
- Qué revisaron a mano: _[p. ej., las citas del agente contra los PDF; revisen también la revisión manual del informe, sección 5]_
- Errores que detectaron y corrigieron: _[listar]_
- Reparto del trabajo: _[Daniel: … · Tomas: …]_

## 4. Uso ético

- El Naive Bayes y las métricas están implementados con NumPy; scikit-learn solo se usa para TF-IDF y para verificar el Naive Bayes.
- Solo se usan datos ambientales públicos y agregados por estación; no hay datos personales.
- La aplicación advierte que no es consejo médico, legal ni de seguridad, ni un pronóstico oficial.
- La llave del LLM va en `.env`, que no se sube al repositorio.
- El equipo es responsable del contenido entregado y debe poder explicarlo.
