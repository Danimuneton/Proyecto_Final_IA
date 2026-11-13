# Declaración de uso de IA generativa

**Proyecto:** Agente de calidad del aire del Valle de Aburrá · SI3003 Inteligencia Artificial · EAFIT 2026-1
**Integrantes:** Daniel Muñeton · Tomas Vera

De acuerdo con el pacto pedagógico del curso, declaramos las herramientas de IA generativa usadas, su propósito y el alcance de su participación.

## 1. IA generativa usada para desarrollar el proyecto

| Herramienta | Proveedor / modelo | Propósito | Alcance |
|---|---|---|---|
| **Kiro** (agente de programación en la web) | Modelo Claude (Anthropic) | Revisar el enunciado y la rúbrica; verificar el acceso a las fuentes de datos; proponer la arquitectura; **generar la mayor parte del código** (pipeline de datos, HMM, regresión logística, evaluación, RAG, herramientas, agente y dashboard); redactar el corpus de síntesis, los notebooks, el README, el informe y esta plantilla de declaración; ejecutar pruebas y la evaluación con un LLM local | Alto. El código y los textos fueron producidos por la herramienta a partir de nuestras instrucciones y luego revisados por el equipo (ver §3) |
| _[Otra herramienta, si aplica: ChatGPT, Copilot, Gemini…]_ | | | |

## 2. IA generativa que forma parte del producto

| Componente | Modelo | Uso |
|---|---|---|
| LLM del agente | `openai/gpt-oss-120b` vía Groq (por defecto); `qwen2.5:7b` vía Ollama (evaluación local) | Decide qué herramienta llamar y redacta la respuesta con citas |
| Embeddings del RAG | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Representación vectorial del corpus para la búsqueda en FAISS |

Estos modelos **no** se usan para generar los datos ni las métricas: las cifras provienen de los datos del SIATA y de los modelos de la Parte 1 implementados en el repositorio.

## 3. Qué hizo el equipo (completar con honestidad antes de entregar)

- Elección de la idea 3.1 y definición del alcance: _[describir]_
- Revisión y comprensión del código: _[indicar qué módulos revisó cada integrante; p. ej., Daniel: `hmm.py`, `modeling.py`; Tomas: `agent.py`, `app.py`]_
- Verificación de resultados: _[p. ej., ejecutaron los notebooks, revisaron las métricas y las respuestas del agente]_
- Cambios o decisiones propias: _[listar]_
- Evaluación del agente con Groq y revisión manual de las respuestas: _[describir]_
- Grabación del video y preparación de la presentación: _[describir]_

## 4. Verificación y uso ético

- Todas las cifras del informe y del README se obtienen ejecutando los scripts del repositorio (`scripts/train_models.py`, `scripts/build_index.py`, `scripts/evaluate_agent.py`); no fueron escritas a mano por la IA.
- El HMM y la regresión logística están implementados con NumPy; scikit-learn solo se usa para TF-IDF y para **verificar** que nuestra regresión logística da los mismos resultados.
- Las fuentes del corpus se verificaron (enlaces revisados). Los documentos de síntesis se marcan como tales y no reemplazan los textos oficiales.
- El sistema advierte que no es consejo médico, legal ni de seguridad, ni un pronóstico oficial.
- Se usaron solo datos ambientales abiertos y agregados por estación; no hay datos personales.
- Ninguna llave de API se guarda en el repositorio (`.env` está en `.gitignore`).

## 5. Riesgos reconocidos

La IA generativa puede producir código o textos incorrectos. Por eso el proyecto incluye verificaciones automáticas (comparación del forward contra enumeración exhaustiva, recuperación de parámetros simulados con Baum-Welch, comparación con scikit-learn, evaluación del RAG y del agente con preguntas fijas y un verificador de citas). El equipo es responsable del contenido entregado y debe poder explicarlo.
