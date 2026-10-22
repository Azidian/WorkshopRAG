# Workshop RAG · SI3003 Inteligencia Artificial

## Información de la estudiante

- **Nombre:** Isabella Cadavid Posada
- **Código estudiantil:** 1015070258
- **Usuario de GitHub:** [IsabellaCada2506](https://github.com/IsabellaCada2506)
- **Fecha:** 21 de octubre de 2026
- **Entrega a:** [jipadillab@eafit.edu.co](mailto:jipadillab@eafit.edu.co)
- **Curso:** Inteligencia Artificial — Universidad EAFIT

---

## Descripción

Este repositorio reúne el trabajo realizado para el **Workshop RAG** de SI3003. El taller aborda los fundamentos de la búsqueda aproximada por vecinos cercanos (ANN), los índices HNSW, la recuperación vectorial y la construcción de sistemas **Retrieval-Augmented Generation (RAG)**.

La parte práctica integra **Python, Google Colab, LangChain, NVIDIA NIM, Qdrant, FAISS y Gradio**. Se desarrolló una aplicación RAG empresarial y se evaluó un segundo sistema de preguntas sobre artículos académicos, prestando especial atención a la calidad del contexto recuperado, las fuentes y las respuestas.

## Organización de la entrega

- **`RESPUESTAS.md`**: desarrollo teórico, resultados, comparaciones, análisis y reflexiones de las tareas.
- **Notebooks de Google Colab**: código de la aplicación empresarial (02b) y del RAG de artículos (03), con sus celdas de experimentos y pruebas. La ubicación exacta depende de la organización del repositorio.
- **`evidencias/`**: capturas de pruebas, cuando estén disponibles y subidas al repositorio.

> Los nombres de los notebooks y los archivos de evidencia deben corresponder a los archivos realmente subidos a GitHub. Este README no presupone que todas las capturas estén presentes.

---

## Progreso de actividades

**Leyenda:** ✅ documentada con resultados · ◐ desarrollo descrito, pero falta verificar alguna evidencia o medición · — sin evidencia suficiente para confirmar.

| ID | Tarea | Puntos | Estado | Evidencia / estado de verificación |
|---|---|---:|:---:|---|
| T1 | Cuántos nodos viven en cada capa | 2 | ✅ | [RESPUESTAS.md](RESPUESTAS.md), cálculo teórico y simulación |
| T2 | `ef`, `k` y el tamaño de la cola | 2 | ✅ | [RESPUESTAS.md](RESPUESTAS.md), explicación de parámetros |
| T3 | Inserción frente a búsqueda | 2 | ✅ | [RESPUESTAS.md](RESPUESTAS.md), tabla comparativa y análisis |
| T4 | El invariante de `FAISSVectorStore` | 2 | ✅ | [RESPUESTAS.md](RESPUESTAS.md), mapeo de posiciones y errores |
| T5 | El costo de la fuerza bruta | 2 | ✅ | [RESPUESTAS.md](RESPUESTAS.md), estimaciones computacionales |
| A1 | Ejecuta la app RAG de punta a punta | 4 | ✅ | Notebook 02b y [RESPUESTAS.md](RESPUESTAS.md), cinco consultas |
| A2 | Tu propio corpus de documentos | 6 | ✅ | Notebook 02b y [RESPUESTAS.md](RESPUESTAS.md), consultas al corpus |
| A3 | Ingeniería del prompt (antes y después) | 5 | ✅ | [RESPUESTAS.md](RESPUESTAS.md), diff y comparación |
| A4 | Muestra el score de similitud en las fuentes | 5 | ✅ | [RESPUESTAS.md](RESPUESTAS.md) |
| B1 | Ejecuta el RAG de papers y evalúa con ground truth | 6 | ✅ | Notebook 03 y [RESPUESTAS.md](RESPUESTAS.md), Q1–Q5 |
| B2 | Experimentos con `chunk_size`, `k` y `score_threshold` | 8 | ✅ | Notebook 03 y [RESPUESTAS.md](RESPUESTAS.md), nueve configuraciones |
| B3 | Preguntas trampa: ¿el RAG admite que no sabe? | 6 | ✅ | [RESPUESTAS.md](RESPUESTAS.md), tres preguntas trampa |
| B4 | Preguntas en español sobre papers en inglés | 5 | ✅ | [RESPUESTAS.md](RESPUESTAS.md), tres pares ES/EN |
| B5 | Escribe tu propio quiz con ground truth | 5 | ✅ | Notebook 03 y [RESPUESTAS.md](RESPUESTAS.md), Q6–Q8 |
| R1 | Reflexión final con tus datos | 10 | ✅ | [RESPUESTAS.md](RESPUESTAS.md), cinco reflexiones |
| C1 | HNSW contra fuerza bruta con FAISS (reto) | 15 | ◐ | [RESPUESTAS.md](RESPUESTAS.md); falta consignar `efSearch` y punto de equilibrio medidos |
| C2 | Un tercer paper: el RAG que explica HNSW (reto) | 12 | ◐ | [RESPUESTAS.md](RESPUESTAS.md), preguntas y ground truth; verificar resultados ejecutados |
| C3 | App Gradio para los papers con filtro por paper (reto) | 15 | ◐ | [RESPUESTAS.md](RESPUESTAS.md), descripción; falta confirmar captura de Gradio |
| C4 | Evaluación automática de la recuperación (reto) | 12 | ◐ | [RESPUESTAS.md](RESPUESTAS.md); faltan valores de la evaluación y CSV |
| C5 | MMR frente a similitud pura (reto) | 8 | ◐ | [RESPUESTAS.md](RESPUESTAS.md), interpretación; falta comparación observada de las consultas |
| C6 | Persistencia del índice en disco (reto) | 8 | ◐ | [RESPUESTAS.md](RESPUESTAS.md); faltan los tiempos medidos de indexación y reapertura |

**Autoevaluación:** pendiente de completar y contrastar con la rúbrica del profesor. Las actividades T1–T5, A1–A4, B1–B5 y R1 suman **70 puntos posibles**; los retos C1–C6 suman **70 puntos adicionales**. La lista completa contiene **140 puntos nominales**, por lo que no es correcto asumir una calificación de 140/100 ni asignarse 100/100 sin conocer cómo se ponderan los retos.

---

## Resultados destacados

### Teoría: HNSW, FAISS y búsqueda vectorial

Se analizó la distribución de nodos entre capas de HNSW, el efecto de `ef` sobre la exploración de candidatos y la diferencia entre insertar nuevos vectores y buscar vecinos. También se estudió el riesgo de desincronizar vectores y documentos en una implementación sencilla con FAISS, y el costo de una búsqueda exhaustiva frente a métodos aproximados.

### Bloque A: aplicación RAG empresarial

Se evaluó la aplicación con preguntas sobre precios, trabajo remoto, reembolsos y onboarding de CloudSync Pro. También se diseñó un corpus propio, se compararon instrucciones del prompt y se examinaron los scores de similitud. La propuesta de umbral `0.38` surge de los ejemplos observados, pero requiere validación adicional antes de adoptarse como regla general.

### Bloque B: RAG de artículos académicos

Se trabajó con los papers *Attention Is All You Need* y *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*.

- **B1:** las cinco preguntas originales coincidieron con el ground truth; la pregunta bonus se respondió parcialmente.
- **B2:** con `chunk_size` 400, 900 y 1500 se obtuvieron **326, 150 y 96 chunks**, respectivamente. En varias pruebas se lograron cinco respuestas correctas; otros intentos fallaron por errores `503` o timeout de NVIDIA, que no equivalen a respuestas conceptualmente incorrectas.
- **B3:** en tres preguntas sin respuesta respaldada por los papers, el sistema reconoció la falta de evidencia.
- **B4:** las preguntas equivalentes en español e inglés recuperaron fuentes del mismo artículo en las tres pruebas documentadas.
- **B5:** dos de tres preguntas propias recibieron respuestas completas. En la pregunta que exigía integrar ambos artículos, el sistema recuperó evidencia de ambos, pero la síntesis final fue parcial.

### Retos C1–C6

El documento de respuestas incluye desarrollos o criterios de análisis sobre comparación HNSW–FAISS, incorporación del paper HNSW, filtrado de papers en Gradio, evaluación automática, MMR y persistencia. Algunas tareas aún necesitan incorporar **mediciones, resultados impresos o capturas reales** antes de marcarlas como completamente verificadas.

---

## Cómo ejecutar los notebooks

1. Abre el notebook correspondiente en **Google Colab**.
2. Conecta el entorno de ejecución y ejecuta la celda de instalación de dependencias.
3. En **Secrets** de Colab, crea `NVIDIA_API_KEY`, introduce una clave válida y habilita el acceso para el notebook.
4. Ejecuta las celdas en orden. En el notebook 03, primero se descargan y convierten los papers; después se generan fragmentos, embeddings e índices.
5. Comprueba que el índice vectorial se creó correctamente antes de invocar las preguntas o abrir Gradio.
6. Ejecuta el quiz, los experimentos y las comparaciones; consulta [RESPUESTAS.md](RESPUESTAS.md) para revisar los resultados documentados.
7. Cuando corresponda, detén Gradio desde Colab y cierra instancias anteriores con `gr.close_all()`.

> **Seguridad:** no incluyas claves de API en los notebooks, archivos `.env` versionados, capturas o commits de GitHub. Una clave que se haya compartido accidentalmente debe revocarse y sustituirse.

> **Disponibilidad:** las llamadas a NVIDIA NIM dependen de servicios externos. Durante las pruebas se presentaron errores `503` y `ReadTimeout`; conviene distinguir un error de infraestructura de una respuesta incorrecta del sistema RAG.

---

## Decisiones y aprendizajes

Durante este taller comprendí cómo una arquitectura RAG conecta la representación vectorial de documentos, la recuperación por similitud y un modelo generativo. Aprendí que no basta con obtener una respuesta: también es necesario revisar si los fragmentos recuperados son pertinentes y si realmente respaldan lo que afirma el modelo.

Los experimentos con `chunk_size`, `k` y `score_threshold` me permitieron comparar configuraciones y reconocer que los cambios de parámetros no siempre producen mejoras evidentes en un corpus pequeño. También descubrí que los fallos de una API externa pueden alterar una evaluación, por lo que es importante separar los errores de infraestructura de los errores de contenido.

Al comparar prompts y diseñar preguntas con ground truth observé que pedir fuentes explícitas y reconocer cuando falta información ayuda a interpretar mejor las respuestas. Finalmente, la pregunta que requería relacionar dos papers mostró que recuperar los documentos adecuados no garantiza que el modelo consiga sintetizarlos completamente.

## Conclusión

El Workshop RAG me permitió conectar conceptos teóricos de HNSW, búsqueda aproximada y almacenamiento vectorial con implementaciones prácticas basadas en LangChain, Qdrant y NVIDIA NIM. La principal conclusión es que la confiabilidad de un RAG depende de todo el pipeline: documentos, fragmentación, embeddings, recuperación, prompt, generación y evaluación. Para mejorar el prototipo, sería necesario ampliar los conjuntos de prueba, comprobar las fuentes de manera sistemática y medir estabilidad y tiempos de respuesta en condiciones reproducibles.

---

## Autor

**Isabella Cadavid Posada**  
Ingeniería de Sistemas — Universidad EAFIT  
2026
