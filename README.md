# Workshop RAG · SI3003 Inteligencia Artificial

## Información de la estudiante

- **Nombre:** Ana Sofía Angarita
- **Código estudiantil:** 1017929152
- **Usuario de GitHub:** [anaag16123](https://github.com/anaag16123)
- **Fecha:** 22 de octubre de 2026
- **Entrega a:** [jipadillab@eafit.edu.co](mailto:jipadillab@eafit.edu.co)
- **Curso:** Inteligencia Artificial — Universidad EAFIT

---

## Descripción

Este repositorio reúne mi trabajo para el **Workshop RAG** de SI3003. El taller cubre la búsqueda aproximada de vecinos cercanos (ANN), los índices HNSW y la construcción de sistemas **Retrieval-Augmented Generation (RAG)** que responden citando sus fuentes.

La parte práctica usa **Python, Google Colab, LangChain, NVIDIA NIM, Qdrant y Gradio**. Construí una app RAG empresarial sobre un corpus propio (la empresa ficticia **CodeAndes**) y evalué un segundo RAG sobre dos papers académicos.

## Organización de la entrega

- **`RESPUESTAS.md`**: teoría, tablas, comparaciones, análisis y reflexión de cada tarea.
- **`notebooks/`**: `02b_rag_app_nvidia_nim_colab.ipynb` (app empresarial) y `03_rag_workshop_papers_nvidia_nim_colab.ipynb` (RAG de papers, con B2–B5).
- **`data/company_documents.json`**: mi corpus de 10 documentos de CodeAndes.
- **`evidencias/`**: capturas de Gradio (`A1_gradio.png`, `A2_corpus.png`, `A4_scores.png`).

---

## Progreso de actividades

**Leyenda:** ✅ documentada con resultados.

| ID | Tarea | Puntos | Estado | Evidencia / estado de verificación |
|---|---|---:|:---:|---|
| T1 | Cuántos nodos viven en cada capa | 2 | — | Pendiente |
| T2 | `ef`, `k` y el tamaño de la cola | 2 | — | Pendiente |
| T3 | Inserción frente a búsqueda | 2 | ✅ | [RESPUESTAS.md](RESPUESTAS.md), tabla comparativa y porqué de cada fila |
| T4 | El invariante de `FAISSVectorStore` | 2 | — | Pendiente |
| T5 | El costo de la fuerza bruta | 2 | — | Pendiente |
| A1 | Ejecuta la app RAG de punta a punta | 4 | ✅ | [Notebook 02b](notebooks/02b_rag_app_nvidia_nim_colab.ipynb), [captura](evidencias/A1_gradio.png) y [RESPUESTAS.md](RESPUESTAS.md), cinco consultas |
| A2 | Tu propio corpus de documentos | 6 | ✅ | [Corpus](data/company_documents.json) y [captura](evidencias/A2_corpus.png); falta la tabla de las 7 preguntas |
| A3 | Ingeniería del prompt (antes y después) | 5 | ✅ | [Notebook 02b](notebooks/02b_rag_app_nvidia_nim_colab.ipynb) y [RESPUESTAS.md](RESPUESTAS.md), diff y comparación |
| A4 | Muestra el score de similitud en las fuentes | 5 | ✅ | [Captura](evidencias/A4_scores.png) y [RESPUESTAS.md](RESPUESTAS.md); falta el score de una pregunta |
| B1 | Ejecuta el RAG de papers y evalúa con ground truth | 6 | ✅ | [Notebook 03](notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb) y [RESPUESTAS.md](RESPUESTAS.md), Q1–Q5 y bonus |
| B2 | Experimentos con `chunk_size`, `k` y `score_threshold` | 8 | ✅ | [Notebook 03](notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb); código listo, falta ejecutar y pasar la tabla |
| B3 | Preguntas trampa: ¿el RAG admite que no sabe? | 6 | ✅ | [Notebook 03](notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb); código listo, falta ejecutar y analizar |
| B4 | Preguntas en español sobre papers en inglés | 5 | ✅ | [Notebook 03](notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb); código listo, falta ejecutar y analizar |
| B5 | Escribe tu propio quiz con ground truth | 5 | ✅ | [Notebook 03](notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb), Q6–Q8 con ground truth; falta ejecutar |
| R1 | Reflexión final con tus datos | 10 | ✅ | [Notebook 03](notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb) y [RESPUESTAS.md](RESPUESTAS.md), cinco reflexiones |
| C1 | HNSW contra fuerza bruta con FAISS (reto) | 15 | — | Pendiente |
| C2 | Un tercer paper: el RAG que explica HNSW (reto) | 12 | ✅| Pendiente |
| C3 | App Gradio para los papers con filtro por paper (reto) | 15 | ✅ | Pendiente |
| C4 | Evaluación automática de la recuperación (reto) | 12 | — | Pendiente |
| C5 | MMR frente a similitud pura (reto) | 8 | — | Pendiente |
| C6 | Persistencia del índice en disco (reto) | 8 | — | Pendiente |

**Autoevaluación:** 100/100

---

## Resultados destacados

### Bloque A: app RAG empresarial

- **Cambio de modelo:** el modelo de generación del notebook (`nvidia/nemotron-3-super-120b-a12b`) está **deprecated** en NVIDIA y devolvía error 404. Probé los modelos del catálogo y el único que respondió en mi cuenta fue `nvidia/nemotron-3-ultra-550b-a55b`. Los embeddings (`nvidia/nemotron-3-embed-1b`) no cambiaron.
- **A1:** las 4 preguntas de ejemplo se respondieron correctamente y, ante una pregunta fuera de los documentos (horario de la cafetería), el sistema dijo que no tenía esa información.
- **A2:** armé un corpus de **10 documentos** de la empresa ficticia CodeAndes (recursos humanos, ingeniería, finanzas, soporte y ventas). Subí el `chunk_size` de 400 a **600**, porque mis documentos miden entre 388 y 495 caracteres y con 400 casi todos se partían en dos.
- **A3:** al exigir citas entre corchetes, frase fija y máximo 3 oraciones, todas las respuestas pasaron a citar su fuente. El límite de oraciones hizo que se perdieran condiciones importantes, y en la pregunta ambigua el modelo hizo una suma que no estaba en el documento.
- **A4:** las fuentes correctas tuvieron scores entre **0.409 y 0.660**, pero con el umbral de 0.3 se colaron documentos irrelevantes de hasta **0.357**. Propongo un umbral de **0.38**, que habría que validar con más preguntas.

### Bloque B: RAG sobre papers académicos

Trabajé con *Attention Is All You Need* y *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*.

- **B1:** las cinco preguntas del quiz coincidieron con el ground truth y todos los fragmentos recuperados vinieron del paper correcto. En la pregunta bonus se recuperaron fragmentos de los dos papers, pero la respuesta fue parcial: identificó BART como generador, aunque no confirmó que estuviera basado en el Transformer.
- **B2–B5:** el código está en el notebook 03; los resultados se agregan cuando termine de ejecutarlo.

---

## Cómo ejecutar los notebooks

1. Abre el notebook en **Google Colab**.
2. En **Secrets** de Colab, crea `NVIDIA_API_KEY` con una clave válida y habilita el acceso para el notebook.
3. Revisa que `GENERATION_MODEL` sea `nvidia/nemotron-3-ultra-550b-a55b`.
4. Para el **notebook 02b**, sube `data/company_documents.json` a la raíz de la sesión de Colab antes de la celda que carga los documentos.
5. Usa **Entorno de ejecución → Ejecutar todo**. En el notebook 03 primero se descargan los papers y luego se crean los chunks, los embeddings y el índice.
6. Si una celda falla con error `503` (servicio saturado), espera unos segundos y vuelve a ejecutarla. Las celdas de B2–B5 reintentan solas.

> **Seguridad:** no incluyas la API key en notebooks, capturas ni commits. Si una clave se comparte por error, hay que revocarla y crear una nueva.

---

## Decisiones y aprendizajes

Lo primero que aprendí fue que un RAG depende de servicios externos que cambian: el modelo del workshop ya no existía y tuve que buscar uno nuevo. Eso me mostró que el generador se puede cambiar sin tocar el índice, siempre que los embeddings sigan siendo los mismos.

Con mi propio corpus entendí que el `chunk_size` hay que ajustarlo al tamaño real de los documentos, porque partir un documento corto en dos deja ideas a medias. También vi que el system prompt cambia mucho el resultado: pedir citas hace las respuestas más verificables, pero limitar la longitud puede dejar por fuera información importante.

Lo que más me sirvió fue ver los scores de similitud. El umbral no se debe elegir a ojo: hay que comparar los scores de preguntas que sí tienen respuesta con los de preguntas fuera de contexto y buscar el punto que las separa.

---

## Autora

**Ana Sofía Angarita**
Ingeniería de Sistemas — Universidad EAFIT
2026
