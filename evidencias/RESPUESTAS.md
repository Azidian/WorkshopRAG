# RESPUESTAS — Evaluación del sistema RAG sobre papers académicos

## Configuración del experimento

- **Notebook:** `03_rag_workshop_papers_nvidia_nim_colab.ipynb`
- **Modelo generativo:** `nvidia/nemotron-3-ultra-550b-a55b`
- **Modelo de embeddings:** `nvidia/nemotron-3-embed-1b`
- **Base vectorial:** Qdrant en memoria
- **Documentos:** *Attention Is All You Need* y *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*
- **Chunks indexados:** 150
- **Dimensión de embeddings:** 2048
- **Parámetros de recuperación:** `k=5`, `score_threshold=0.25`

## Tabla de respuestas y comparación con el ground truth

| Pregunta | Respuesta del RAG (resumen fiel) | ¿Correcta? | ¿Contexto del paper correcto? | Comparación con el ground truth |
|---|---|---|---|---|
| **Q1 — BLEU score** | Transformer big obtuvo **28.4 BLEU** en WMT 2014 inglés-alemán. | **Sí** | **Sí:** *Attention Is All You Need* (`attention_paper.pdf`; chunks 28, 1, 26, 40 y 31). | Coincide con el valor de referencia; la respuesta no destacó la mejora de más de dos puntos frente a resultados anteriores. |
| **Q2 — Memoria RAG** | Combina **BART** como memoria paramétrica con un índice denso de Wikipedia como memoria no paramétrica; **DPR** recupera los documentos, usados para generar la respuesta. | **Sí** | **Sí:** *Retrieval-Augmented Generation* (`rag_paper.pdf`; chunks 1, 8, 32, 7 y 50). | Coincide con la explicación del ground truth e incorpora detalles adicionales del modelo probabilístico. |
| **Q3 — Scaling factor** | El factor **1/√dₖ** evita productos punto demasiado grandes, que saturarían softmax y producirían gradientes pequeños. | **Sí** | **Sí:** *Attention Is All You Need* (`attention_paper.pdf`; chunks 11, 12, 13, 16 y 20). | Coincide con la explicación esperada del escalamiento y su efecto en el entrenamiento. |
| **Q4 — Variantes RAG** | **RAG-Sequence** condiciona la secuencia completa con un documento; **RAG-Token** permite diferentes documentos por token. | **Sí** | **Sí:** *Retrieval-Augmented Generation* (`rag_paper.pdf`; chunks 10, 11, 1, 50 y 7). | Explica correctamente la diferencia; omite el costo computacional adicional de RAG-Token, que no era el foco de la pregunta. |
| **Q5 — Atención multi-cabeza** | Transformer base usa **8 cabezas**, con **dₖ = dᵥ = 64** por cabeza. | **Sí** | **Sí:** *Attention Is All You Need* (`attention_paper.pdf`; chunks 14, 15, 9, 0 y 40). | Coincide con los valores del ground truth e incluye la relación 512/8 = 64. |
| **Bonus — Relación RAG y Transformer** | Identificó **BART** como generador seq2seq, pero indicó que los extractos no confirmaban si BART se basa en Transformer. | **Parcial** | **Fuente citada en la respuesta:** *Retrieval-Augmented Generation* (`rag_paper.pdf`). **Lista de fragmentos recuperados aún no compartida**; no se confirma la aparición de ambos papers. | El *ground truth* exige **BART-large** y confirmar que es un modelo basado en Transformer. La respuesta identifica la familia BART, pero deja sin resolver la segunda parte. |

## Evidencia específica del Bonus

**Pregunta original:**

> The RAG paper uses FAISS for fast retrieval with an HNSW approximation. The Attention paper proposes the Transformer architecture. What seq2seq model does RAG use as its generator component, and is it based on the Transformer?

**Respuesta real generada por el RAG:**

> Based on the supplied excerpts from the RAG paper, the seq2seq model used as the generator component is **BART** (referenced as "BART[32]"). The excerpts state: "the seq2seq model (BART[32]) then conditions on these latent documents together with the input to generate the output" and "Like T5[51] or BART, RAG..."
>
> The provided excerpts do not explicitly state whether BART is based on the Transformer architecture. The Attention paper (separate document) proposes the Transformer architecture, but the RAG paper excerpts do not describe BART's architectural basis.
>
> **Paper title supporting the answer:** Retrieval-Augmented Generation | rag_paper.pdf

**Ground truth de la pregunta:** el generador seq2seq utilizado por RAG es **BART-large**; BART se basa en la arquitectura Transformer.

### Evaluación del Bonus

- **¿La respuesta fue correcta? Parcial.** Identifica BART, que es el modelo generador pertinente, pero no especifica la variante *large* ni confirma que su arquitectura está basada en Transformer.
- **¿El contexto vino del paper correcto? Parcialmente verificado.** La respuesta cita explícitamente *Retrieval-Augmented Generation* (`rag_paper.pdf`), fuente pertinente para identificar el generador. No se ha facilitado todavía el listado de fragmentos recuperados por `retriever.invoke(q_bonus)`.
- **¿Aparecieron ambos papers? No verificable con la evidencia compartida.** La respuesta menciona *Attention Is All You Need* como un documento separado, pero eso no demuestra que el recuperador haya devuelto fragmentos de ese artículo. Para responder **sí** o **no** hace falta la salida de los títulos y archivos recuperados.
- **Completitud:** parcial. La respuesta reconoció una falta de evidencia para la segunda parte en vez de afirmarla sin respaldo.

**Evidencia pendiente para el requisito del profesor:** copiar la salida de `retriever.invoke(q_bonus)` con los títulos, archivos e índices de chunks; solo entonces confirmar si se recuperaron ambos papers.

### Código para verificar los papers recuperados

```python
# Ejecutar en una nueva celda de Colab: no vuelve a generar la respuesta
fragmentos_bonus = retriever.invoke(q_bonus)

for i, doc in enumerate(fragmentos_bonus, 1):
    print(f"{i}. {doc.metadata.get('title')} | {doc.metadata.get('source')} | chunk {doc.metadata.get('chunk_index')}")

archivos = {doc.metadata.get('source') for doc in fragmentos_bonus}
print("¿Aparecieron ambos papers?", "Sí" if {"rag_paper.pdf", "attention_paper.pdf"}.issubset(archivos) else "No")
```

## Conclusión

En las cinco preguntas Q1–Q5, las respuestas del RAG coincidieron con los datos esenciales del *ground truth*. El recuperador devolvió cinco fragmentos del artículo esperado en cada consulta (25 de 25 fragmentos del paper correspondiente). Este resultado describe las preguntas evaluadas y no debe interpretarse como una medida general de precisión del sistema. En el Bonus, el modelo identificó BART correctamente, pero no confirmó la relación arquitectónica con Transformer, por lo que su respuesta se clasificó como **parcial**. La relación con las fuentes solo puede cerrarse cuando se registre el listado de fragmentos recuperados y se compruebe si aparecieron ambos artículos.
