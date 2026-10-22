# Evidencias de funcionamiento y evaluación del sistema RAG

## 1. Objetivo de la evaluación

El objetivo de esta evaluación es comprobar que el sistema RAG implementado es capaz de recuperar información relevante de los artículos académicos y generar respuestas correctas, completas y respaldadas por las fuentes originales.

Para ello, se ejecutaron cinco consultas relacionadas con los conceptos de Transformer y Retrieval-Augmented Generation (RAG). Los resultados se compararon con las respuestas de referencia (*ground truth*) proporcionadas en el notebook.

Se evaluaron tres criterios:

- **Correctitud:** coincidencia entre la información generada y la respuesta de referencia.
- **Fuente:** correspondencia entre los fragmentos recuperados y el artículo académico adecuado.
- **Completitud:** presencia de los conceptos y datos necesarios para responder cada consulta.

## 2. Configuración del sistema evaluado

| Componente | Configuración |
|---|---|
| Modelo generativo | `nvidia/nemotron-3-ultra-550b-a55b` |
| Modelo de embeddings | `nvidia/nemotron-3-embed-1b` |
| Base vectorial | Qdrant en memoria |
| Artículos indexados | 2 |
| Fragmentos indexados | 150 |
| Dimensión de embeddings | 2048 |
| Tamaño de fragmento | 900 caracteres |
| Solapamiento | 120 caracteres |
| Fragmentos recuperados por consulta | 5 |
| Umbral mínimo de similitud | 0.25 |

La indexación se completó correctamente, permitiendo consultar los documentos mediante búsquedas semánticas. El modelo generativo recibió los fragmentos recuperados como contexto para construir sus respuestas.

## 3. Resultados generales

| Evaluación | Correctitud factual | Fuente correcta | Completitud |
|---|---|---|---|
| Pregunta 1 | Sí | Sí | Sí |
| Pregunta 2 | Sí | Sí | Sí |
| Pregunta 3 | Sí | Sí | Sí |
| Pregunta 4 | Sí | Sí | Sí |
| Pregunta 5 | Sí | Sí | Sí |

**Resultado general:** las cinco respuestas coincidieron con la información esencial del *ground truth*, y las fuentes recuperadas correspondieron al artículo esperado en cada caso.

En algunas respuestas se incluyó información adicional o una explicación más breve que la referencia, sin afectar la respuesta principal.

## 4. Evidencia de recuperación de documentos

Los resultados de la búsqueda semántica permiten comprobar que el recuperador seleccionó fragmentos de los artículos correspondientes a cada pregunta.

| Pregunta | Artículo recuperado | Índices de chunks |
|---|---|---|
| 1 | Attention Is All You Need | 28, 1, 26, 40, 31 |
| 2 | Retrieval-Augmented Generation | 1, 8, 32, 7, 50 |
| 3 | Attention Is All You Need | 11, 12, 13, 16, 20 |
| 4 | Retrieval-Augmented Generation | 10, 11, 1, 50, 7 |
| 5 | Attention Is All You Need | 14, 15, 9, 0, 40 |

### Análisis de las fuentes

En las cinco consultas, los cinco fragmentos recuperados pertenecieron al artículo académico esperado.

Esto indica que, para las preguntas evaluadas, el sistema de recuperación identificó correctamente el documento relevante.

Los resultados también muestran que el recuperador no seleccionó información del otro artículo en estas consultas, lo que evidencia una separación temática adecuada entre los contenidos utilizados en la evaluación.

Sin embargo, la recuperación del artículo correcto no garantiza por sí sola que todos los fragmentos sean igualmente relevantes. La calidad de las respuestas debe analizarse conjuntamente con su contenido.

## 5. Indicadores obtenidos

A partir de las cinco evaluaciones se calcularon los siguientes indicadores descriptivos:

| Indicador | Resultado |
|---|---|
| Preguntas evaluadas | 5 |
| Respuestas factualmente correctas | 5 de 5 |
| Respuestas con fuente esperada | 5 de 5 |
| Respuestas suficientemente completas | 5 de 5 |
| Fragmentos recuperados en total | 25 |
| Fragmentos provenientes del artículo esperado | 25 de 25 |

### Interpretación

**Correctitud factual observada: 100 % (5/5).**

Las cinco respuestas incluyeron la información esencial esperada.

**Correspondencia de fuentes: 100 % (25/25).**

Los 25 fragmentos recuperados procedieron del artículo correspondiente a cada pregunta.

**Completitud suficiente: 100 % (5/5).**

Todas las respuestas abordaron los elementos principales solicitados, aunque algunas omitieron detalles complementarios de las respuestas de referencia.

Estos porcentajes describen únicamente las cinco consultas ejecutadas. No representan una medición general del desempeño del modelo en preguntas distintas o más complejas.

## 6. Observaciones del experimento

Durante la evaluación se identificaron los siguientes comportamientos:

- El sistema recuperó información de acuerdo con el contenido semántico de las consultas.
- Las respuestas incluyeron conceptos y valores consistentes con las referencias académicas.
- Los fragmentos recuperados provinieron de los documentos adecuados en todas las preguntas.
- El modelo presentó información de forma organizada y respondió en inglés, de acuerdo con el idioma de las consultas.
- Algunas respuestas incorporaron explicaciones adicionales, mientras que otras sintetizaron la información del artículo.
- La evaluación se realizó con un conjunto pequeño de preguntas cuyos temas estaban cubiertos por los documentos indexados.

## 7. Conclusión

La evaluación permitió comprobar el funcionamiento del sistema RAG construido con NVIDIA NIM, LangChain y Qdrant.

En las cinco consultas realizadas, el sistema recuperó fragmentos de los artículos correctos y generó respuestas consistentes con las referencias proporcionadas.

Se obtuvo una correctitud factual observada del 100 % en las cinco preguntas y una correspondencia del 100 % entre los fragmentos recuperados y los artículos esperados.

Estos resultados muestran que la combinación de recuperación semántica y generación de respuestas funcionó adecuadamente para las consultas evaluadas.

Como limitación, el experimento incluye únicamente cinco preguntas académicas. Una evaluación más amplia debería incorporar consultas ambiguas, preguntas sin respuesta en los documentos y casos que requieran integrar información de diferentes artículos.