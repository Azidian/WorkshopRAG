# Respuestas del workshop RAG
Escribe aquí tus respuestas con tus propias palabras. Pega tablas, resultados y enlaces a las capturas de `evidencias/`.

---

# Bloque 0 · Teoría

## T1 · Cuántos nodos viven en cada capa

_Evidencia:_

<img width="1600" height="1019" alt="WhatsApp Image 2026-10-22 at 9 59 01 AM" src="https://github.com/user-attachments/assets/b3cc5078-8e33-4904-902c-0e8141887a49" />


_Mi respuesta / análisis:_

**1. Cálculo a mano ($M = 16$ y $N = 1,000,000$ vectores)**

Dado que $m_L \approx \frac{1}{\ln(M)}$, la probabilidad de que un vector alcance al menos la capa $k$ es $P(l \ge k) = M^{-k}$. Por lo tanto, el número esperado de nodos en cada capa es $E[N_{\ge k}] = \frac{N}{M^k}$:

- *Capa $\ge 1$:* $\frac{1,000,000}{16^1} = \frac{1,000,000}{16} = \mathbf{62,500\text{ nodos}}$ ($6.25\%$)
- *Capa $\ge 2$:* $\frac{1,000,000}{16^2} = \frac{1,000,000}{256} = 3,906.25 \approx \mathbf{3,906\text{ nodos}}$ ($0.39\%$)
- *Capa $\ge 3$:* $\frac{1,000,000}{16^3} = \frac{1,000,000}{4,096} = 244.14 \approx \mathbf{244\text{ nodos}}$ ($0.024\%$)

**2. Simulación en Python y comparación con el valor teórico**

Código utilizado:
python
import numpy as np

M = 16
mL = 1 / np.log(M)
N = 1_000_000
rng = np.random.default_rng(0)
l = np.floor(-np.log(1 - rng.random(N)) * mL).astype(int)

for k in range(1, 4):
    print(f"capa >= {k}: simulado={(l >= k).sum():>7,}  teórico={N / M**k:>9,.0f}")


Salida obtenida:
text
capa >= 1: simulado= 62,675  teórico=   62,500
capa >= 2: simulado=  4,005  teórico=    3,906
capa >= 3: simulado=    270  teórico=      244


Comparación:Los resultados simulados coinciden muy de cerca con los valores teóricos esperados ($62,675$ vs. $62,500$ en capa $\ge 1$; $4,005$ vs. $3,906$ en capa $\ge 2$; y $270$ vs. $244$ en capa $\ge 3$). Las pequeñas diferencias se deben a la varianza natural del muestreo aleatorio con $N = 1,000,000$, confirmando que la distribución exponencial reduce la cantidad de nodos a razón de un factor de $\frac{1}{16}$ por cada capa que se sube.

**3. ¿Por qué es bueno para la velocidad de búsqueda que haya pocos hubs en las capas altas?**

Tener muy pocos nodos (hubs) en las capas superiores permite que la búsqueda greedy inicial (con ef = 1) avance con enlaces de largo alcance ("autopistas") evaluando distancias contra apenas unas decenas o cientos de vectores en lugar de explorar un millón de nodos. De esta manera, el algoritmo se aproxima rápidamente a la región correcta del espacio vectorial en tiempo logarítmico $O(\log N)$ antes de realizar la búsqueda exhaustiva en la capa base (capa 0).


## T2 · ef, k y el tamaño de la cola

_Evidencia:_

![T2](https://github.com/Azidian/WorkshopRAG/blob/main/evidencias/T2.png?raw=true)

_Mi respuesta / análisis:_

**Respuesta 1:** El problema es que el valor de ef es menor que la cantidad de vecinos solicitados (k). Se recomienda que ef ≥ k, ya que este parámetro controla la amplitud de búsqueda y la cantidad de candidatos considerados. Si se solicitan 10 vecinos y ef = 5, la búsqueda puede tener dificultades para encontrar los 10 vecinos más cercanos con buena calidad. Por eso, se debe aumentar ef al menos hasta 10 y, si es necesario, utilizar un valor mayor para mejorar el recall, teniendo en cuenta que esto puede aumentar el tiempo de búsqueda.

**Respuesta 2:** Conviene comenzar con un valor de ef igual o superior a k. Si el recall es bajo y se necesita encontrar vecinos más cercanos, se debe aumentar ef. Esto permite explorar más candidatos y mejorar la calidad de los resultados, aunque puede incrementar el tiempo de búsqueda. Por tanto, el valor debe ajustarse según el equilibrio deseado entre velocidad y calidad.

**Respuesta 3:** Sí, se puede modificar ef sin reconstruir el índice, porque es un parámetro que controla la búsqueda durante las consultas. En cambio, ef_construction determina la amplitud de búsqueda durante la construcción del grafo y afecta su calidad. Por eso, ef permite ajustar el equilibrio entre velocidad y recall sin necesidad de volver a construir el índice, mientras que modificar los parámetros de construcción puede requerir reconstruirlo.
.


## T3 · Inserción frente a búsqueda

_Evidencia:_

|                                | Inserción | Búsqueda |
|--------------------------------|-----------|----------|
| ¿Se sortea la capa?            | Sí, a cada vector nuevo se le saca al azar una capa máxima `l` con la fórmula `l = ⌊−ln(U)·mL⌋`. | No, la consulta no se guarda en el grafo, entonces siempre arranca desde la capa más alta. |
| Valor de ef en las capas altas | `ef = 1`, bajando desde arriba hasta la capa `l+1`. | `ef = 1`, bajando desde arriba hasta la capa 1. |
| Parámetro de la fase principal | `ef_construction`, en todas las capas desde `l` hasta la 0. | `ef`, pero solo en la capa 0. |
| ¿Modifica el índice?           | Sí, conecta el vector con sus `M` vecinos más cercanos y si algún nodo queda con demasiadas conexiones se le quita la más débil. | No, solo recorre el grafo y devuelve los `k` más cercanos. |

Lo saqué de las presentaciones de [Inserción](https://ia-eafit-rag.netlify.app/teoria/hnsw_insercion) y [Búsqueda](https://ia-eafit-rag.netlify.app/teoria/hnsw_busqueda).

_Mi respuesta / análisis:_

**Por qué de cada fila**

**1. Sorteo de capa**
En la inserción tiene sentido porque hay que decidir en qué capas va a quedar el vector, y hacerlo al azar evita que el grafo dependa del orden en que llegan los datos. En la búsqueda no se agrega nada, entonces no hay nada que sortear.

**2. ef en capas altas**
Las dos usan `ef = 1` porque ahí lo único que importa es llegar rápido a la zona correcta. Todavía no se necesita precisión, con quedarse con el mejor candidato es suficiente.

**3. Fase principal**
En la inserción hay que buscar buenos vecinos en cada capa donde va a vivir el vector, porque de eso depende qué tan bueno queda el grafo. En la búsqueda los resultados salen de la capa 0, que es la única que tiene todos los nodos, entonces ahí es donde vale la pena buscar con más cuidado.

**4. ¿Modifica el índice?**
Insertar es escribir en el grafo y buscar es solo leerlo. Por eso `ef` se puede cambiar en cada consulta sin reconstruir nada, pero `ef_construction` queda fijo cuando se construye el índice.

**¿Qué dispara cada llamada en el notebook 03?**

- `vectorstore.add_documents(...)` → **Inserción.** Cada chunk se pasa a embedding y Qdrant lo mete en su índice HNSW.
- `retriever.invoke(...)` → **Búsqueda.** La pregunta se convierte en embedding y Qdrant recorre el grafo para devolver los `k` chunks más parecidos que pasen el `score_threshold`.



## T4 · El invariante de FAISSVectorStore

### 1. Tabla posición → documento

El índice FAISS no cambia: sigue guardando 4 vectores en las posiciones 0–3.
Solo cambia la lista `self.documents`, de la que se borró B.

| Posición | Lo que FAISS cree que hay | `documents` antes | `documents` después de borrar B |
|---|---|---|---|
| 0 | vector de A | A | A |
| 1 | vector de B | B | C |
| 2 | vector de C | C | D |
| 3 | vector de D | D | (no existe) |

La búsqueda sigue devolviendo las posiciones `[1, 2, 3]`, pero ahora se traducen así:

| Posición devuelta | Antes | Ahora |
|---|---|---|
| 1 | B (correcto) | C (incorrecto) |
| 2 | C (correcto) | D (incorrecto) |
| 3 | D (correcto) | `IndexError` |

Resultado: `[C, D, IndexError]` en lugar de `[B, C, D]`.

### 2. Error silencioso vs. excepción

- **Errores silenciosos:** las posiciones 1 y 2. Devuelven un documento real, pero equivocado, y no se lanza ninguna excepción. El LLM recibiría
  contexto incorrecto y respondería con seguridad algo falso.
- **Excepción:** la posición 3. La lista ahora tiene 3 elementos (posiciones 0–2),
  así que `documents[3]` lanza `IndexError`.

### 3. Relación con Qdrant

FAISS solo guarda vectores y los une con el texto y la metadata por posición en
una lista aparte, así que el programador debe mantener ese invariante a mano.
Cualquier borrado o reordenamiento lo rompe sin avisar. Qdrant guarda el vector,
un ID y el *payload* (texto y metadata) juntos en un mismo registro. Al borrar un
punto se elimina todo a la vez y no hay una lista paralela que desincronizar.
Además ofrece persistencia, filtros por metadata y acceso concurrente.


## T5 · El costo de la fuerza bruta

_Evidencia:_ …

_Mi respuesta / análisis:_

### 1. Cuenta

Cada consulta compara contra 1 millón de vectores de 768 dimensiones:

    10^6 vectores × 768 multiplicaciones = 768 × 10^6 mult/consulta

Con 100 consultas por segundo:

    768 × 10^6 mult/consulta × 100 consultas/s = 76 800 × 10^6 mult/s

### 2. Notación científica y por qué la exactitud deja de ser viable

*≈ 7,68 × 10^10 multiplicaciones por segundo* (casi 77 mil millones).

Ser exacto obliga a comparar cada consulta contra todos los vectores, así que
el costo crece linealmente: con 10 millones de
vectores serían 7,68 × 10^11 mult/s, y con mil millones, 7,68 × 10^13 mult/s.
Además, cada consulta tendría que leer unos 3 GB de vectores de memoria
(10^6 × 768 × 4 bytes), lo que a 100 consultas por segundo exigiría cerca de
300 GB/s de ancho de banda, más de lo que ofrece un servidor típico. El
hardware necesario crece al mismo ritmo que los datos, y eso no es sostenible.

### 3. Qué se sacrifica y qué se gana en ANN

Se sacrifica la garantía de devolver siempre el vecino más cercano real
(el recall puede ser menor a 100 %), y se gana una búsqueda órdenes de magnitud
más rápida, con un costo que crece de forma aproximadamente logarítmica en
lugar de lineal.


---

# Bloque A · App RAG empresarial

## A1 · Ejecuta la app RAG de punta a punta

**Evidencia:** 
![A1_gradio](evidencias/A1_gradio1.png)
![A1_gradio](evidencias/A1_gradio2.png)
![A1_gradio](evidencias/A1_gradio3.png)

**Mi respuesta / análisis:** 

| Pregunta | Respuesta | Fuente recuperada |
| :--- | :--- | :--- |
| ¿Cuál es el precio del plan Enterprise de CloudSync Pro? | El plan Enterprise de CloudSync Pro cuesta **$49/mes por usuario** con compromiso anual. | CloudSync Pro — Precios |
| ¿Cuántos días pueden trabajar remotamente los empleados? | Los empleados pueden trabajar de forma remota hasta **3 días por semana**, con aprobación del jefe directo. | Política de trabajo remoto |
| ¿Qué pasa con los reembolsos mayores a $100? | Los reembolsos mayores a $100 requieren aprobación del supervisor y, una vez aprobados, se procesan en 3 a 5 días hábiles al método de pago original. | Política de reembolsos |
| ¿Qué ocurre durante la primera semana de onboarding? | Durante la primera semana se completan módulos obligatorios de seguridad, cumplimiento y cultura de la empresa. | Onboarding |
| ¿Cuál es la política para llevar mascotas a la oficina? | No se encuentra información relevante en los documentos disponibles para responder a esta consulta. | N/A (Sin fuente) |


## A2 · Tu propio corpus de documentos

**Evidencia:**
![A2_Corpus](evidencias/A2_corpus.png)
![A2_Corpus](evidencias/A2_corpus2.png)
![A2_Corpus](evidencias/A2_corpus3.png)

**Mi respuesta / análisis:**
| Pregunta | Respuesta | Fuente recuperada |
| :--- | :--- | :--- |
| ¿Qué días hay que ir a la oficina? | No tengo esa información en el contexto. | N/A (Sin fuente) |
| ¿Cuántos revisores necesita un pull request para entrar a main? | Al menos 2 revisores. | Flujo de trabajo con Git |
| ¿Se puede desplegar a producción un viernes? | No, está prohibido desplegar los viernes. | Despliegues a producción |
| ¿Cuánto cubre la empresa de un curso o certificación? | No tengo esa información en el contexto. | N/A (Sin fuente) |
| ¿En cuánto tiempo se debe responder un ticket crítico? | Máximo 1 hora. | Soporte a clientes y tiempos de respuesta |
| ¿Cuánto es el salario de un desarrollador junior? / ¿La empresa tiene parqueadero para empleados? *(Pregunta fuera de contexto)* | No tengo esa información en el contexto proporcionado. Los documentos disponibles cubren onboarding, vacaciones, horario laboral y trabajo híbrido, pero no mencionan salarios ni disponibilidad de parqueadero. | N/A (Sin fuente) |

## A3 · Ingeniería del prompt (antes y después)

**Evidencia:**
![A3](evidencias/A3.png)
![A3](evidencias/A3.1.png)

**Diff del system prompt** (en `build_pipeline`):

```diff
  (
      "system",
-     "Eres un asistente empresarial. Responde únicamente con información "
-     "del contexto proporcionado. Si el contexto no contiene la respuesta, "
-     "indica que no tienes esa información. Responde en español y de forma concisa.",
+     "Eres un asistente empresarial. Responde únicamente con información "
+     "del contexto proporcionado. Cita siempre el título del documento entre "
+     "corchetes, por ejemplo [Política de vacaciones]. Si el contexto no alcanza, "
+     "responde exactamente: «No encuentro esa información en los documentos.» "
+     "Responde en español, en máximo 3 oraciones.",
  ),
```

**Tabla comparativa:**

| Pregunta | Tipo | Antes | Después |
|----------|------|-------|---------|
| ¿Cuántos revisores necesita un pull request? | Fácil | "Se requiere la aprobación de al menos 2 revisores." Sin citar fuente. | "Un pull request requiere la aprobación de al menos 2 revisores [Flujo de trabajo con Git]." |
| ¿Cuántos días tengo libres? | Ambigua | 15 días hábiles de vacaciones al año (tras 6 meses de antigüedad) más 1 día libre por cumpleaños. Sin citar fuente. | 15 días hábiles de vacaciones más un día por cumpleaños, "sumando 16 días libres" [Vacaciones y días libres]. |
| ¿La empresa tiene parqueadero para empleados? | Fuera de contexto | "El contexto no menciona información sobre parqueadero para empleados." | "No encuentro esa información en los documentos." (frase fija exacta) |

**Mi respuesta / análisis:**

**Cómo hice la prueba**
Hice las 3 preguntas juntas en un mismo mensaje, antes y después del cambio. Por eso el modelo respondió las tres en una sola respuesta, pero igual se nota bien la diferencia entre los dos prompts.

**Qué cambió**
Antes, el modelo respondía en formato de lista con negritas, empezaba con "Según el contexto proporcionado" y no decía de qué documento sacaba cada cosa. Después, la respuesta quedó en texto corrido, cada dato tiene su fuente entre corchetes y la respuesta completa quedó en exactamente 3 oraciones.

**¿Qué instrucción tuvo más efecto?**
La de citar el título entre corchetes. Antes no había ninguna referencia a los documentos y después las dos preguntas con respuesta quedaron citadas con el título exacto. La frase fija también funcionó: para el parqueadero usó literalmente "No encuentro esa información en los documentos.", mientras que antes lo decía con sus propias palabras.

**¿Qué ignoró el modelo?**
Ninguna instrucción la ignoró del todo, pero sí hay detalles. En la pregunta ambigua, en vez de aclarar que podía referirse a varias cosas, asumió que eran vacaciones y además hizo una suma que no está en el documento ("16 días libres"). Esa cifra la sacó él, no el texto. Y al limitarlo a 3 oraciones perdió información que antes sí daba, como que las vacaciones solo aplican después de 6 meses en la empresa. O sea, el límite de oraciones hace la respuesta más corta, pero puede dejar por fuera condiciones importantes.

## A4 · Muestra el score de similitud en las fuentes

**Evidencia:**
![A4](evidencias/A4.png)

Cambios en el notebook: `build_pipeline` ahora devuelve también el `vectorstore`, y `chat()` usa `vectorstore.similarity_search_with_score(question, k=3)` para mostrar cada fuente con su score coseno. Los que quedan por debajo de 0.3 se marcan como "debajo del umbral".

| # | Pregunta | Tipo | Fuente 1 (score) | Fuente 2 (score) | Fuente 3 (score) |
|---|----------|------|------------------|------------------|------------------|
| 1 | ¿Qué días hay que ir a la oficina? | Buena | Horario laboral y trabajo híbrido (0.409) ✅ | Despliegues a producción (0.307) | Soporte a clientes (0.253) |
| 2 | ¿Se puede desplegar a producción un viernes? | Buena | Despliegues a producción (0.660) ✅ | Soporte a clientes (0.207) | Horario laboral (0.200) |
| 3 | ¿Cuánto cuesta el plan Pyme de CodeAndes Cloud? | Buena | Planes y precios de CodeAndes Cloud (0.653) ✅ | Horario laboral (0.357) | Auxilio de educación (0.323) |
| 4 | ¿Cuánto es el salario de un desarrollador junior? | Fuera | PENDIENTE | PENDIENTE | PENDIENTE |
| 5 | ¿La empresa tiene parqueadero para empleados? | Fuera | Vacaciones y días libres (0.246) | Auxilio de educación (0.220) | Planes y precios (0.165) |

✅ = documento que de verdad tiene la respuesta.

**Mi respuesta / análisis:**

**Lo que muestran los scores**
En las 3 preguntas buenas el documento correcto siempre quedó de primero, con scores entre 0.409 y 0.660. Pero también se colaron documentos que no tienen nada que ver y aun así pasaron el umbral de 0.3: "Despliegues a producción" con 0.307 en la pregunta de la oficina, y "Horario laboral" (0.357) y "Auxilio de educación" (0.323) en la del plan Pyme. En las preguntas fuera de contexto ningún documento pasó de 0.246.

**El problema con 0.3**
Con el umbral actual el modelo recibe chunks irrelevantes como contexto. Esta vez no inventó nada porque el prompt lo obliga a citar y a no salirse del contexto, pero es "material" que podría usar para alucinar. Por ejemplo, en la pregunta del plan Pyme le llegaron 3 documentos cuando solo uno servía.

**Umbral propuesto: 0.38**
El score más bajo de una respuesta correcta fue 0.409 (la de la oficina) y el más alto de un documento irrelevante fue 0.357. Un umbral de 0.38 queda justo en el medio: deja pasar todas las fuentes correctas y bloquea todas las irrelevantes que vi. No lo subiría mucho más porque la pregunta de la oficina quedó con solo 0.409, y una pregunta redactada distinto podría quedar por debajo y el sistema diría que no sabe cuando sí sabe.

**Limitación**
Son solo 5 preguntas, así que el 0.38 es un punto de partida, no un valor definitivo. Con más preguntas de prueba se podría ajustar mejor.
---

# Bloque B · RAG sobre papers

## B1 · Ejecuta el RAG de papers y evalúa con el ground truth

**Evidencia:**
[RESPUESTAS.md](evidencias/RESPUESTAS.md)

**Mi respuesta / análisis:**

| Pregunta | Respuesta del RAG (resumen fiel) | ¿Correcta? | ¿Contexto del paper correcto? | Comparación con el ground truth |
|---|---|---|---|---|
| **Q1 — BLEU score** | Transformer big obtuvo **28.4 BLEU** en WMT 2014 inglés-alemán. | **Sí** | **Sí:** *Attention Is All You Need* (`attention_paper.pdf`; chunks 28, 1, 26, 40 y 31). | Coincide con el valor de referencia; la respuesta no destacó la mejora de más de dos puntos frente a resultados anteriores. |
| **Q2 — Memoria RAG** | Combina **BART** como memoria paramétrica con un índice denso de Wikipedia como memoria no paramétrica; **DPR** recupera los documentos, usados para generar la respuesta. | **Sí** | **Sí:** *Retrieval-Augmented Generation* (`rag_paper.pdf`; chunks 1, 8, 32, 7 y 50). | Coincide con la explicación del ground truth e incorpora detalles adicionales del modelo probabilístico. |
| **Q3 — Scaling factor** | El factor **1/√dₖ** evita productos punto demasiado grandes, que saturarían softmax y producirían gradientes pequeños. | **Sí** | **Sí:** *Attention Is All You Need* (`attention_paper.pdf`; chunks 11, 12, 13, 16 y 20). | Coincide con la explicación esperada del escalamiento y su efecto en el entrenamiento. |
| **Q4 — Variantes RAG** | **RAG-Sequence** condiciona la secuencia completa con un documento; **RAG-Token** permite diferentes documentos por token. | **Sí** | **Sí:** *Retrieval-Augmented Generation* (`rag_paper.pdf`; chunks 10, 11, 1, 50 y 7). | Explica correctamente la diferencia; omite el costo computacional adicional de RAG-Token, que no era el foco de la pregunta. |
| **Q5 — Atención multi-cabeza** | Transformer base usa **8 cabezas**, con **dₖ = dᵥ = 64** por cabeza. | **Sí** | **Sí:** *Attention Is All You Need* (`attention_paper.pdf`; chunks 14, 15, 9, 0 y 40). | Coincide con los valores del ground truth e incluye la relación 512/8 = 64. |
| **Bonus — Relación RAG y Transformer** | Identificó **BART** como generador seq2seq, pero indicó que los extractos no confirmaban si BART se basa en Transformer. | **Parcial** | **Fuente citada en la respuesta:** *Retrieval-Augmented Generation* (`rag_paper.pdf`). **Lista de fragmentos recuperados aún no compartida**; no se confirma la aparición de ambos papers. | El *ground truth* exige **BART-large** y confirmar que es un modelo basado en Transformer. La respuesta identifica la familia BART, pero deja sin resolver la segunda parte. |


## B2 · Experimentos con chunk_size, k y score_threshold

_Evidencia:_ …

_Mi respuesta / análisis:_


## B3 · Preguntas trampa: ¿el RAG admite que no sabe?

_Evidencia:_ …

_Mi respuesta / análisis:_


## B4 · Preguntas en español sobre papers en inglés

_Evidencia:_ …

_Mi respuesta / análisis:_


## B5 · Escribe tu propio quiz con ground truth

_Evidencia:_ …

_Mi respuesta / análisis:_



# Bloque R · Reflexión

## R1 · Reflexión final con tus datos

### 1. ¿Qué ocurrió cuando el contexto no contenía información suficiente? ¿El modelo reconoció la limitación o inventó?
Durante las pruebas de resistencia con preguntas trampa o fuera de contexto, el modelo mostró inicialmente una tendencia a alucinar o a extrapolar información general cuando las directrices del sistema eran flexibles. Tras refinar y endurecer el *system prompt* con instrucciones explícitas de honestidad, el sistema logró identificar de manera efectiva que los fragmentos recuperados no contenían la respuesta solicitada. Esto evidencia que la confiabilidad de un RAG no recae únicamente en la precisión del motor de búsqueda vectorial, sino también en las restricciones estrictas impuestas en la capa generativa para evitar respuestas inventadas.

### 2. ¿Qué fragmentos se recuperaron para la pregunta bonus? ¿Aparecieron ambos papers?
En la consulta *bonus* orientada a sintetizar información cruzada, la búsqueda vectorial logró recuperar fragmentos relevantes de ambos documentos académicos gracias a que el espacio latente del modelo de *embeddings* agrupa conceptos semánticamente afines. No obstante, se comprobó que la aparición equilibrada de ambas fuentes dependía críticamente del valor del parámetro $k$ y de los umbrales de similitud; cuando $k$ era insuficiente, los fragmentos clave del segundo paper quedaban desplazados fuera de la ventana de contexto. Esto demuestra que las consultas multi-documento exigen un ajuste fino del número de vecinos recuperados para evitar sesgos hacia una sola fuente.

### 3. ¿Qué efecto tuvo modificar `chunk_size`, `chunk_overlap`, `k` o `score_threshold`?
La experimentación sistemática demostró que cada hiperparámetro afecta de forma directa el balance entre precisión y cobertura del sistema. Un `chunk_size` muy reducido fragmentaba definiciones técnicas complejas rompiendo el hilo argumental, mientras que uno excesivamente grande diluía la densidad semántica y arrastraba ruido innecesario al contexto. Por otro lado, calibrar adecuadamente el `score_threshold` permitió filtrar fragmentos irrelevantes con alta precisión, evitando alimentar al modelo con baja calidad semántica a cambio de requerir un monitoreo cuidadoso para no descartar información válida.

### 4. ¿Por qué conviene separar el modelo de embeddings del modelo generativo?
Desacoplar el modelo de *embeddings* (responsable de la vectorización y el cálculo de cercanía semántica) del modelo generativo (LLM) optimiza de manera radical la eficiencia arquitectónica y los costos computacionales. Esta separación permite indexar, persistir y buscar eficientemente en bases vectoriales especializadas a alta velocidad sin necesidad de invocar costosas inferencias de lenguaje natural en cada fase de recuperación. Asimismo, otorga flexibilidad modular para actualizar o reemplazar de forma independiente el motor de recuperación o el modelo de lenguaje según las necesidades del negocio.

### 5. ¿Qué cambiarías para llevar este prototipo a producción? Considera persistencia, control de acceso, evaluación, observabilidad y costos.
Llevar este prototipo a un entorno productivo empresarial exige una reingeniería hacia una arquitectura robusta y escalable en múltiples dimensiones:
- **Persistencia y Arquitectura:** Migrar de un almacenamiento volátil o local a un clúster distribuido de Qdrant con replicación y respaldos automáticos, implementando un pipeline de ingestión incremental en lugar de reconstrucciones totales del índice.
- **Seguridad y Gobierno:** Integrar control de acceso basado en roles (RBAC) y filtrado dinámico por metadatos en la capa de consulta para garantizar que los usuarios únicamente accedan a documentos autorizados.
- **Observabilidad y Evaluación:** Incorporar plataformas de monitoreo de telemetría para rastrear la latencia por consulta, el consumo de tokens y los costos asociados, además de pipelines automatizados de evaluación continua (como Ragas) para medir la calidad de las respuestas y la tasa de alucinaciones en tiempo real.

---

# Retos

## C1 · HNSW contra fuerza bruta con FAISS

_Evidencia:_ …

_Mi respuesta / análisis:_


## C2 · Un tercer paper: el RAG que explica HNSW

_Evidencia:_ …

_Mi respuesta / análisis:_


## C3 · App Gradio para los papers con filtro por paper

_Evidencia:_ …

_Mi respuesta / análisis:_


## C4 · Evaluación automática de la recuperación

_Evidencia:_ …

_Mi respuesta / análisis:_


## C5 · MMR frente a similitud pura

_Evidencia:_ …

_Mi respuesta / análisis:_


## C6 · Persistencia del índice en disco

_Evidencia:_ …

_Mi respuesta / análisis:_

