# Respuestas del workshop RAG
Escribe aquí tus respuestas con tus propias palabras. Pega tablas, resultados y enlaces a las capturas de `evidencias/`.

---

# Bloque 0 · Teoría

## T1 · Cuántos nodos viven en cada capa

_Evidencia:_ …

_Mi respuesta / análisis:_


## T2 · ef, k y el tamaño de la cola

_Evidencia:_ …

_Mi respuesta / análisis:_


## T3 · Inserción frente a búsqueda

_Evidencia:_ …

_Mi respuesta / análisis:_


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


---

# Bloque A · App RAG empresarial

## A1 · Ejecuta la app RAG de punta a punta

_Evidencia:_ …

_Mi respuesta / análisis:_


## A2 · Tu propio corpus de documentos

_Evidencia:_ …

_Mi respuesta / análisis:_


## A3 · Ingeniería del prompt (antes y después)

_Evidencia:_ …

_Mi respuesta / análisis:_


## A4 · Muestra el score de similitud en las fuentes

_Evidencia:_ …

_Mi respuesta / análisis:_


---

# Bloque B · RAG sobre papers

## B1 · Ejecuta el RAG de papers y evalúa con el ground truth

_Evidencia:_ …

_Mi respuesta / análisis:_


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


---

# Reflexión

## R1 · Reflexión final con tus datos

_Evidencia:_ …

_Mi respuesta / análisis:_


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

