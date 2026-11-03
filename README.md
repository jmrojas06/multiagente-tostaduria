# Arquitectura multiagente: diseño, implementación y medición

Sistema de varios agentes de IA que responde preguntas de negocio y se compara, con modelos reales, contra un agente único. Se mide **calidad, costo, latencia, robustez y costo de coordinación**, para decidir con datos si el multiagente vale la pena.

**Caso guía.** La *Tostaduría Samán* (empresa **ficticia** de Cali) debe decidir su próxima inversión. Los agentes consultan una base de conocimiento interna de 11 documentos y responden citando las fuentes (`[D2]`, `[D7]`...).

> Partimos de los notebooks base del curso (repositorio del profesor `armandoordonez/multiagente`). Este repositorio es nuestra versión ejecutada, corregida y documentada.

## Índice
1. [Contenido del repositorio](#1-contenido-del-repositorio)
2. [Qué hicimos](#2-qué-hicimos)
3. [Cómo funciona el multiagente](#3-cómo-funciona-el-multiagente)
4. [Cómo se comparten información y memoria](#4-cómo-se-comparten-información-y-memoria)
5. [Un ejemplo real, paso a paso](#5-un-ejemplo-real-paso-a-paso)
6. [Cómo se mide](#6-cómo-se-mide)
7. [Resultados](#7-resultados) (incluye el detalle de **tokens, llamadas, latencia y calidad por tarea**)
8. [Conclusiones](#8-conclusiones)
9. [Qué corregimos respecto a los notebooks base](#9-qué-corregimos-respecto-a-los-notebooks-base)
10. [Cómo correrlo](#10-cómo-correrlo)
11. [Limitaciones y pendiente](#11-limitaciones-y-pendiente)

---

## 1. Contenido del repositorio

| Archivo | Qué es |
|---|---|
| `arquitectura_multiagente_langgraph.ipynb` | Versión con **LangGraph**: el flujo entre agentes es un grafo de estados. Ejecutado con modelos reales. |
| `arquitectura_multiagente.ipynb` | Versión **desde cero** (sin frameworks, solo el SDK `openai` apuntando a OpenRouter). Ejecutado con modelos reales. |
| `resultados_langgraph.csv`, `resultados_experimento.csv` | Una fila por tarea (120 cada uno): tokens, llamadas, latencia, calidad, costo, etc. LangGraph y desde cero, respectivamente. |
| `trazas_langgraph.csv`, `trazas.csv` | Una fila por llamada al modelo: agente, modelo, tokens, costo, latencia. |
| `img/` | Gráficas exportadas del notebook de LangGraph (grafos, Gantt, Pareto). |
| `pyproject.toml`, `uv.lock` | Dependencias, gestionadas con `uv`. |

## 2. Qué hicimos

1. **Revisamos** el repositorio del profesor: dos notebooks que implementan el mismo sistema multiagente, uno con LangGraph y otro desde cero.
2. **Los hicimos funcionar**: instalamos las dependencias que faltaban, corregimos celdas rotas y errores propios de DeepSeek (ver [sección 9](#9-qué-corregimos-respecto-a-los-notebooks-base)).
3. **Los ejecutamos con modelos reales** de DeepSeek a través de OpenRouter, con un juez de otra familia (Claude Haiku) para evitar sesgo.
4. **Medimos cinco configuraciones** (A–E) sobre 8 preguntas repetidas 3 veces (24 tareas por configuración, ≈ 1.000 llamadas, ≈ 0,25 USD por notebook).
5. **Analizamos los resultados** y los documentamos aquí, incluidas las limitaciones.

## 3. Cómo funciona el multiagente

```
                 ┌──────────────┐
  pregunta ───▶  │ Orquestador  │  divide en subtareas (JSON)
                 └──────┬───────┘
          ┌─────────────┼─────────────┐        en paralelo
   ┌──────▼─────┐ ┌─────▼──────┐ ┌────▼───────┐
   │Investigador│ │Investigador│ │Investigador│  buscan en el corpus y resumen con citas
   └──────┬─────┘ └─────┬──────┘ └────┬───────┘
          └─────────────┼─────────────┘
                 ┌──────▼───────┐   rechaza   ┌──────────┐
                 │   Redactor   │ ◀────────── │  Crítico │  rúbrica (JSON)
                 └──────┬───────┘ ──────────▶ └──────────┘
                        ▼            propone
                  respuesta final
```

Grafo real generado por LangGraph:

![Grafo multiagente](img/grafo_multiagente.png)

### Los agentes

| Agente | Qué hace | Herramientas | Modelo |
|---|---|---|---|
| **Orquestador** | Divide la pregunta en hasta 3 subtareas independientes (ventas, costos, riesgos...). Responde JSON validado con Pydantic. | ninguna | `deepseek-v4.1-flash` |
| **Investigador** (×N) | Resuelve **una** subtarea: busca en el corpus y entrega hallazgos con cifras y citas `[D#]`. Se ejecutan **en paralelo**. | `buscar_en_corpus` | `deepseek-v4.1-flash` |
| **Redactor** | Con las notas escribe la respuesta final (máx. 200 palabras, con recomendación explícita y citas). | ninguna | `deepseek-v4.1-flash` (C: `deepseek-v4-pro`) |
| **Crítico** | Verifica con una rúbrica: responde la pregunta, recomienda, cita cada cifra, no excede las notas. Devuelve `{aprobado, puntaje, comentarios}`. Si rechaza, el redactor corrige (máx. 2 rondas). | ninguna | `deepseek-v4.1-flash` (C: `deepseek-v4-pro`) |
| **Router** (solo config. E) | Clasifica la pregunta: *simple* → agente único, *compleja* → equipo multiagente. | ninguna | `deepseek-v4.1-flash` |
| **Juez** (solo evaluación) | Califica las respuestas contra puntos clave esperados. | ninguna | `anthropic/claude-haiku-5.5` |

Solo los investigadores pueden buscar. El orquestador, el redactor y el crítico no tienen herramientas: cada agente tiene un prompt **corto y enfocado** y solo lo que necesita.

### Patrones de diseño usados
- **Orquestador–trabajadores:** un agente reparte el trabajo a otros.
- **Paralelización (map-reduce):** los investigadores corren a la vez; en LangGraph se hace con `Send`, y en la versión desde cero con `asyncio.gather`.
- **Evaluador–optimizador:** el crítico evalúa y el redactor mejora el borrador.
- **Enrutamiento (router):** un clasificador barato decide qué arquitectura usar.

### La herramienta de búsqueda
`buscar_en_corpus(consulta)` busca por **palabras clave** (normaliza acentos, quita palabras vacías y puntúa por términos en común) y devuelve hasta 3 documentos con `id`, `título` y `texto`. En un sistema real sería un buscador vectorial o una base de datos. Cada vez que un agente la usa se registra qué documentos consultó, para medir después el trabajo repetido.

### Controles de costo y robustez
- **Tope de gasto por tarea:** 0,05 USD; si una tarea lo supera, se corta la siguiente llamada.
- **Criterio de parada:** `max_pasos` por agente; al agotarse, se fuerza una respuesta sin herramientas. El crítico tiene `max_rondas_critico = 2`.
- **Reintentos** con espera exponencial ante fallos de la API.
- **Degradación elegante:** si el plan del orquestador llega con formato inválido o cortado, se usa **una sola subtarea**; si la evaluación del crítico llega inválida, la respuesta se da por **aprobada** (y queda registrado en la traza). Así un error de formato no tumba la tarea.
- **Concurrencia limitada:** 3 tareas simultáneas (y 4 llamadas simultáneas en la versión desde cero), para no chocar con límites de tasa.

### Las cinco configuraciones

| | Descripción |
|---|---|
| **A** | **Agente único** con todas las herramientas (línea base). |
| **B** | **Multiagente barato:** todos los roles con `deepseek-v4.1-flash`. |
| **C** | **Multiagente mixto:** orquestador, redactor y crítico con el modelo fuerte `deepseek-v4-pro`; investigadores con el barato. |
| **D** | **Multiagente optimizado:** 2 subtareas, 1 ronda de crítico, resúmenes de 80 palabras (menos paralelismo, parada temprana, compresión de contexto). |
| **E** | **Router:** simple → agente único, compleja → equipo optimizado. |

Grafos de las versiones con un agente y con router:

| Agente único | Router |
|---|---|
| ![Grafo agente único](img/grafo_agente_unico.png) | ![Grafo router](img/grafo_router.png) |

## 4. Cómo se comparten información y memoria

**Los agentes no comparten una memoria global.** Es una decisión de diseño: cada agente ve solo lo que necesita, lo que mantiene los contextos cortos y baratos (menos tokens). Hay tres niveles:

1. **Memoria propia de cada agente** = su historial de mensajes (`MessagesState` en LangGraph; la lista `mensajes` en la versión desde cero). Contiene la pregunta, sus búsquedas y los resultados de la herramienta. Se descarta cuando el agente termina. Cada investigador **solo ve su subtarea**, no lo que hacen los demás.
2. **Estado compartido** (solo LangGraph): el `EstadoMulti` es el contrato entre agentes, un diccionario común:

   | Clave | Quién la escribe | Quién la lee |
   |---|---|---|
   | `pregunta` | entrada | todos |
   | `subtareas` | orquestador | reparto (`Send`) |
   | `notas` | **cada investigador** (reductor `operator.add`) | redactor y crítico |
   | `borrador` | redactor | crítico |
   | `comentarios`, `aprobado`, `rondas` | crítico | redactor / control del ciclo |

   Cada nodo lee solo las claves que necesita y devuelve solo las que cambia. El reductor `operator.add` en `notas` es lo que permite que varios investigadores en paralelo escriban a la vez sin pisarse: LangGraph junta sus listas.
   En la versión **desde cero** el mismo trabajo lo hace el código: `asyncio.gather` devuelve las notas de los investigadores y se pasan como texto al redactor.
3. **Lo que no hay:** memoria entre tareas ni entre ejecuciones. Cada pregunta empieza de cero (no usamos `checkpointer`). Agregarla (por ejemplo con `MemorySaver` para humano en el bucle) es el ejercicio 4 del notebook y queda pendiente.

**Qué se "pasan" los agentes:** el orquestador entrega subtareas → los investigadores entregan notas con cifras y citas → el redactor entrega un borrador → el crítico entrega comentarios → el redactor corrige. La comunicación entre ellos es **JSON validado** (orquestador y crítico) o texto con citas (investigadores y redactor), de modo que los errores de formato se detectan en el borde.

## 5. Un ejemplo real, paso a paso

Pregunta: *«¿Qué opción de expansión conviene más si la junta exige recuperar la inversión en menos de 24 meses?»*

**Recorrido del grafo** (`stream_mode="updates"`, un evento por nodo que termina):

```
orquestar    actualizó: subtareas, rondas, comentarios, notas
investigar   actualizó: notas        ← tres investigadores, en paralelo
investigar   actualizó: notas
investigar   actualizó: notas
redactar     actualizó: borrador
criticar     actualizó: aprobado, comentarios, rondas
```

**Línea de tiempo** de esa ejecución. Los tres investigadores corren a la vez; el **redactor es la ruta crítica** (≈ 4 de los 8 s):

![Gantt](img/gantt_multiagente.png)

**Costo por rol** de esa misma tarea:

| Rol | Llamadas | Tokens entrada | Tokens salida | Costo (USD) | % del costo |
|---|---|---|---|---|---|
| investigador | 7 | 5.419 | 1.220 | 0,00074 | 55,9 |
| critico | 1 | 1.326 | 81 | 0,00035 | 26,2 |
| redactor | 1 | 999 | 290 | 0,00017 | 13,0 |
| orquestador | 1 | 118 | 90 | 0,00007 | 5,0 |

**Respuesta del agente único (A)** — 2 llamadas, USD 0,00024, 3,2 s:

> **Recomendación: Opción A (tienda online con suscripciones).** Con el requisito de recuperar la inversión en menos de 24 meses, solo la Opción A cumple: inversión de COP 95 millones y punto de equilibrio estimado en **14 meses** [D7]. La Opción B (segunda tienda física) requiere COP 420 millones y tiene punto de equilibrio en **30 meses** [D8]: excede el plazo. La Opción C (exportación a Chile) exige certificación sanitaria de 6 meses y no reporta punto de equilibrio [D9].

**Respuesta del equipo multiagente (B)** — 10 llamadas, USD 0,00132, 8,2 s:

> La opción que conviene más es la **Opción A (tienda online con suscripciones)**, porque es la única con punto de equilibrio documentado dentro del umbral exigido: inversión de COP 95 millones y recuperación estimada en 14 meses [D7]. La Opción B requiere COP 420 millones, arriendo mensual de COP 11 millones y alcanza el punto de equilibrio en 30 meses, por lo que no cumple el requisito de <24 meses [D8]. […] **Recomendación explícita:** aprobar la Opción A […]; conviene además monitorear los riesgos transversales (concentración en clientes B2B y volatilidad del café verde) [D10].

Las dos llegan a la misma recomendación con las mismas cifras correctas. El equipo agrega el contexto de riesgos (documento D10), pero cuesta ≈ 5,5 veces más y tarda ≈ 2,5 veces más.

**Indicadores de coordinación** de esa tarea:

| | A · agente único | B · multiagente |
|---|---|---|
| Llamadas al modelo | 2 | 10 |
| % del costo en coordinación (orquestador + crítico) | 0 % | 31,2 % |
| Documentos consultados | 6 | 19 |
| % de consultas repetidas | 33,3 % | 68,4 % |

## 6. Cómo se mide

- **Trazas.** Toda llamada al modelo pasa por un único punto (`llamar_llm` en la versión desde cero; un *callback handler* `Trazador` en LangGraph). Cada una deja una fila con agente, modelo, tokens de entrada/salida/caché, **costo real reportado por OpenRouter** (`usage.cost`) y latencia. De ahí salen el costo por rol, el Gantt y los indicadores.
- **Dataset.** 8 preguntas de negocio, cada una con sus **puntos clave esperados** (por ejemplo: «la tienda online recupera la inversión en 14 meses»). Se repiten **3 veces** cada una → 24 tareas por configuración, porque las respuestas de los modelos varían entre ejecuciones.
- **Juez LLM de otra familia** (`anthropic/claude-haiku-5.5`) para no calificar a DeepSeek con DeepSeek. Para cada respuesta indica qué puntos clave contiene, y califica fidelidad y claridad:

  | Criterio | Escala | Peso |
  |---|---|---|
  | Cobertura de puntos clave | fracción de puntos presentes | 60 % |
  | Fidelidad (sin cifras inventadas, con citas) | 1 a 5 | 25 % |
  | Claridad y recomendación explícita | 1 a 5 | 15 % |

  El costo del juez se registra aparte para no mezclarlo con el del sistema evaluado.
- **Métricas reportadas:** calidad (media y desviación estándar), cobertura, % de éxito, costo por tarea y por 1.000 tareas, latencia p50 y p95, llamadas y tokens por tarea, **% del costo en coordinación** (orquestador, crítico, router), **% de consultas repetidas** entre agentes y **caché de prompts**.
- **Frontera de Pareto:** una configuración está en la frontera si ninguna otra es a la vez más barata y de mejor calidad; las demás se pueden descartar.

## 7. Resultados

Modelos reales vía OpenRouter, 24 tareas por configuración, 0 juicios fallidos y 100 % de éxito en todas las configuraciones.

### 7.1 Versión LangGraph (`arquitectura_multiagente_langgraph.ipynb`)

| Config | Calidad (0-1) | ± desv. | Cobertura | Costo/tarea (USD) | USD / 1.000 tareas | Latencia p50 | Latencia p95 | Llamadas | Tokens | % coord. | % repetidas |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A · agente único | **0,905** | 0,108 | 0,95 | **0,00027** | 0,27 | **2,41 s** | 3,75 s | 2,1 | 1.951 | 0 | 33,3 |
| B · multiagente barato | 0,853 | 0,133 | 0,88 | 0,00157 | 1,57 | 7,20 s | 10,14 s | 9,2 | 8.998 | 21,3 | 70,6 |
| C · multiagente mixto | 0,821 | 0,145 | 0,86 | 0,00545 | 5,45 | 13,88 s | 25,45 s | 10,0 | 9.634 | 40,1 | 66,4 |
| D · optimizado | 0,850 | 0,139 | 0,89 | 0,00112 | 1,12 | 6,43 s | 17,54 s | 7,2 | 6.029 | 22,1 | 56,9 |
| E · router | 0,847 | 0,158 | 0,89 | 0,00110 | 1,10 | 6,12 s | 13,85 s | 8,2 | 5.959 | 21,3 | 57,9 |

Frontera de Pareto de A, B y C (el agente único domina: es más barato **y** de mejor calidad):

![Pareto A-B-C](img/pareto_abc.png)

Con las versiones optimizadas D y E:

![Pareto todas](img/pareto_todas.png)

**Efecto de optimizar, frente a B (multiagente sin optimizar):**

| | Δ costo | Δ calidad (puntos) | Δ latencia p50 |
|---|---|---|---|
| D · optimizado | −28,7 % | −0,003 | −10,7 % |
| E · router | −29,9 % | −0,006 | −15,0 % |

**Calidad por pregunta** (media de 3 repeticiones):

| Pregunta | A | B | C |
|---|---|---|---|
| Opción de expansión con límite de 24 meses | 0,96 | 0,94 | 0,84 |
| Canal que más crece y mejor margen | 0,90 | 0,91 | 0,91 |
| Capacidad de la planta para exportar a Chile | 0,84 | 0,78 | 0,77 |
| Riesgos financieros y operativos | 0,98 | 0,90 | 0,93 |
| Evidencia de demanda para suscripción | 1,00 | 0,95 | 0,88 |
| Financiar la segunda tienda sin crédito | 0,92 | 0,76 | 0,70 |
| Precio del café verde y margen B2B | 0,92 | 0,93 | 0,94 |
| Qué mejorar primero en domicilios | 0,70 | 0,66 | 0,60 |

**Caché de prompts** (tokens de entrada que el proveedor reutilizó, lo que abarata el costo):

| Config | Tokens de entrada | Tokens en caché | % en caché |
|---|---|---|---|
| A | 38.384 | 20.608 | 53,7 |
| B | 193.211 | 61.476 | 31,8 |
| C | 189.763 | 65.328 | 34,4 |
| D | 118.894 | 35.078 | 29,5 |
| E | 117.438 | 35.200 | 30,0 |

**Costo real de toda la sesión** (incluye al juez):

| | Llamadas | USD |
|---|---|---|
| A | 52 | 0,00672 |
| B | 242 | 0,04036 |
| C | 240 | 0,13087 |
| D | 174 | 0,02678 |
| E | 196 | 0,02638 |
| Juez | 120 | 0,01825 |
| **Total** | **1.024** | **0,24938** |

### 7.2 Versión desde cero (`arquitectura_multiagente.ipynb`)

Mismo diseño, mismas preguntas, otra implementación. Sirve para comprobar que los resultados no dependen del framework.

| Config | Calidad (0-1) | ± desv. | Cobertura | Costo/tarea (USD) | Latencia p50 | Latencia p95 | Llamadas | % coord. | % repetidas |
|---|---|---|---|---|---|---|---|---|---|
| A · agente único | 0,869 | 0,156 | 0,89 | **0,00028** | **2,55 s** | 4,82 s | 2,2 | 0 | 24,1 |
| B · multiagente barato | **0,894** | 0,092 | 0,94 | 0,00148 | 12,08 s | 19,48 s | 9,3 | 17,7 | 67,9 |
| C · multiagente mixto | 0,741 | 0,245 | 0,78 | 0,00466 | 16,20 s | 27,23 s | 9,6 | 41,8 | 66,7 |
| D · optimizado | 0,866 | 0,131 | 0,89 | 0,00102 | 7,73 s | 16,41 s | 7,3 | 21,0 | 58,1 |
| E · router | 0,873 | 0,141 | 0,94 | 0,00116 | 8,34 s | 14,09 s | 8,4 | 20,5 | 58,2 |

Costo total de la sesión: **0,22635 USD** (1.015 llamadas, incluye al juez).
En C, la pregunta de riesgos sacó 0,29 en promedio, lo que explica su calidad baja y su desviación estándar de 0,245.

> **Media frente a mediana.** En las tablas de abajo la latencia es el **promedio**; en la 7.1 es la **mediana (p50)**, por eso difieren un poco (p. ej. B: 7,3 s de promedio frente a 7,2 s de mediana). Los tokens son entrada + salida de todas las llamadas de la tarea, incluida la caché.

### 7.3 Detalle por tarea: versión LangGraph

Preguntas del dataset:

| | Pregunta |
|---|---|
| P1 | ¿Qué opción de expansión conviene más si la junta exige recuperar la inversión en menos de 24 meses? |
| P2 | ¿Qué canal de venta crece más y cuál deja mejor margen? |
| P3 | ¿La capacidad de la planta alcanza para la exportación a Chile? |
| P4 | ¿Qué riesgos financieros y operativos enfrenta la empresa? |
| P5 | ¿Hay evidencia de demanda para un modelo de suscripción? |
| P6 | ¿Puede la empresa financiar la segunda tienda sin crédito? |
| P7 | ¿Cómo afecta el precio del café verde la rentabilidad del canal B2B? |
| P8 | ¿Qué debería mejorar primero en domicilios antes de lanzar una tienda online? |

Cada celda es el **promedio de las 3 repeticiones** de esa pregunta. La fila «Todas» promedia las 24 tareas.

**Tokens por tarea (entrada + salida)**

| Pregunta | A | B | C | D | E |
|---|---|---|---|---|---|
| P1 | 1.819 | 9.653 | 9.658 | 7.427 | 6.244 |
| P2 | 1.798 | 9.051 | 8.565 | 5.233 | 5.345 |
| P3 | 1.754 | 7.597 | 10.228 | 5.621 | 6.056 |
| P4 | 1.796 | 9.392 | 11.007 | 6.491 | 5.758 |
| P5 | 1.787 | 8.351 | 9.084 | 5.337 | 5.931 |
| P6 | 1.757 | 9.727 | 9.810 | 5.997 | 5.785 |
| P7 | 1.804 | 8.403 | 8.982 | 5.554 | 5.437 |
| P8 | 3.097 | 9.810 | 9.740 | 6.571 | 7.119 |
| **Todas** | **1.951** | **8.998** | **9.634** | **6.029** | **5.959** |

**Llamadas al modelo por tarea**

| Pregunta | A | B | C | D | E |
|---|---|---|---|---|---|
| P1 | 2,0 | 9,7 | 9,7 | 7,7 | 8,3 |
| P2 | 2,0 | 9,0 | 9,0 | 7,0 | 8,0 |
| P3 | 2,0 | 9,0 | 11,0 | 7,3 | 8,3 |
| P4 | 2,0 | 9,0 | 11,0 | 7,3 | 8,0 |
| P5 | 2,0 | 9,0 | 9,7 | 7,0 | 8,3 |
| P6 | 2,0 | 9,7 | 10,3 | 7,3 | 8,0 |
| P7 | 2,0 | 9,0 | 9,7 | 7,0 | 8,0 |
| P8 | 2,7 | 9,7 | 9,7 | 7,3 | 8,3 |
| **Todas** | **2,1** | **9,2** | **10,0** | **7,2** | **8,2** |

**Latencia por tarea (promedio)**

| Pregunta | A | B | C | D | E |
|---|---|---|---|---|---|
| P1 | 3,4 s | 7,6 s | 11,1 s | 10,3 s | 8,7 s |
| P2 | 2,7 s | 7,0 s | 12,8 s | 5,2 s | 4,8 s |
| P3 | 1,9 s | 6,9 s | 20,3 s | 10,9 s | 6,2 s |
| P4 | 2,4 s | 9,9 s | 18,9 s | 10,9 s | 9,7 s |
| P5 | 2,1 s | 5,4 s | 12,3 s | 5,7 s | 9,6 s |
| P6 | 3,7 s | 7,9 s | 13,8 s | 9,0 s | 5,8 s |
| P7 | 1,9 s | 6,5 s | 16,1 s | 6,2 s | 5,6 s |
| P8 | 2,4 s | 6,9 s | 17,2 s | 6,0 s | 5,8 s |
| **Todas** | **2,6 s** | **7,3 s** | **15,3 s** | **8,0 s** | **7,0 s** |

**Calidad por tarea (0-1, según el juez)**

| Pregunta | A | B | C | D | E |
|---|---|---|---|---|---|
| P1 | 0,96 | 0,94 | 0,84 | 0,79 | 0,85 |
| P2 | 0,90 | 0,91 | 0,91 | 0,85 | 0,95 |
| P3 | 0,84 | 0,78 | 0,77 | 0,78 | 0,71 |
| P4 | 0,98 | 0,90 | 0,93 | 0,90 | 0,81 |
| P5 | 1,00 | 0,95 | 0,88 | 0,96 | 0,95 |
| P6 | 0,92 | 0,76 | 0,70 | 0,90 | 0,91 |
| P7 | 0,92 | 0,93 | 0,94 | 0,95 | 0,90 |
| P8 | 0,70 | 0,66 | 0,60 | 0,68 | 0,69 |
| **Todas** | **0,90** | **0,85** | **0,82** | **0,85** | **0,85** |

**Costo por tarea (USD)**

| Pregunta | A | B | C | D | E |
|---|---|---|---|---|---|
| P1 | 0,00024 | 0,00176 | 0,00553 | 0,00183 | 0,00111 |
| P2 | 0,00020 | 0,00157 | 0,00498 | 0,00085 | 0,00085 |
| P3 | 0,00023 | 0,00116 | 0,00647 | 0,00087 | 0,00119 |
| P4 | 0,00027 | 0,00157 | 0,00692 | 0,00129 | 0,00109 |
| P5 | 0,00025 | 0,00179 | 0,00331 | 0,00087 | 0,00091 |
| P6 | 0,00027 | 0,00198 | 0,00641 | 0,00085 | 0,00127 |
| P7 | 0,00037 | 0,00117 | 0,00543 | 0,00103 | 0,00102 |
| P8 | 0,00032 | 0,00153 | 0,00458 | 0,00135 | 0,00136 |
| **Todas** | **0,00027** | **0,00157** | **0,00545** | **0,00112** | **0,00110** |

<details>
<summary>Las 120 tareas una por una (versión LangGraph)</summary>

| # | Config | Pregunta | Tokens | Llamadas | Latencia | Calidad | Cobertura | Fidelidad | Claridad | Costo (USD) | % coord. | % repetidas |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | P1 | 1.859 | 2 | 4,2 s | 0,94 | 1,00 | 4 | 5 | 0,00025 | 0,0 | 50,0 |
| 2 | A | P1 | 1.786 | 2 | 3,3 s | 1,00 | 1,00 | 5 | 5 | 0,00023 | 0,0 | 50,0 |
| 3 | A | P1 | 1.811 | 2 | 2,8 s | 0,94 | 1,00 | 4 | 5 | 0,00025 | 0,0 | 50,0 |
| 4 | A | P2 | 1.806 | 2 | 2,8 s | 0,94 | 1,00 | 4 | 5 | 0,00023 | 0,0 | 33,3 |
| 5 | A | P2 | 1.807 | 2 | 2,8 s | 0,84 | 1,00 | 3 | 4 | 0,00020 | 0,0 | 33,3 |
| 6 | A | P2 | 1.782 | 2 | 2,5 s | 0,94 | 1,00 | 4 | 5 | 0,00019 | 0,0 | 33,3 |
| 7 | A | P3 | 1.874 | 2 | 2,2 s | 0,90 | 1,00 | 4 | 4 | 0,00027 | 0,0 | 33,3 |
| 8 | A | P3 | 1.699 | 2 | 1,8 s | 0,69 | 0,75 | 3 | 4 | 0,00022 | 0,0 | 40,0 |
| 9 | A | P3 | 1.688 | 2 | 1,8 s | 0,94 | 1,00 | 4 | 5 | 0,00020 | 0,0 | 40,0 |
| 10 | A | P4 | 1.590 | 2 | 2,1 s | 0,94 | 1,00 | 4 | 5 | 0,00025 | 0,0 | 0,0 |
| 11 | A | P4 | 1.837 | 2 | 3,0 s | 1,00 | 1,00 | 5 | 5 | 0,00029 | 0,0 | 0,0 |
| 12 | A | P4 | 1.960 | 2 | 2,2 s | 1,00 | 1,00 | 5 | 5 | 0,00028 | 0,0 | 16,7 |
| 13 | A | P5 | 1.845 | 2 | 2,0 s | 1,00 | 1,00 | 5 | 5 | 0,00025 | 0,0 | 33,3 |
| 14 | A | P5 | 1.785 | 2 | 2,3 s | 1,00 | 1,00 | 5 | 5 | 0,00026 | 0,0 | 20,0 |
| 15 | A | P5 | 1.732 | 2 | 1,9 s | 1,00 | 1,00 | 5 | 5 | 0,00025 | 0,0 | 40,0 |
| 16 | A | P6 | 1.762 | 2 | 3,7 s | 0,94 | 1,00 | 4 | 5 | 0,00029 | 0,0 | 16,7 |
| 17 | A | P6 | 1.757 | 2 | 3,8 s | 1,00 | 1,00 | 5 | 5 | 0,00024 | 0,0 | 16,7 |
| 18 | A | P6 | 1.751 | 2 | 3,6 s | 0,84 | 1,00 | 3 | 4 | 0,00028 | 0,0 | 16,7 |
| 19 | A | P7 | 1.787 | 2 | 1,9 s | 0,94 | 1,00 | 4 | 5 | 0,00036 | 0,0 | 50,0 |
| 20 | A | P7 | 1.800 | 2 | 2,0 s | 1,00 | 1,00 | 5 | 5 | 0,00037 | 0,0 | 50,0 |
| 21 | A | P7 | 1.825 | 2 | 2,0 s | 0,84 | 1,00 | 3 | 4 | 0,00039 | 0,0 | 33,3 |
| 22 | A | P8 | 3.680 | 3 | 2,7 s | 0,74 | 0,67 | 4 | 5 | 0,00033 | 0,0 | 50,0 |
| 23 | A | P8 | 3.684 | 3 | 2,8 s | 0,74 | 0,67 | 4 | 5 | 0,00035 | 0,0 | 58,3 |
| 24 | A | P8 | 1.926 | 2 | 1,9 s | 0,64 | 0,67 | 3 | 4 | 0,00027 | 0,0 | 33,3 |
| 25 | B | P1 | 10.628 | 10 | 8,8 s | 0,94 | 1,00 | 4 | 5 | 0,00183 | 11,7 | 70,8 |
| 26 | B | P1 | 8.707 | 9 | 7,1 s | 0,94 | 1,00 | 4 | 5 | 0,00125 | 37,5 | 64,7 |
| 27 | B | P1 | 9.625 | 10 | 6,8 s | 0,94 | 1,00 | 4 | 5 | 0,00221 | 16,6 | 61,1 |
| 28 | B | P2 | 8.835 | 9 | 10,4 s | 0,90 | 1,00 | 4 | 4 | 0,00140 | 20,1 | 71,4 |
| 29 | B | P2 | 8.796 | 9 | 5,5 s | 0,90 | 1,00 | 4 | 4 | 0,00162 | 16,9 | 76,2 |
| 30 | B | P2 | 9.523 | 9 | 5,0 s | 0,94 | 1,00 | 4 | 5 | 0,00170 | 26,9 | 77,8 |
| 31 | B | P3 | 6.974 | 9 | 4,9 s | 0,75 | 0,75 | 4 | 4 | 0,00096 | 16,3 | 58,3 |
| 32 | B | P3 | 7.663 | 9 | 8,2 s | 0,84 | 1,00 | 3 | 4 | 0,00130 | 16,6 | 57,1 |
| 33 | B | P3 | 8.153 | 9 | 7,8 s | 0,75 | 0,75 | 4 | 4 | 0,00123 | 14,6 | 61,1 |
| 34 | B | P4 | 9.699 | 9 | 15,7 s | 0,70 | 0,67 | 4 | 4 | 0,00208 | 17,7 | 70,4 |
| 35 | B | P4 | 9.632 | 9 | 8,2 s | 1,00 | 1,00 | 5 | 5 | 0,00158 | 35,6 | 72,0 |
| 36 | B | P4 | 8.846 | 9 | 5,8 s | 1,00 | 1,00 | 5 | 5 | 0,00104 | 15,1 | 61,9 |
| 37 | B | P5 | 8.176 | 9 | 4,7 s | 0,90 | 1,00 | 4 | 4 | 0,00230 | 15,5 | 70,0 |
| 38 | B | P5 | 8.515 | 9 | 4,2 s | 1,00 | 1,00 | 5 | 5 | 0,00147 | 10,9 | 73,9 |
| 39 | B | P5 | 8.362 | 9 | 7,3 s | 0,94 | 1,00 | 4 | 5 | 0,00159 | 35,3 | 68,4 |
| 40 | B | P6 | 8.075 | 9 | 8,4 s | 0,64 | 0,67 | 3 | 4 | 0,00144 | 28,4 | 64,7 |
| 41 | B | P6 | 12.486 | 11 | 8,0 s | 0,70 | 0,67 | 4 | 4 | 0,00285 | 14,7 | 72,4 |
| 42 | B | P6 | 8.619 | 9 | 7,5 s | 0,94 | 1,00 | 4 | 5 | 0,00164 | 18,9 | 61,9 |
| 43 | B | P7 | 8.447 | 9 | 5,9 s | 0,90 | 1,00 | 4 | 4 | 0,00149 | 32,4 | 81,0 |
| 44 | B | P7 | 7.780 | 9 | 6,7 s | 0,90 | 1,00 | 4 | 4 | 0,00102 | 13,2 | 83,3 |
| 45 | B | P7 | 8.982 | 9 | 7,1 s | 1,00 | 1,00 | 5 | 5 | 0,00099 | 24,4 | 83,3 |
| 46 | B | P8 | 10.332 | 10 | 8,5 s | 0,50 | 0,33 | 4 | 4 | 0,00185 | 22,3 | 78,3 |
| 47 | B | P8 | 9.416 | 9 | 7,3 s | 0,74 | 0,67 | 4 | 5 | 0,00168 | 33,9 | 76,9 |
| 48 | B | P8 | 9.682 | 10 | 4,9 s | 0,74 | 0,67 | 4 | 5 | 0,00108 | 16,9 | 78,3 |
| 49 | C | P1 | 10.742 | 10 | 10,5 s | 0,94 | 1,00 | 4 | 5 | 0,00582 | 49,6 | 66,7 |
| 50 | C | P1 | 9.738 | 10 | 13,2 s | 0,94 | 1,00 | 4 | 5 | 0,00586 | 56,3 | 55,6 |
| 51 | C | P1 | 8.495 | 9 | 9,7 s | 0,64 | 0,50 | 4 | 5 | 0,00491 | 33,3 | 68,8 |
| 52 | C | P2 | 8.194 | 9 | 13,2 s | 0,94 | 1,00 | 4 | 5 | 0,00474 | 53,5 | 72,2 |
| 53 | C | P2 | 8.568 | 9 | 12,5 s | 0,90 | 1,00 | 4 | 4 | 0,00500 | 28,7 | 81,0 |
| 54 | C | P2 | 8.932 | 9 | 12,7 s | 0,90 | 1,00 | 4 | 4 | 0,00521 | 52,4 | 76,2 |
| 55 | C | P3 | 7.482 | 9 | 14,8 s | 0,79 | 0,75 | 4 | 5 | 0,00478 | 31,3 | 66,7 |
| 56 | C | P3 | 11.736 | 12 | 19,7 s | 0,69 | 0,75 | 3 | 4 | 0,01053 | 24,8 | 58,3 |
| 57 | C | P3 | 11.467 | 12 | 26,3 s | 0,84 | 1,00 | 3 | 4 | 0,00410 | 31,3 | 70,6 |
| 58 | C | P4 | 9.684 | 10 | 17,8 s | 0,94 | 1,00 | 4 | 5 | 0,00602 | 48,3 | 57,9 |
| 59 | C | P4 | 13.827 | 13 | 25,9 s | 0,90 | 1,00 | 4 | 4 | 0,00890 | 27,3 | 53,3 |
| 60 | C | P4 | 9.511 | 10 | 12,9 s | 0,94 | 1,00 | 4 | 5 | 0,00584 | 49,7 | 46,2 |
| 61 | C | P5 | 7.681 | 9 | 11,6 s | 0,90 | 1,00 | 4 | 4 | 0,00268 | 28,1 | 53,3 |
| 62 | C | P5 | 10.141 | 10 | 14,7 s | 0,74 | 0,67 | 4 | 5 | 0,00387 | 39,6 | 74,1 |
| 63 | C | P5 | 9.431 | 10 | 10,7 s | 1,00 | 1,00 | 5 | 5 | 0,00339 | 42,5 | 65,0 |
| 64 | C | P6 | 12.037 | 12 | 15,6 s | 0,64 | 0,67 | 3 | 4 | 0,00974 | 31,6 | 57,9 |
| 65 | C | P6 | 8.090 | 9 | 12,9 s | 0,84 | 1,00 | 3 | 4 | 0,00514 | 38,0 | 56,2 |
| 66 | C | P6 | 9.303 | 10 | 13,0 s | 0,64 | 0,67 | 3 | 4 | 0,00436 | 39,1 | 66,7 |
| 67 | C | P7 | 8.124 | 9 | 14,5 s | 0,94 | 1,00 | 4 | 5 | 0,00316 | 55,6 | 77,8 |
| 68 | C | P7 | 7.987 | 9 | 11,0 s | 0,94 | 1,00 | 4 | 5 | 0,00456 | 37,9 | 77,8 |
| 69 | C | P7 | 10.834 | 11 | 22,6 s | 0,94 | 1,00 | 4 | 5 | 0,00856 | 37,3 | 77,8 |
| 70 | C | P8 | 9.983 | 10 | 16,8 s | 0,74 | 0,67 | 4 | 5 | 0,00491 | 39,6 | 77,3 |
| 71 | C | P8 | 8.050 | 9 | 17,4 s | 0,64 | 0,67 | 3 | 4 | 0,00356 | 50,4 | 53,3 |
| 72 | C | P8 | 11.188 | 10 | 17,3 s | 0,44 | 0,33 | 3 | 4 | 0,00526 | 36,2 | 83,3 |
| 73 | D | P1 | 7.531 | 8 | 8,1 s | 0,79 | 0,75 | 4 | 5 | 0,00163 | 21,5 | 50,0 |
| 74 | D | P1 | 6.735 | 7 | 6,0 s | 0,90 | 1,00 | 4 | 4 | 0,00168 | 20,8 | 44,4 |
| 75 | D | P1 | 8.014 | 8 | 16,8 s | 0,69 | 0,75 | 3 | 4 | 0,00219 | 16,6 | 55,6 |
| 76 | D | P2 | 5.402 | 7 | 4,1 s | 0,64 | 0,67 | 3 | 4 | 0,00090 | 12,1 | 58,3 |
| 77 | D | P2 | 5.111 | 7 | 7,0 s | 1,00 | 1,00 | 5 | 5 | 0,00088 | 8,0 | 58,3 |
| 78 | D | P2 | 5.187 | 7 | 4,4 s | 0,90 | 1,00 | 4 | 4 | 0,00077 | 12,8 | 58,3 |
| 79 | D | P3 | 4.909 | 7 | 17,7 s | 0,90 | 1,00 | 4 | 4 | 0,00075 | 19,6 | 37,5 |
| 80 | D | P3 | 6.612 | 8 | 7,8 s | 0,90 | 1,00 | 4 | 4 | 0,00118 | 12,8 | 58,3 |
| 81 | D | P3 | 5.341 | 7 | 7,1 s | 0,54 | 0,50 | 3 | 4 | 0,00066 | 16,6 | 44,4 |
| 82 | D | P4 | 7.796 | 8 | 21,4 s | 0,70 | 0,67 | 4 | 4 | 0,00161 | 12,3 | 65,0 |
| 83 | D | P4 | 5.513 | 7 | 5,6 s | 1,00 | 1,00 | 5 | 5 | 0,00086 | 17,7 | 50,0 |
| 84 | D | P4 | 6.163 | 7 | 5,6 s | 1,00 | 1,00 | 5 | 5 | 0,00138 | 11,0 | 60,0 |
| 85 | D | P5 | 5.782 | 7 | 5,8 s | 1,00 | 1,00 | 5 | 5 | 0,00071 | 38,8 | 66,7 |
| 86 | D | P5 | 5.221 | 7 | 4,9 s | 0,94 | 1,00 | 4 | 5 | 0,00108 | 23,3 | 54,5 |
| 87 | D | P5 | 5.007 | 7 | 6,4 s | 0,94 | 1,00 | 4 | 5 | 0,00082 | 43,4 | 58,3 |
| 88 | D | P6 | 5.898 | 7 | 6,4 s | 0,90 | 1,00 | 4 | 4 | 0,00081 | 47,5 | 53,3 |
| 89 | D | P6 | 6.890 | 8 | 15,6 s | 0,90 | 1,00 | 4 | 4 | 0,00090 | 15,2 | 50,0 |
| 90 | D | P6 | 5.203 | 7 | 5,0 s | 0,90 | 1,00 | 4 | 4 | 0,00084 | 12,8 | 36,4 |
| 91 | D | P7 | 5.554 | 7 | 3,8 s | 1,00 | 1,00 | 5 | 5 | 0,00088 | 38,7 | 66,7 |
| 92 | D | P7 | 5.499 | 7 | 7,8 s | 0,94 | 1,00 | 4 | 5 | 0,00098 | 31,5 | 66,7 |
| 93 | D | P7 | 5.609 | 7 | 6,9 s | 0,90 | 1,00 | 4 | 4 | 0,00122 | 27,2 | 73,3 |
| 94 | D | P8 | 5.720 | 7 | 6,2 s | 0,64 | 0,67 | 3 | 4 | 0,00153 | 29,0 | 58,3 |
| 95 | D | P8 | 7.725 | 8 | 6,7 s | 0,70 | 0,67 | 4 | 4 | 0,00159 | 26,5 | 68,2 |
| 96 | D | P8 | 6.267 | 7 | 5,1 s | 0,70 | 0,67 | 4 | 4 | 0,00093 | 15,8 | 72,2 |
| 97 | E | P1 | 5.788 | 8 | 6,3 s | 0,94 | 1,00 | 4 | 5 | 0,00137 | 11,1 | 33,3 |
| 98 | E | P1 | 7.082 | 9 | 8,0 s | 0,94 | 1,00 | 4 | 5 | 0,00080 | 20,6 | 46,7 |
| 99 | E | P1 | 5.863 | 8 | 12,0 s | 0,69 | 0,75 | 3 | 4 | 0,00114 | 17,2 | 41,7 |
| 100 | E | P2 | 5.242 | 8 | 4,3 s | 1,00 | 1,00 | 5 | 5 | 0,00083 | 18,9 | 66,7 |
| 101 | E | P2 | 5.501 | 8 | 4,6 s | 0,84 | 1,00 | 3 | 4 | 0,00093 | 29,1 | 66,7 |
| 102 | E | P2 | 5.291 | 8 | 5,5 s | 1,00 | 1,00 | 5 | 5 | 0,00078 | 18,6 | 66,7 |
| 103 | E | P3 | 5.376 | 8 | 5,5 s | 0,54 | 0,50 | 3 | 4 | 0,00150 | 20,9 | 44,4 |
| 104 | E | P3 | 5.319 | 8 | 6,1 s | 0,69 | 0,75 | 3 | 4 | 0,00098 | 25,6 | 66,7 |
| 105 | E | P3 | 7.474 | 9 | 6,9 s | 0,90 | 1,00 | 4 | 4 | 0,00110 | 14,8 | 68,8 |
| 106 | E | P4 | 5.693 | 8 | 9,9 s | 0,44 | 0,33 | 3 | 4 | 0,00078 | 21,5 | 54,5 |
| 107 | E | P4 | 6.179 | 8 | 4,9 s | 1,00 | 1,00 | 5 | 5 | 0,00119 | 25,3 | 66,7 |
| 108 | E | P4 | 5.402 | 8 | 14,2 s | 1,00 | 1,00 | 5 | 5 | 0,00129 | 23,0 | 40,0 |
| 109 | E | P5 | 5.971 | 8 | 8,2 s | 1,00 | 1,00 | 5 | 5 | 0,00084 | 18,1 | 60,0 |
| 110 | E | P5 | 6.414 | 9 | 6,4 s | 0,90 | 1,00 | 4 | 4 | 0,00079 | 18,4 | 50,0 |
| 111 | E | P5 | 5.408 | 8 | 14,2 s | 0,94 | 1,00 | 4 | 5 | 0,00110 | 23,7 | 50,0 |
| 112 | E | P6 | 5.922 | 8 | 6,2 s | 0,90 | 1,00 | 4 | 4 | 0,00106 | 15,8 | 60,0 |
| 113 | E | P6 | 5.669 | 8 | 5,5 s | 0,94 | 1,00 | 4 | 5 | 0,00138 | 24,0 | 46,2 |
| 114 | E | P6 | 5.765 | 8 | 5,8 s | 0,90 | 1,00 | 4 | 4 | 0,00137 | 22,9 | 41,7 |
| 115 | E | P7 | 5.458 | 8 | 5,4 s | 0,84 | 1,00 | 3 | 4 | 0,00157 | 13,1 | 66,7 |
| 116 | E | P7 | 5.475 | 8 | 6,7 s | 0,94 | 1,00 | 4 | 5 | 0,00076 | 32,1 | 66,7 |
| 117 | E | P7 | 5.378 | 8 | 4,6 s | 0,94 | 1,00 | 4 | 5 | 0,00073 | 32,0 | 75,0 |
| 118 | E | P8 | 7.648 | 9 | 5,7 s | 0,70 | 0,67 | 4 | 4 | 0,00123 | 13,2 | 63,2 |
| 119 | E | P8 | 6.684 | 8 | 5,5 s | 0,74 | 0,67 | 4 | 5 | 0,00140 | 23,5 | 68,4 |
| 120 | E | P8 | 7.025 | 8 | 6,2 s | 0,64 | 0,67 | 3 | 4 | 0,00145 | 27,4 | 79,2 |

</details>

### 7.4 Detalle por tarea: versión desde cero

Preguntas del dataset:

| | Pregunta |
|---|---|
| P1 | ¿Qué opción de expansión conviene más si la junta exige recuperar la inversión en menos de 24 meses? |
| P2 | ¿Qué canal de venta crece más y cuál deja mejor margen? |
| P3 | ¿La capacidad de la planta alcanza para la exportación a Chile? |
| P4 | ¿Qué riesgos financieros y operativos enfrenta la empresa? |
| P5 | ¿Hay evidencia de demanda para un modelo de suscripción? |
| P6 | ¿Puede la empresa financiar la segunda tienda sin crédito? |
| P7 | ¿Cómo afecta el precio del café verde la rentabilidad del canal B2B? |
| P8 | ¿Qué debería mejorar primero en domicilios antes de lanzar una tienda online? |

Cada celda es el **promedio de las 3 repeticiones** de esa pregunta. La fila «Todas» promedia las 24 tareas.

**Tokens por tarea (entrada + salida)**

| Pregunta | A | B | C | D | E |
|---|---|---|---|---|---|
| P1 | 2.080 | 8.956 | 9.143 | 6.659 | 7.633 |
| P2 | 1.811 | 9.182 | 8.598 | 5.316 | 5.573 |
| P3 | 1.662 | 8.121 | 8.276 | 5.499 | 7.018 |
| P4 | 2.093 | 8.918 | 11.561 | 6.401 | 6.728 |
| P5 | 1.737 | 9.210 | 9.697 | 6.330 | 5.927 |
| P6 | 1.777 | 10.193 | 7.709 | 6.486 | 6.752 |
| P7 | 1.902 | 8.861 | 9.388 | 5.401 | 5.553 |
| P8 | 3.064 | 9.780 | 9.320 | 7.064 | 6.485 |
| **Todas** | **2.016** | **9.152** | **9.211** | **6.144** | **6.459** |

**Llamadas al modelo por tarea**

| Pregunta | A | B | C | D | E |
|---|---|---|---|---|---|
| P1 | 2,3 | 9,3 | 9,3 | 7,3 | 8,7 |
| P2 | 2,0 | 9,0 | 9,0 | 7,0 | 8,0 |
| P3 | 2,0 | 9,3 | 9,7 | 7,0 | 9,0 |
| P4 | 2,3 | 9,0 | 10,7 | 7,7 | 8,7 |
| P5 | 2,0 | 9,3 | 10,0 | 7,3 | 8,0 |
| P6 | 2,0 | 10,0 | 9,0 | 7,7 | 8,7 |
| P7 | 2,0 | 9,3 | 9,7 | 7,0 | 8,0 |
| P8 | 2,7 | 9,3 | 9,7 | 7,3 | 8,0 |
| **Todas** | **2,2** | **9,3** | **9,6** | **7,3** | **8,4** |

**Latencia por tarea (promedio)**

| Pregunta | A | B | C | D | E |
|---|---|---|---|---|---|
| P1 | 4,2 s | 9,9 s | 16,7 s | 12,7 s | 12,3 s |
| P2 | 4,0 s | 14,7 s | 25,7 s | 8,6 s | 6,3 s |
| P3 | 2,0 s | 9,2 s | 18,3 s | 10,1 s | 11,3 s |
| P4 | 2,6 s | 13,7 s | 19,7 s | 12,2 s | 17,0 s |
| P5 | 2,3 s | 10,8 s | 14,1 s | 6,8 s | 11,2 s |
| P6 | 3,9 s | 19,3 s | 14,2 s | 7,8 s | 6,7 s |
| P7 | 2,1 s | 12,9 s | 21,5 s | 4,5 s | 6,0 s |
| P8 | 2,4 s | 12,2 s | 19,3 s | 10,4 s | 8,5 s |
| **Todas** | **2,9 s** | **12,8 s** | **18,7 s** | **9,1 s** | **9,9 s** |

**Calidad por tarea (0-1, según el juez)**

| Pregunta | A | B | C | D | E |
|---|---|---|---|---|---|
| P1 | 0,91 | 0,94 | 0,90 | 0,94 | 0,85 |
| P2 | 0,93 | 0,90 | 0,90 | 0,96 | 0,97 |
| P3 | 0,79 | 0,80 | 0,76 | 0,83 | 0,81 |
| P4 | 1,00 | 0,90 | 0,29 | 0,90 | 0,95 |
| P5 | 0,98 | 1,00 | 0,94 | 0,89 | 0,94 |
| P6 | 0,87 | 0,90 | 0,77 | 0,72 | 0,81 |
| P7 | 0,92 | 0,91 | 0,90 | 0,98 | 0,90 |
| P8 | 0,55 | 0,80 | 0,46 | 0,70 | 0,75 |
| **Todas** | **0,87** | **0,89** | **0,74** | **0,87** | **0,87** |

**Costo por tarea (USD)**

| Pregunta | A | B | C | D | E |
|---|---|---|---|---|---|
| P1 | 0,00026 | 0,00171 | 0,00413 | 0,00100 | 0,00148 |
| P2 | 0,00022 | 0,00122 | 0,00395 | 0,00088 | 0,00098 |
| P3 | 0,00023 | 0,00144 | 0,00429 | 0,00095 | 0,00132 |
| P4 | 0,00029 | 0,00140 | 0,00753 | 0,00124 | 0,00157 |
| P5 | 0,00022 | 0,00141 | 0,00396 | 0,00106 | 0,00101 |
| P6 | 0,00027 | 0,00147 | 0,00364 | 0,00097 | 0,00107 |
| P7 | 0,00043 | 0,00143 | 0,00529 | 0,00108 | 0,00098 |
| P8 | 0,00029 | 0,00172 | 0,00451 | 0,00098 | 0,00091 |
| **Todas** | **0,00028** | **0,00148** | **0,00466** | **0,00102** | **0,00116** |

<details>
<summary>Las 120 tareas una por una (versión desde cero)</summary>

| # | Config | Pregunta | Tokens | Llamadas | Latencia | Calidad | Cobertura | Fidelidad | Claridad | Costo (USD) | % coord. | % repetidas |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | P1 | 1.584 | 2 | 3,6 s | 1,00 | 1,00 | 5 | 5 | 0,00021 | 0,0 | 0,0 |
| 2 | A | P1 | 2.976 | 3 | 4,9 s | 0,79 | 0,75 | 4 | 5 | 0,00032 | 0,0 | 37,5 |
| 3 | A | P1 | 1.679 | 2 | 4,0 s | 0,94 | 1,00 | 4 | 5 | 0,00026 | 0,0 | 0,0 |
| 4 | A | P2 | 1.821 | 2 | 2,9 s | 0,94 | 1,00 | 4 | 5 | 0,00022 | 0,0 | 33,3 |
| 5 | A | P2 | 1.823 | 2 | 5,2 s | 0,90 | 1,00 | 4 | 4 | 0,00023 | 0,0 | 33,3 |
| 6 | A | P2 | 1.789 | 2 | 3,8 s | 0,94 | 1,00 | 4 | 5 | 0,00021 | 0,0 | 33,3 |
| 7 | A | P3 | 1.637 | 2 | 1,9 s | 0,84 | 1,00 | 3 | 4 | 0,00022 | 0,0 | 25,0 |
| 8 | A | P3 | 1.674 | 2 | 2,0 s | 0,79 | 0,75 | 4 | 5 | 0,00023 | 0,0 | 25,0 |
| 9 | A | P3 | 1.676 | 2 | 2,3 s | 0,75 | 0,75 | 4 | 4 | 0,00023 | 0,0 | 25,0 |
| 10 | A | P4 | 1.604 | 2 | 2,0 s | 1,00 | 1,00 | 5 | 5 | 0,00025 | 0,0 | 0,0 |
| 11 | A | P4 | 1.858 | 2 | 2,3 s | 1,00 | 1,00 | 5 | 5 | 0,00029 | 0,0 | 0,0 |
| 12 | A | P4 | 2.817 | 3 | 3,5 s | 1,00 | 1,00 | 5 | 5 | 0,00033 | 0,0 | 42,9 |
| 13 | A | P5 | 1.743 | 2 | 2,3 s | 0,94 | 1,00 | 4 | 5 | 0,00022 | 0,0 | 20,0 |
| 14 | A | P5 | 1.740 | 2 | 2,6 s | 1,00 | 1,00 | 5 | 5 | 0,00023 | 0,0 | 20,0 |
| 15 | A | P5 | 1.728 | 2 | 2,2 s | 1,00 | 1,00 | 5 | 5 | 0,00022 | 0,0 | 20,0 |
| 16 | A | P6 | 1.777 | 2 | 3,4 s | 0,94 | 1,00 | 4 | 5 | 0,00026 | 0,0 | 16,7 |
| 17 | A | P6 | 1.769 | 2 | 4,1 s | 0,94 | 1,00 | 4 | 5 | 0,00028 | 0,0 | 16,7 |
| 18 | A | P6 | 1.784 | 2 | 4,2 s | 0,74 | 0,67 | 4 | 5 | 0,00025 | 0,0 | 16,7 |
| 19 | A | P7 | 1.902 | 2 | 2,0 s | 0,84 | 1,00 | 3 | 4 | 0,00041 | 0,0 | 33,3 |
| 20 | A | P7 | 1.922 | 2 | 2,4 s | 1,00 | 1,00 | 5 | 5 | 0,00043 | 0,0 | 33,3 |
| 21 | A | P7 | 1.881 | 2 | 1,9 s | 0,94 | 1,00 | 4 | 5 | 0,00043 | 0,0 | 16,7 |
| 22 | A | P8 | 1.902 | 2 | 2,2 s | 0,44 | 0,33 | 3 | 4 | 0,00024 | 0,0 | 33,3 |
| 23 | A | P8 | 3.697 | 3 | 2,6 s | 0,74 | 0,67 | 4 | 5 | 0,00032 | 0,0 | 50,0 |
| 24 | A | P8 | 3.593 | 3 | 2,5 s | 0,47 | 0,33 | 3 | 5 | 0,00032 | 0,0 | 45,5 |
| 25 | B | P1 | 8.231 | 9 | 7,3 s | 0,94 | 1,00 | 4 | 5 | 0,00215 | 19,1 | 62,5 |
| 26 | B | P1 | 8.633 | 9 | 10,1 s | 0,94 | 1,00 | 4 | 5 | 0,00112 | 16,7 | 46,7 |
| 27 | B | P1 | 10.004 | 10 | 12,4 s | 0,94 | 1,00 | 4 | 5 | 0,00187 | 8,8 | 66,7 |
| 28 | B | P2 | 8.926 | 9 | 12,0 s | 0,90 | 1,00 | 4 | 4 | 0,00125 | 12,9 | 78,3 |
| 29 | B | P2 | 9.139 | 9 | 15,7 s | 0,90 | 1,00 | 4 | 4 | 0,00130 | 28,4 | 79,2 |
| 30 | B | P2 | 9.481 | 9 | 16,5 s | 0,90 | 1,00 | 4 | 4 | 0,00112 | 13,3 | 77,8 |
| 31 | B | P3 | 9.577 | 10 | 11,2 s | 0,75 | 0,75 | 4 | 4 | 0,00143 | 11,7 | 71,4 |
| 32 | B | P3 | 7.268 | 9 | 9,1 s | 0,75 | 0,75 | 4 | 4 | 0,00143 | 10,2 | 50,0 |
| 33 | B | P3 | 7.517 | 9 | 7,3 s | 0,90 | 1,00 | 4 | 4 | 0,00146 | 27,0 | 46,2 |
| 34 | B | P4 | 9.107 | 9 | 10,7 s | 1,00 | 1,00 | 5 | 5 | 0,00135 | 28,3 | 61,9 |
| 35 | B | P4 | 8.727 | 9 | 13,2 s | 0,70 | 0,67 | 4 | 4 | 0,00125 | 14,8 | 55,6 |
| 36 | B | P4 | 8.920 | 9 | 17,3 s | 1,00 | 1,00 | 5 | 5 | 0,00160 | 29,2 | 61,1 |
| 37 | B | P5 | 9.318 | 9 | 11,4 s | 1,00 | 1,00 | 5 | 5 | 0,00128 | 14,1 | 69,2 |
| 38 | B | P5 | 8.450 | 9 | 12,1 s | 1,00 | 1,00 | 5 | 5 | 0,00135 | 25,9 | 65,0 |
| 39 | B | P5 | 9.861 | 10 | 8,8 s | 1,00 | 1,00 | 5 | 5 | 0,00160 | 13,0 | 75,0 |
| 40 | B | P6 | 8.971 | 9 | 17,4 s | 0,90 | 1,00 | 4 | 4 | 0,00136 | 35,4 | 71,4 |
| 41 | B | P6 | 11.677 | 11 | 26,9 s | 0,90 | 1,00 | 4 | 4 | 0,00125 | 12,1 | 66,7 |
| 42 | B | P6 | 9.931 | 10 | 13,5 s | 0,90 | 1,00 | 4 | 4 | 0,00181 | 9,9 | 57,9 |
| 43 | B | P7 | 8.521 | 9 | 10,5 s | 0,94 | 1,00 | 4 | 5 | 0,00120 | 10,8 | 85,7 |
| 44 | B | P7 | 10.049 | 10 | 14,7 s | 0,90 | 1,00 | 4 | 4 | 0,00185 | 26,6 | 79,2 |
| 45 | B | P7 | 8.012 | 9 | 13,4 s | 0,90 | 1,00 | 4 | 4 | 0,00124 | 11,4 | 77,8 |
| 46 | B | P8 | 9.165 | 9 | 8,1 s | 0,94 | 1,00 | 4 | 5 | 0,00220 | 8,4 | 69,6 |
| 47 | B | P8 | 10.199 | 9 | 8,7 s | 0,74 | 0,67 | 4 | 5 | 0,00169 | 23,1 | 81,2 |
| 48 | B | P8 | 9.975 | 10 | 19,8 s | 0,74 | 0,67 | 4 | 5 | 0,00127 | 13,5 | 73,9 |
| 49 | C | P1 | 8.082 | 9 | 12,7 s | 0,84 | 1,00 | 3 | 4 | 0,00351 | 43,2 | 53,3 |
| 50 | C | P1 | 10.602 | 10 | 16,1 s | 0,94 | 1,00 | 4 | 5 | 0,00444 | 39,5 | 59,1 |
| 51 | C | P1 | 8.746 | 9 | 21,2 s | 0,94 | 1,00 | 4 | 5 | 0,00443 | 38,1 | 62,5 |
| 52 | C | P2 | 8.799 | 9 | 17,2 s | 0,84 | 1,00 | 3 | 4 | 0,00302 | 46,8 | 83,3 |
| 53 | C | P2 | 8.845 | 9 | 15,5 s | 0,94 | 1,00 | 4 | 5 | 0,00361 | 38,8 | 75,0 |
| 54 | C | P2 | 8.150 | 9 | 44,3 s | 0,94 | 1,00 | 4 | 5 | 0,00523 | 52,3 | 72,2 |
| 55 | C | P3 | 7.139 | 9 | 14,2 s | 0,69 | 0,75 | 3 | 4 | 0,00470 | 47,2 | 66,7 |
| 56 | C | P3 | 7.445 | 9 | 13,3 s | 0,84 | 1,00 | 3 | 4 | 0,00260 | 22,1 | 66,7 |
| 57 | C | P3 | 10.244 | 11 | 27,3 s | 0,75 | 0,75 | 4 | 4 | 0,00558 | 45,9 | 61,5 |
| 58 | C | P4 | 14.406 | 12 | 25,3 s | 0,11 | 0,00 | 1 | 4 | 0,01323 | 29,0 | 50,0 |
| 59 | C | P4 | 9.902 | 10 | 17,8 s | 0,38 | 0,33 | 2 | 4 | 0,00478 | 38,5 | 55,6 |
| 60 | C | P4 | 10.375 | 10 | 16,1 s | 0,38 | 0,33 | 2 | 4 | 0,00458 | 42,7 | 55,6 |
| 61 | C | P5 | 8.884 | 10 | 11,6 s | 0,94 | 1,00 | 4 | 5 | 0,00311 | 45,0 | 64,7 |
| 62 | C | P5 | 10.257 | 10 | 16,3 s | 0,94 | 1,00 | 4 | 5 | 0,00387 | 37,6 | 69,2 |
| 63 | C | P5 | 9.951 | 10 | 14,5 s | 0,94 | 1,00 | 4 | 5 | 0,00490 | 30,6 | 69,6 |
| 64 | C | P6 | 8.115 | 9 | 14,5 s | 0,74 | 0,67 | 4 | 5 | 0,00347 | 41,0 | 55,6 |
| 65 | C | P6 | 7.102 | 9 | 14,4 s | 0,64 | 0,67 | 3 | 4 | 0,00379 | 54,3 | 58,3 |
| 66 | C | P6 | 7.909 | 9 | 13,6 s | 0,94 | 1,00 | 4 | 5 | 0,00366 | 36,6 | 66,7 |
| 67 | C | P7 | 11.895 | 11 | 27,0 s | 0,94 | 1,00 | 4 | 5 | 0,00649 | 50,7 | 81,0 |
| 68 | C | P7 | 8.218 | 9 | 21,1 s | 0,84 | 1,00 | 3 | 4 | 0,00516 | 32,6 | 78,9 |
| 69 | C | P7 | 8.050 | 9 | 16,6 s | 0,94 | 1,00 | 4 | 5 | 0,00421 | 64,2 | 77,8 |
| 70 | C | P8 | 9.556 | 10 | 15,8 s | 0,38 | 0,33 | 2 | 4 | 0,00429 | 39,9 | 71,4 |
| 71 | C | P8 | 9.804 | 10 | 22,1 s | 0,38 | 0,33 | 2 | 4 | 0,00404 | 50,1 | 72,2 |
| 72 | C | P8 | 8.599 | 9 | 20,0 s | 0,64 | 0,67 | 3 | 4 | 0,00521 | 35,5 | 73,7 |
| 73 | D | P1 | 5.977 | 7 | 10,6 s | 0,94 | 1,00 | 4 | 5 | 0,00106 | 16,9 | 41,7 |
| 74 | D | P1 | 7.664 | 8 | 12,0 s | 0,94 | 1,00 | 4 | 5 | 0,00091 | 21,9 | 50,0 |
| 75 | D | P1 | 6.336 | 7 | 15,6 s | 0,94 | 1,00 | 4 | 5 | 0,00101 | 23,0 | 50,0 |
| 76 | D | P2 | 5.386 | 7 | 13,8 s | 0,94 | 1,00 | 4 | 5 | 0,00085 | 24,7 | 58,3 |
| 77 | D | P2 | 5.246 | 7 | 7,1 s | 1,00 | 1,00 | 5 | 5 | 0,00086 | 22,9 | 66,7 |
| 78 | D | P2 | 5.316 | 7 | 4,9 s | 0,94 | 1,00 | 4 | 5 | 0,00093 | 23,0 | 66,7 |
| 79 | D | P3 | 5.142 | 7 | 4,8 s | 0,90 | 1,00 | 4 | 4 | 0,00075 | 18,2 | 55,6 |
| 80 | D | P3 | 5.484 | 7 | 18,4 s | 0,84 | 1,00 | 3 | 4 | 0,00100 | 35,1 | 72,7 |
| 81 | D | P3 | 5.870 | 7 | 7,3 s | 0,75 | 0,75 | 4 | 4 | 0,00111 | 11,8 | 64,3 |
| 82 | D | P4 | 5.541 | 7 | 16,6 s | 1,00 | 1,00 | 5 | 5 | 0,00124 | 10,5 | 50,0 |
| 83 | D | P4 | 6.870 | 8 | 13,7 s | 1,00 | 1,00 | 5 | 5 | 0,00165 | 8,6 | 45,5 |
| 84 | D | P4 | 6.791 | 8 | 6,2 s | 0,70 | 0,67 | 4 | 4 | 0,00083 | 16,7 | 41,7 |
| 85 | D | P5 | 6.815 | 8 | 6,8 s | 1,00 | 1,00 | 5 | 5 | 0,00132 | 8,2 | 60,0 |
| 86 | D | P5 | 5.614 | 7 | 7,1 s | 0,94 | 1,00 | 4 | 5 | 0,00095 | 36,8 | 57,1 |
| 87 | D | P5 | 6.562 | 7 | 6,4 s | 0,74 | 0,67 | 4 | 5 | 0,00091 | 25,8 | 65,0 |
| 88 | D | P6 | 7.123 | 8 | 8,2 s | 0,64 | 0,67 | 3 | 4 | 0,00120 | 9,8 | 58,8 |
| 89 | D | P6 | 6.831 | 8 | 5,5 s | 0,90 | 1,00 | 4 | 4 | 0,00097 | 22,4 | 50,0 |
| 90 | D | P6 | 5.503 | 7 | 9,6 s | 0,64 | 0,67 | 3 | 4 | 0,00075 | 20,6 | 40,0 |
| 91 | D | P7 | 5.681 | 7 | 4,9 s | 1,00 | 1,00 | 5 | 5 | 0,00065 | 34,1 | 66,7 |
| 92 | D | P7 | 5.246 | 7 | 5,1 s | 1,00 | 1,00 | 5 | 5 | 0,00110 | 29,0 | 75,0 |
| 93 | D | P7 | 5.275 | 7 | 3,5 s | 0,94 | 1,00 | 4 | 5 | 0,00148 | 12,8 | 66,7 |
| 94 | D | P8 | 7.157 | 7 | 11,9 s | 0,74 | 0,67 | 4 | 5 | 0,00090 | 16,3 | 63,6 |
| 95 | D | P8 | 7.366 | 8 | 10,0 s | 0,64 | 0,67 | 3 | 4 | 0,00095 | 15,1 | 62,5 |
| 96 | D | P8 | 6.669 | 7 | 9,3 s | 0,74 | 0,67 | 4 | 5 | 0,00108 | 38,9 | 66,7 |
| 97 | E | P1 | 6.325 | 8 | 11,9 s | 0,94 | 1,00 | 4 | 5 | 0,00121 | 44,6 | 30,8 |
| 98 | E | P1 | 8.344 | 9 | 12,9 s | 0,94 | 1,00 | 4 | 5 | 0,00152 | 21,5 | 52,4 |
| 99 | E | P1 | 8.230 | 9 | 12,1 s | 0,69 | 0,75 | 3 | 4 | 0,00172 | 25,2 | 52,6 |
| 100 | E | P2 | 5.471 | 8 | 6,3 s | 1,00 | 1,00 | 5 | 5 | 0,00115 | 14,6 | 66,7 |
| 101 | E | P2 | 5.720 | 8 | 6,6 s | 0,90 | 1,00 | 4 | 4 | 0,00099 | 19,4 | 66,7 |
| 102 | E | P2 | 5.529 | 8 | 6,0 s | 1,00 | 1,00 | 5 | 5 | 0,00082 | 18,0 | 66,7 |
| 103 | E | P3 | 8.347 | 10 | 12,0 s | 0,75 | 0,75 | 4 | 4 | 0,00125 | 29,5 | 56,2 |
| 104 | E | P3 | 5.889 | 8 | 7,5 s | 0,78 | 1,00 | 2 | 4 | 0,00153 | 23,8 | 66,7 |
| 105 | E | P3 | 6.818 | 9 | 14,3 s | 0,90 | 1,00 | 4 | 4 | 0,00118 | 10,4 | 50,0 |
| 106 | E | P4 | 5.415 | 8 | 7,7 s | 1,00 | 1,00 | 5 | 5 | 0,00128 | 11,6 | 37,5 |
| 107 | E | P4 | 7.442 | 9 | 31,7 s | 0,94 | 1,00 | 4 | 5 | 0,00192 | 14,0 | 50,0 |
| 108 | E | P4 | 7.327 | 9 | 11,7 s | 0,90 | 1,00 | 4 | 4 | 0,00150 | 20,6 | 50,0 |
| 109 | E | P5 | 5.877 | 8 | 11,5 s | 0,94 | 1,00 | 4 | 5 | 0,00135 | 9,6 | 60,0 |
| 110 | E | P5 | 5.913 | 8 | 11,3 s | 0,94 | 1,00 | 4 | 5 | 0,00070 | 20,2 | 60,0 |
| 111 | E | P5 | 5.991 | 8 | 10,9 s | 0,94 | 1,00 | 4 | 5 | 0,00096 | 12,8 | 62,5 |
| 112 | E | P6 | 7.090 | 9 | 6,3 s | 0,64 | 0,67 | 3 | 4 | 0,00125 | 12,2 | 46,7 |
| 113 | E | P6 | 7.507 | 9 | 5,8 s | 0,90 | 1,00 | 4 | 4 | 0,00124 | 36,5 | 56,2 |
| 114 | E | P6 | 5.658 | 8 | 8,1 s | 0,90 | 1,00 | 4 | 4 | 0,00070 | 16,1 | 53,8 |
| 115 | E | P7 | 5.589 | 8 | 6,6 s | 0,84 | 1,00 | 3 | 4 | 0,00109 | 28,4 | 66,7 |
| 116 | E | P7 | 5.576 | 8 | 6,2 s | 0,94 | 1,00 | 4 | 5 | 0,00109 | 28,9 | 66,7 |
| 117 | E | P7 | 5.493 | 8 | 5,1 s | 0,94 | 1,00 | 4 | 5 | 0,00074 | 34,3 | 66,7 |
| 118 | E | P8 | 6.817 | 8 | 10,2 s | 0,94 | 1,00 | 4 | 5 | 0,00096 | 10,3 | 70,0 |
| 119 | E | P8 | 6.149 | 8 | 6,6 s | 0,38 | 0,33 | 2 | 4 | 0,00085 | 14,6 | 68,8 |
| 120 | E | P8 | 6.490 | 8 | 8,6 s | 0,94 | 1,00 | 4 | 5 | 0,00091 | 14,2 | 72,2 |

</details>

## 8. Conclusiones

1. **Para este problema, el agente único es la mejor opción.** Con 11 documentos y preguntas que se resuelven con pocas búsquedas, el agente único es de **4 a 20 veces más barato** (0,00027 frente a 0,00110–0,00545 USD por tarea) y **2,5 a 6 veces más rápido**, con calidad igual o mejor. Está solo en la frontera de Pareto.
2. **El multiagente tiene un costo de coordinación real.** Sube de 2 a ~9–10 llamadas y de ~2.000 a ~9.000–10.000 tokens por tarea. Entre el 21 % y el 40 % del costo se va en coordinar (orquestador y crítico), y ≈ 70 % de las consultas a documentos son repetidas entre investigadores.
3. **Un modelo más fuerte no ayuda.** La configuración C (`deepseek-v4-pro` en orquestador, redactor y crítico) cuesta **3,5 veces más que B** y no mejora la calidad; además tiene la latencia más alta.
4. **Optimizar sí funciona.** Reducir subtareas, rondas del crítico y longitud de los resúmenes (D) o enrutar (E) baja el costo ≈ 30 % frente a B sin perder calidad. Aun así **no alcanza el reto del ejercicio 3** (costar al menos 40 % menos que B sin perder más de 0,05 de calidad): la calidad se mantiene, pero el ahorro es de ≈ 30 %, no de 40 %.
5. **No encontramos ninguna pregunta donde el equipo supere claramente al agente único** (en LangGraph el agente único iguala o supera a B en 6 de las 8; la mayor brecha es «financiar la segunda tienda sin crédito»: 0,92 frente a 0,76). Es una hipótesis **no probada** que el reparto del trabajo se justifique con un corpus mucho mayor o con preguntas que exijan combinar muchos documentos: este experimento no lo cubre.
6. **Medimos el ruido y es grande.** Con 24 tareas por configuración, la desviación estándar de la calidad es de 0,09 a 0,25, y el **orden entre A y B se invierte** entre las dos implementaciones (LangGraph: A 0,905 > B 0,853; desde cero: B 0,894 > A 0,869). El juez también varía: calificó el mismo texto entre 0,84 y 0,94. Por eso **solo las diferencias de costo, latencia y número de llamadas son firmes**; las diferencias de calidad entre A, B, D y E **no son concluyentes**. Con una sola corrida y un solo juez habríamos podido «demostrar» cualquier ranking.

## 9. Qué corregimos respecto a los notebooks base

- **Resultados simulados.** El notebook «desde cero» traía resultados guardados que coincidían dígito a dígito con el modo simulado (0,843 / 0,731 / 0,775 y respuestas que solo copiaban texto de los documentos). Se reejecutó con modelos reales.
- **Celdas rotas** en ese notebook: ruta absoluta de Windows (`C:\Users\...`), variable `ORIGEN_CLAVE` sin definir y celdas duplicadas. Ahora usa `find_dotenv` y corre en cualquier máquina.
- **Dependencias:** `pyproject.toml` no incluía `langgraph`, `langchain-openai` ni `langchain-core`; se agregaron. Se quitó `%pip install` (no hay pip en un entorno de `uv`).
- **DeepSeek y el razonamiento:** los modelos gastaban los `max_tokens` pensando y cortaban el JSON (`LengthFinishReasonError`). Se desactivó el razonamiento con `reasoning: {enabled: false}` y se subieron los límites del orquestador, crítico y router a 600 tokens y del juez a 1.000.
- **Fallos que tumbaban tareas:** una salida cortada o inválida del router, orquestador o crítico ya no rompe la tarea; cae en el fallback documentado (equipo completo, una subtarea, o respuesta aprobada).
- **Juez roto:** Haiku omitía la clave `claridad` y 21 de 24 juicios fallaban, dejando tablas calculadas con 3 datos. Se dejaron las cuatro claves explícitas y obligatorias en el prompt, y los juicios fallidos se cuentan en la tabla (`juicios fallidos`: 0).
- **Latencia mal medida** (versión desde cero): se lanzaban las 24 tareas a la vez y la latencia incluía la espera en la cola del semáforo (B marcaba 59 s en vez de ~12 s). Se limitaron las tareas simultáneas a 3.
- **Juez independiente:** se pasó de un juez de la misma familia (DeepSeek) a uno de otra (Claude Haiku) para reducir el sesgo de autoevaluación.
- **3 repeticiones por pregunta** y columnas de desviación estándar, número de tareas y juicios fallidos en la tabla resumen.

## 10. Cómo correrlo

Requisitos: [`uv`](https://docs.astral.sh/uv/) y una clave de [OpenRouter](https://openrouter.ai).

```bash
uv sync
echo "OPENROUTER_API_KEY=sk-or-v1-..." > .env     # .env está en .gitignore: nunca se sube
```

Abrir uno de los notebooks en VS Code con el kernel de `.venv` (o `uv add --dev jupyterlab && uv run jupyter lab`) y ejecutar todas las celdas. Una corrida completa cuesta ≈ 0,25 USD y tarda unos minutos.

Si no hay clave, el notebook de LangGraph corre en **modo simulado**: un modelo de chat falso imita las respuestas para que el grafo, las trazas y las métricas funcionen sin gastar crédito. Sus puntajes de calidad **no** describen a ningún modelo real.

Cambiar de modelo es editar el diccionario `MODELOS` (un modelo por rol); cambiar el comportamiento del sistema, editar `Config` (`max_subtareas`, `max_rondas_critico`, `palabras_resumen`, `tope_usd`, modelos por rol).

## 11. Limitaciones y pendiente

- **Corpus pequeño y ficticio** (11 documentos) con búsqueda por palabras clave: favorece al agente único. Los resultados no se pueden extrapolar a problemas con mucha más información.
- **8 preguntas × 3 repeticiones** es una muestra pequeña, y el juez es un LLM con su propia variabilidad.
- **Sin memoria entre tareas** (ver [sección 4](#4-cómo-se-comparten-información-y-memoria)).
- **Ejercicios del notebook sin hacer:** verificador de hechos (1), pipeline fijo en lugar de `Send` (2), humano en el bucle con `MemorySaver` (4), agente prefabricado `create_react_agent` (5) y comparación de ambas versiones en detalle (6). El 3 (reto de costo ≥ 40 % menos que B) **no se alcanzó**: D y E llegan a ≈ 30 %.
- Las gráficas de `img/` salen del notebook de LangGraph; las del notebook desde cero están dentro del propio notebook.
