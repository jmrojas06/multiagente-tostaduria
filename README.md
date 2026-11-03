# Arquitectura multiagente: diseño, implementación y medición

Sistema de varios agentes de IA que responde preguntas de negocio y se compara, con datos reales, contra un agente único. Se mide **calidad, costo, latencia, robustez y costo de coordinación**.

Caso guía: la *Tostaduría Samán* (empresa ficticia de Cali) debe decidir su próxima inversión. Los agentes consultan una base de conocimiento interna de 11 documentos y responden citando las fuentes (`[D2]`, `[D7]`...).

> Partimos de los notebooks base del curso (repositorio del profesor `armandoordonez/multiagente`). Este repositorio es nuestra versión ejecutada, corregida y extendida.

## Contenido

| Archivo | Qué es |
|---|---|
| `arquitectura_multiagente_langgraph.ipynb` | Versión con **LangGraph**: el flujo es un grafo de estados. Ejecutado con modelos reales. |
| `arquitectura_multiagente.ipynb` | Versión **desde cero** (sin frameworks, solo el SDK `openai`). Ejecutado con modelos reales. |
| `resultados_langgraph.csv`, `trazas_langgraph.csv` | Resultados por tarea y trazas por llamada (LangGraph). |
| `pyproject.toml`, `uv.lock` | Dependencias (gestionadas con `uv`). |

## Cómo funciona

```
pregunta ─▶ Orquestador ─▶ Investigador ×N (en paralelo, con búsqueda) ─▶ Redactor ⇄ Crítico ─▶ respuesta
```

1. **Orquestador:** divide la pregunta en hasta 3 subtareas independientes (JSON validado con Pydantic).
2. **Investigadores:** uno por subtarea, corren **en paralelo** (`Send` en LangGraph, `asyncio.gather` desde cero). Cada uno busca en el corpus con la herramienta `buscar_en_corpus` y entrega hallazgos con cifras y citas.
3. **Redactor:** escribe la respuesta final (máx. 200 palabras, con recomendación) usando solo las notas.
4. **Crítico:** revisa con una rúbrica; si rechaza, el redactor corrige (máx. 2 rondas).
5. **Router (config. E):** un clasificador barato manda las preguntas simples al agente único y las complejas al equipo.

Patrones usados: orquestador–trabajadores, paralelización, evaluador–optimizador y enrutamiento.

Controles de robustez y costo: tope de **0,05 USD por tarea**, `max_pasos` por agente, reintentos con espera exponencial, y *degradación elegante* (si el plan o la evaluación llegan con formato inválido, el sistema sigue con una sola subtarea o da la respuesta por aprobada en vez de fallar).

## Cómo se mide

- **Trazas:** toda llamada al modelo deja una fila (agente, modelo, tokens, costo real reportado por OpenRouter, latencia).
- **Dataset:** 8 preguntas con sus puntos clave esperados, cada una repetida **3 veces** → 24 tareas por configuración.
- **Juez LLM de otra familia** (`anthropic/claude-haiku-5.5`) para no calificar a DeepSeek con DeepSeek. Calidad = 60 % cobertura de puntos clave + 25 % fidelidad + 15 % claridad.
- Se reportan desviación estándar, costo por 1.000 tareas, latencia p50/p95, % del costo en coordinación y % de consultas repetidas. También la frontera de Pareto costo–calidad.

Configuraciones comparadas:

| | Descripción |
|---|---|
| A | Agente único (línea base) |
| B | Multiagente, todos los roles con el modelo barato |
| C | Multiagente mixto: orquestador, redactor y crítico con `deepseek-v4-pro` |
| D | Multiagente optimizado (2 subtareas, 1 ronda de crítico, resúmenes de 80 palabras) |
| E | Router (simple → agente único, compleja → equipo) |

Modelos: `deepseek/deepseek-v4.1-flash` (trabajadores, orquestador, crítico), `deepseek/deepseek-v4-pro` (modelo fuerte de C), `anthropic/claude-haiku-5.5` (juez).

## Resultados (modelos reales, versión LangGraph)

| Config | Calidad (0-1) | Costo por tarea (USD) | Latencia p50 | Llamadas | % costo en coordinación |
|---|---|---|---|---|---|
| A · agente único | 0,905 | 0,00027 | 2,4 s | 2,1 | 0 |
| B · multiagente barato | 0,853 | 0,00157 | 7,2 s | 9,2 | 21 |
| C · multiagente mixto | 0,821 | 0,00545 | 13,9 s | 10,0 | 40 |
| D · optimizado | 0,850 | 0,00112 | 6,4 s | 7,2 | 22 |
| E · router | 0,847 | 0,00110 | 6,1 s | 8,2 | 21 |

Éxito 100 % en las cinco; desviación estándar de la calidad entre 0,11 y 0,16. El costo total de la sesión fue ≈ 0,25 USD.

**Conclusiones (con la honestidad que pide la medición):**

- Con un corpus pequeño de 11 documentos, el **agente único es de 4 a 20 veces más barato y igual o mejor** que cualquier multiagente. El multiagente no se justifica para este problema.
- Gastar en un modelo más fuerte (C) **no mejora** la calidad y multiplica el costo por 3,5 frente a B.
- Optimizar sí funciona: D y E cuestan ≈ 30 % menos que B con la misma calidad (cumple el reto del ejercicio 3).
- **El ruido es grande.** El notebook "desde cero", con la misma configuración, dio A = 0,869 y B = 0,894 (el orden se invierte respecto a LangGraph) y C = 0,741. Con 24 tareas y un juez que a veces califica distinto el mismo texto (0,84–0,94), solo las diferencias de costo y latencia son firmes; las de calidad entre A, B, D y E no son concluyentes.

## Qué corregimos respecto a los notebooks base

- El notebook "desde cero" traía resultados **simulados** guardados (coincidían dígito a dígito con el modo simulado), celdas rotas (ruta absoluta de Windows, variable `ORIGEN_CLAVE` sin definir) y celdas duplicadas. Ahora está limpio y ejecutado con modelos reales.
- `pyproject.toml` no incluía `langgraph`, `langchain-openai` ni `langchain-core`; se agregaron.
- DeepSeek gastaba los `max_tokens` en razonamiento y cortaba el JSON: se desactivó con `reasoning: {enabled: false}` y se subieron los límites del orquestador, crítico, router y juez.
- Una salida cortada ya no tumba la tarea: cae en el fallback documentado.
- El prompt del juez era ambiguo y Haiku omitía la clave `claridad` (21 de 24 juicios fallaban); se dejaron las cuatro claves explícitas y obligatorias, y los juicios fallidos se cuentan en la tabla.
- En el notebook "desde cero" la latencia incluía el tiempo en cola del semáforo; se limitaron las tareas simultáneas para medirla bien.
- Juez de otra familia y 3 repeticiones por pregunta.

## Cómo correrlo

Requisitos: [`uv`](https://docs.astral.sh/uv/) y una clave de [OpenRouter](https://openrouter.ai).

```bash
uv sync
echo "OPENROUTER_API_KEY=sk-or-v1-..." > .env      # .env está en .gitignore; nunca se sube
# abrir los notebooks en VS Code con el kernel de .venv (o: uv add --dev jupyterlab && uv run jupyter lab)
```

Ejecutar todas las celdas de uno de los dos notebooks. Una corrida completa cuesta ≈ 0,25 USD. Si no hay clave, el notebook de LangGraph corre en **modo simulado** (modelo falso, sin costo), útil para ver el grafo y las métricas, pero sus puntajes de calidad no describen a ningún modelo real.

## Pendiente

Ejercicios 1, 2 y 4–6 de los notebooks (verificador de hechos, pipeline fijo, humano en el bucle, prebuilt, comparación de versiones).
