# MEGADAPT — Trazabilidad del Pipeline de Análisis Cualitativo

**Proyecto:** MEGADAPT — Análisis de vulnerabilidad hídrica en Ciudad de México  
**Responsable:** Omar Palacios-Barajas (omarpb@ciencias.unam.mx)  
**Última actualización:** 2026-10-10

---

## 1. Historial de versiones

| Versión | Archivo | Modelo base | Novedades principales | Estado |
|---------|---------|-------------|----------------------|--------|
| v1–v4 | `megadapt_qualitative_pipeline_v1-4.py` | Google Gemini | Desarrollo inicial; extracción iterativa; distintos formatos de salida | Archivado |
| v5 | `megadapt_qualitative_pipeline_v5.py` | OpenAI (`gpt-4o-mini`) | Migración a OpenAI Structured Output con Pydantic; `MODEL_CATALOG`; `ThreadPoolExecutor`; checkpoints individuales JSON + CSV; reintentos `tenacity` | **Referencia base** |
| v6 | `megadapt_qualitative_pipeline_v6.py` | OpenAI | Estandarización opcional con diccionario (`--dictionary`, `--promote-approved`, `--only-standardize`); flujo de dos fases: extracción → estandarización | En uso / evaluación |
| v7 | `megadapt_qualitative_pipeline_v7.py` | OpenAI | **Framework de experimentación con variantes de prompt** (ver §3); fix bug "Original"; parámetro `--max-items`; modo `--run-all-variants`; `experiment_config.json` por corrida | **Activo para experimentos** |

---

## 2. Diagnóstico de calidad (feedback Dra. Ileana)

Al revisar `rev_Comparativa_Final_Humano_vs_LLM_v5.xlsx`, se identificaron cuatro problemas:

### Problema 1 — Bug "Original" (13 ocurrencias confirmadas)
El LLM devuelve la palabra literal `"Original"` como etiqueta de acción cuando no puede identificar una acción concreta para un actor. Ejemplo:
```
actor: "SACMEX", accion: "Original"
```
**Fix en v7:** conjunto `_NULL_LABELS` + funciones `postprocess_items()` / `postprocess_actions()` que filtran estos valores como postproceso.

### Problema 2 — Relleno genérico
El LLM siempre devuelve 4 ítems por campo aunque la entrevista no los justifique. Agrega temas esperables pero no mencionados (e.g., "Inundaciones en la ciudad" en una entrevista sobre salud).

### Problema 3 — Inferencia más allá del texto
El modelo infiere consecuencias o causas que son lógicas en el contexto pero no aparecen en la transcripción. Reduce la fidelidad al discurso del entrevistado.

### Problema 4 — Vocabulario generalizado
Los analistas humanos usan etiquetas específicas al contexto de cada entrevista. El LLM tiende a etiquetar con términos técnicos estándar que podrían aplicar a cualquier entrevista del corpus.

---

## 3. Variantes de prompt (v7)

Cada variante ataca uno o más de los problemas del §2. Se seleccionan con `--prompt-variant`.

| Nombre | Problema que ataca | Estrategia | Cuándo usar |
|--------|-------------------|------------|-------------|
| `baseline` | — (referencia) | Prompt v5 original sin cambios | Siempre como punto de comparación |
| `strict_grounding` | P2 + P3 | Solo codifica lo explícitamente mencionado; permite listas de 1–2 ítems; prohíbe inferir | Cuando el recall es secundario y se prioriza precisión |
| `human_style` | P4 | Ejemplos contrastivos (qué evitar vs. qué preferir) con etiquetas del corpus real | Cuando se quiere maximizar similitud con el estilo humano (BLEU/chrF) |
| `minimal` | P2 | Máximo 2 ítems por campo; calidad sobre cantidad | Entrevistas cortas o cuando hay demasiado ruido |
| `quote_based` | P3 + P4 | Razonamiento en dos pasos: localizar cita → resumir con vocabulario del entrevistado | Cuando la alucinación es el mayor problema |

### Texto completo de cada variante
El texto se encuentra en `PROMPT_VARIANTS` dentro de `megadapt_qualitative_pipeline_v7.py`. Para auditarlo sin ejecutar el script:
```bash
python -c "from megadapt_qualitative_pipeline_v7 import PROMPT_VARIANTS; \
[print(f'\n--- {k} ---\n{v}') for k, v in PROMPT_VARIANTS.items()]"
```

---

## 4. Flujo de experimentación recomendado

### Paso 1 — Estimar costo antes de correr
```bash
python megadapt_qualitative_pipeline_v7.py \
    --folder ./transcripciones \
    --count-only \
    --models gpt-4o-mini,gpt-4o
```
Genera `count_report/token_report.csv` y `model_cost_summary.csv`.

### Paso 2 — Correr todas las variantes
```bash
python megadapt_qualitative_pipeline_v7.py \
    --folder ./transcripciones \
    --model gpt-4o-mini \
    --run-all-variants
```
Genera una carpeta por variante: `output_baseline/`, `output_strict_grounding/`, etc.  
Cada carpeta contiene:
- `analysis_v7.csv` — resultados de extracción
- `experiment_config.json` — fecha, modelo, variante, texto del prompt usado
- `individual/<Id>.json` — JSON por entrevista
- `usage_log_v7.jsonl` — tokens y costo real

### Paso 3 — Evaluar con métricas NLP
```bash
python megadapt_eval_nlp_2.py \
    --human "2.Análisis Cualitativo Integración.xlsx" \
    --llm baseline=output_baseline/analysis_v7.csv \
           strict_grounding=output_strict_grounding/analysis_v7.csv \
           human_style=output_human_style/analysis_v7.csv \
           minimal=output_minimal/analysis_v7.csv \
           quote_based=output_quote_based/analysis_v7.csv \
    --metrics bleu chrf bertscore \
    --xlsx
```
Genera `eval_report.xlsx` con BLEU, chrF++, BERTScore por variante y por campo.

### Paso 4 — Revisar resultados con el equipo
- Comparar scores NLP por campo y variante.
- Revisar manualmente 5–10 entrevistas en la variante ganadora.
- Decidir si combinar variantes (e.g., `strict_grounding` para causas, `human_style` para preocupaciones).

---

## 5. Decisiones de diseño registradas

| Fecha | Decisión | Justificación |
|-------|----------|---------------|
| 2024 | Migrar de Gemini a OpenAI Structured Output | Mejor cumplimiento del schema JSON; menos postproceso manual |
| 2024 | `response_format=QualitativeAnalysisSchema` con Pydantic | Garantiza tipos y estructura; evita parsing frágil |
| 2024 | `tenacity` con backoff exponencial | Maneja rate limits de OpenAI sin intervención manual |
| 2024 | Checkpoints JSON individuales + CSV rolling | Permite reanudar corridas interrumpidas; evita re-procesar |
| 2025 | Añadir diccionario de homologación (v6) | Estandarizar vocabulario para facilitar análisis cuantitativo posterior |
| 2026-10 | Framework de variantes de prompt (v7) | Feedback Dra. Ileana: necesidad de experimentar para mejorar calidad antes de publicar |
| 2026-10 | Fix postproceso "Original" bug | 13 ocurrencias confirmadas en comparativa v5; solución no requiere re-prompt |

---

## 6. Pendientes y próximos pasos

### Inmediatos
- [ ] **Correr experimentos con v7** sobre el corpus completo de transcripciones (todas las 5 variantes)
- [ ] **Revisar resultados cualitativamente** con Dra. Ileana: ¿qué variante captura mejor el estilo del equipo?
- [ ] **Comparar métricas NLP** baseline vs. variantes: ¿mejora BLEU/chrF? ¿a qué costo en recall?

### Mediano plazo
- [ ] **Diccionario homologado** (v6 + v7): construir lista de términos preferidos por el equipo humano e integrarla como constricción en el prompt o en postproceso
- [ ] **Combinar variantes por campo**: e.g., `quote_based` para Acciones (campo con más errores) y `human_style` para Preocupaciones
- [ ] **Evaluación cualitativa estructurada**: tabla de revisión por entrevista con columnas Correcto / Parcial / Error / Alucinación

### A largo plazo
- [ ] **Fine-tuning o few-shot específico**: usar las codificaciones humanas validadas como ejemplos en el prompt (currently prohibido por costo de contexto, pero viable con modelos más baratos)
- [ ] **Integración con análisis de redes**: los actores extraídos alimentan el modelo de red socio-institucional del proyecto

---

## 7. Archivos relacionados

| Archivo | Descripción |
|---------|-------------|
| `claude/megadapt_qualitative_pipeline_v5.py` | Pipeline base (referencia) |
| `claude/megadapt_qualitative_pipeline_v6.py` | Añade estandarización con diccionario |
| `claude/megadapt_qualitative_pipeline_v7.py` | Framework de experimentación (actual) |
| `claude/megadapt_eval_nlp_2.py` | Evaluación NLP: BLEU, chrF++, BERTScore |
| `claude/megadapt_compare_human.py` | Comparación cualitativa humano vs. LLM |
| `rev_Comparativa_Final_Humano_vs_LLM_v5.xlsx` | Análisis de errores v5 (diagnóstico Dra. Ileana) |
