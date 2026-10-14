"""
MEGADAPT - Pipeline de análisis cualitativo automatizado, v7 (OpenAI)
Framework de experimentación con variantes de prompt

Cambios respecto a v5/v6:
  A. PROMPT_VARIANTS: diccionario con 5 variantes de prompt nombradas para
     experimentar con distintas estrategias de extracción. Se elige con --prompt-variant.
  B. El nombre de la variante se incluye en: el CSV de salida, los JSON individuales
     y el nombre del directorio de salida (./output_<variante>/ por defecto).
  C. Fix del bug "Original": se eliminan como post-proceso los ítems donde el LLM
     devuelve la palabra "Original" como etiqueta o como acción.
  D. Parámetro --max-items: límite de ítems por campo (default 4). Usar 2 para
     variantes "minimal".
  E. Modo --run-all-variants: ejecuta todas las variantes en secuencia sobre la
     misma carpeta de transcripciones.

Variantes de prompt disponibles
  baseline          Prompt v5 original (punto de partida).
  strict_grounding  Solo codifica lo explícitamente mencionado; no infiere ni rellena.
  human_style       Prioriza el vocabulario y estilo del equipo humano (etiquetas
                    contextualmente específicas, no genéricas).
  minimal           Máximo 2 ítems por campo; calidad sobre cantidad.
  quote_based       Pide al LLM que ancle cada etiqueta en una cita directa del
                    entrevistado antes de resumirla (reduce hallucination).

Uso típico:
  # Ver tokens y costo sin llamar a la API:
  python megadapt_qualitative_pipeline_v7.py --folder "./LLMs-ModelosMentales/TRANSCRIPCIONES/_TRANSCRIPCIONES_35" --count-only

  # Correr una variante:
  python megadapt_qualitative_pipeline_v7.py \\
      --folder "./LLMs-ModelosMentales/TRANSCRIPCIONES/_TRANSCRIPCIONES_35" \\
      --model gpt-4o-mini \\
      --prompt-variant strict_grounding

  # Correr todas las variantes (genera una carpeta por variante):
  python megadapt_qualitative_pipeline_v7.py \\
      --folder "./LLMs-ModelosMentales/TRANSCRIPCIONES/_TRANSCRIPCIONES_35" \\
      --model gpt-4o-mini \\
      --run-all-variants

  # Evaluar con el script de métricas NLP:
  python megadapt_eval_nlp_2.py \\
      --human "2.Análisis Cualitativo Integración.xlsx" \\
      --llm baseline=output_baseline/analysis_v7.csv \\
             strict_grounding=output_strict_grounding/analysis_v7.csv \\
             human_style=output_human_style/analysis_v7.csv \\
      --metrics bleu chrf bertscore --xlsx

Requiere OPENAI_API_KEY en .env (salvo con --count-only).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tenacity import (before_sleep_log, retry, retry_if_exception,
                      stop_after_attempt, wait_exponential)
from tqdm import tqdm

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 0. Catálogo de modelos
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ModelInfo:
    context: int
    price_in: float   # USD por 1M tokens de entrada
    price_out: float  # USD por 1M tokens de salida

MODEL_CATALOG: dict[str, ModelInfo] = {
    "gpt-4o":        ModelInfo(128_000,   2.50, 10.00),
    "gpt-4o-mini":   ModelInfo(128_000,   0.15,  0.60),
    "gpt-4.1":       ModelInfo(1_047_576, 2.00,  8.00),
    "gpt-4.1-mini":  ModelInfo(1_047_576, 0.40,  1.60),
    "gpt-4.1-nano":  ModelInfo(1_047_576, 0.10,  0.40),
}

TEMPERATURE_PREFIXES = ("gpt-4o", "gpt-4.1")
PROMPT_OVERHEAD_TOKENS = 400
EST_OUTPUT_TOKENS = 900
CONTEXT_SAFETY = 0.90

# ──────────────────────────────────────────────────────────────────────────────
# 1. Estructuras Pydantic y dataclasses
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class InterviewMetadata:
    Id_entrevista: str

@dataclass
class InterviewResult:
    metadata: InterviewMetadata
    prompt_variant: str = "baseline"
    Preocupacion_principal_acerca_del_agua: list[str] = field(default_factory=list)
    Causas_principales: list[str] = field(default_factory=list)
    Consecuencias: list[str] = field(default_factory=list)
    Acciones_y_Actores_que_realizan_dichas_acciones: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    error: Optional[str] = None


class AccionActor(BaseModel):
    actor: str = Field(
        description="Quién realiza la acción (institución, grupo o persona), en 1 a 3 palabras."
    )
    accion: str = Field(
        description="La acción, resumida en 4 o 5 palabras."
    )


class QualitativeAnalysisSchema(BaseModel):
    Preocupacion_principal_acerca_del_agua: list[str] = Field(
        description="Preocupaciones principales sobre el agua, cada una de 4 o 5 palabras."
    )
    Causas_principales: list[str] = Field(
        description="Causas del problema mencionadas, cada una de 4 o 5 palabras."
    )
    Consecuencias: list[str] = Field(
        description="Consecuencias o efectos mencionados, cada una de 4 o 5 palabras."
    )
    Acciones_y_Actores_que_realizan_dichas_acciones: list[AccionActor] = Field(
        description="Acciones realizadas o propuestas, con el actor que las realiza."
    )

# ──────────────────────────────────────────────────────────────────────────────
# 2. Variantes de prompt
# ──────────────────────────────────────────────────────────────────────────────
# Cada variante es una cadena que se usa como system prompt.
# {max_items} se reemplaza por el valor de --max-items al construir el prompt.

_PROMPT_BASE_DEFINITIONS = """
Eres un analista socio-ambiental cualitativo, especializado en gestión del agua y riesgo hídrico en zonas urbanas de la Ciudad de México (inundaciones, escasez, calidad del agua, distribución).

Recibirás la transcripción completa de una entrevista con actores del sistema del agua (funcionarios, académicos, organizaciones civiles, vecinos). Los hablantes pueden venir como Entrevistador/Respondente, E1/E2, etc. Analiza SOLO lo que dicen los participantes entrevistados, NO las preguntas del entrevistador.

Tu tarea es codificar la entrevista en cuatro campos. Definiciones:

1. PREOCUPACIÓN PRINCIPAL ACERCA DEL AGUA: el problema del agua que el participante considera más importante o que motiva su trabajo o participación (escasez, mala calidad, inundaciones, tandeo, hundimientos, etc.). De 1 a {max_items} preocupaciones, ordenadas de mayor a menor importancia.

2. CAUSAS PRINCIPALES: factores que el participante señala como ORIGEN del problema (biofísicos, socio-institucionales, de uso de suelo, de infraestructura).

3. CONSECUENCIAS: efectos o impactos que el participante menciona que resultan del problema (salud, economía, vivienda, ambiente, organización social).

4. ACCIONES Y ACTORES: lo que se hace o propone hacer para enfrentar el problema, indicando QUIÉN lo hace. Si no se menciona el actor, usa "No especificado". NO inventes actores.
"""

_PROMPT_FORMAT_STANDARD = """
FORMATO:
- Cada elemento debe ser una frase corta de 4 o 5 palabras (máximo 6), en forma de etiqueta, no una oración completa.
- Ejemplos del estilo esperado: "Escasez de agua", "Tandeo en las colonias", "Fugas en la red", "Asentamientos irregulares", "Daños a la salud", "Compra de agua en pipas".
- Ejemplos de acciones con actor: {{"actor": "Protección Civil", "accion": "Emite recomendaciones de riesgo"}}, {{"actor": "Vecinos", "accion": "Almacenan agua en cisternas"}}.
- No repitas la misma idea dos veces dentro de un mismo campo.
- No agregues información que no esté en la transcripción. Si un campo no se menciona, devuelve lista vacía.
- Todo en español. Devuelve únicamente el JSON con la estructura solicitada.
"""

PROMPT_VARIANTS: dict[str, str] = {

    # ── P1: Baseline ─────────────────────────────────────────────────────────
    # Idéntico al v5 original. Es el punto de referencia para comparar.
    "baseline": _PROMPT_BASE_DEFINITIONS + _PROMPT_FORMAT_STANDARD,

    # ── P2: Strict grounding ─────────────────────────────────────────────────
    # Añade instrucciones explícitas para no inferir, no generalizar y
    # preferir lista corta de alta confianza. Ataca el problema de ítems
    # genéricos que no aparecen en la transcripción.
    "strict_grounding": _PROMPT_BASE_DEFINITIONS + """
INSTRUCCIONES CRÍTICAS PARA ESTA VARIANTE:
- SOLO codifica conceptos que el entrevistado menciona EXPLÍCITAMENTE. No inferir, no generalizar.
- Si un tema no se menciona con claridad, NO lo incluyas aunque sea esperable en el contexto.
- PREFERIBLE una lista de 1-2 ítems de alta confianza que 4 ítems donde alguno sea incierto.
- Si la entrevista no trata sobre inundaciones, NO pongas "Inundaciones" aunque sea un tema habitual.
- Si el entrevistado no menciona una acción concreta, deja la lista de Acciones vacía.
""" + _PROMPT_FORMAT_STANDARD,

    # ── P3: Human style ──────────────────────────────────────────────────────
    # Enfatiza el estilo del equipo humano: etiquetas específicas al contexto
    # de la entrevista, no genéricas. Incluye ejemplos contrastivos de qué
    # hacer y qué evitar, tomados de la comparativa real.
    "human_style": _PROMPT_BASE_DEFINITIONS + """
ESTILO DEL EQUIPO DE INVESTIGACIÓN (MUY IMPORTANTE):
Los analistas humanos del proyecto usan etiquetas MUY ESPECÍFICAS al contexto de cada entrevista.
Evita etiquetas genéricas que podrían aplicar a cualquier entrevista.

✗ EVITAR (demasiado genérico):
  "Escasez de agua", "Inundaciones en la ciudad", "Mala calidad del agua", "Fugas en la red"

✓ PREFERIR (específico a lo que dijo el entrevistado):
  "Suministro no es transparente/desigual", "Tandeo en colonias populares",
  "Grietas en zonas de alto riesgo", "Conflictos entre comuneros y avecindados",
  "Sesgo presupuestal en infraestructura", "Descargas de 11 mil casas irregulares"

Regla: si tu etiqueta podría aparecer en el 80% de las entrevistas del proyecto, es demasiado
genérica. Busca capturar lo que es CARACTERÍSTICO de esta entrevista en particular.
""" + _PROMPT_FORMAT_STANDARD,

    # ── P4: Minimal (pocas etiquetas, máxima calidad) ────────────────────────
    # Restringe a máximo 2 ítems por campo. Fuerza al modelo a seleccionar
    # solo lo más relevante y reduce el ruido de ítems secundarios.
    "minimal": _PROMPT_BASE_DEFINITIONS.replace(
        "De 1 a {max_items} preocupaciones", "De 1 a 2 preocupaciones MÁXIMO"
    ) + """
INSTRUCCIONES CRÍTICAS PARA ESTA VARIANTE:
- MÁXIMO 2 ítems por campo. Si hay más posibles, elige los 2 más importantes/frecuentes.
- Calidad sobre cantidad: es mejor 1 ítem muy preciso que 4 ítems mezclados.
- Solo incluye lo que el entrevistado repite, enfatiza o describe con detalle.
- Para Acciones: incluye solo las más concretas y verificables en la transcripción.
""" + _PROMPT_FORMAT_STANDARD,

    # ── P5: Quote-based (anclaje en citas directas) ──────────────────────────
    # Pide al modelo que PRIMERO identifique la cita textual que respalda
    # cada etiqueta. Reduce alucinación porque obliga a localizar evidencia
    # en el texto antes de resumirla. La cita no aparece en el JSON final,
    # pero el razonamiento interno mejora la extracción.
    "quote_based": _PROMPT_BASE_DEFINITIONS + """
METODOLOGÍA PARA ESTA VARIANTE — razona en dos pasos (internamente):
  PASO 1: Identifica fragmentos textuales concretos del entrevistado que hablen de
          preocupaciones, causas, consecuencias y acciones.
  PASO 2: Para cada fragmento, resume la idea en 4-5 palabras usando el vocabulario
          del entrevistado.

REGLAS ESTRICTAS:
- Solo incluye una etiqueta si puedes identificar UNA cita directa que la respalde.
- No incluyas conceptos sin soporte textual, aunque sean lógicamente esperables.
- Las etiquetas deben reflejar las PALABRAS del entrevistado, no vocabulario técnico externo.
- Si el entrevistado usa "tandeo", usa "tandeo"; si dice "grietas", usa "grietas".
""" + _PROMPT_FORMAT_STANDARD,
}

USER_TEMPLATE = "=== TRANSCRIPCIÓN ===\n{transcript}\n=== FIN DE TRANSCRIPCIÓN ==="

# Palabras que indican que el LLM no encontró contenido real — se eliminan en postproceso
_NULL_LABELS = {
    "original", "none", "ninguno", "ninguna", "na", "n/a", "n a",
    "no aplica", "no especificado como accion", "acción no especificada",
    "no hay accion", "sin accion", "acción desconocida",
}

# ──────────────────────────────────────────────────────────────────────────────
# 3. Lectura y preprocesamiento
# ──────────────────────────────────────────────────────────────────────────────
def load_transcript(path: Path) -> str:
    if path.suffix.lower() == ".docx":
        from docx import Document
        return "\n".join(p.text.strip() for p in Document(str(path)).paragraphs if p.text.strip())
    if path.suffix.lower() in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="replace")
    return ""


def parse_interview_id(filename: str) -> InterviewMetadata:
    stem = Path(filename).stem.upper()
    base, _, transcript_label = stem.partition("__T-")
    interview_id = base
    if "PARTE" in transcript_label:
        part_label = re.search(r"(\d+RA|\d+DA|\d+TA)\s*PARTE", transcript_label)
        if part_label:
            interview_id = f"{base}_{part_label.group(0).replace(' ', '_')}"
    return InterviewMetadata(Id_entrevista=interview_id)


def preprocess(text: str) -> str:
    """Limpieza ligera; no trunca."""
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"(?m)^\s*E:\s*", "Entrevistador: ", text)
    text = re.sub(r"(?m)^\s*R:\s*", "Respondente: ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

# ──────────────────────────────────────────────────────────────────────────────
# 4. Tokens y costos
# ──────────────────────────────────────────────────────────────────────────────
_ENCODER = None
TOKEN_MODE = "sin inicializar"


def _get_encoder():
    global _ENCODER, TOKEN_MODE
    if _ENCODER is None:
        try:
            import tiktoken
            _ENCODER = tiktoken.get_encoding("o200k_base")
            TOKEN_MODE = "exacto (tiktoken o200k_base)"
        except Exception as exc:
            log.warning(f"tiktoken no disponible ({type(exc).__name__}); aproximación 1.5 tok/pal.")
            _ENCODER = False
            TOKEN_MODE = "APROXIMADO (1.5 tokens/palabra)"
    return _ENCODER


def count_tokens(text: str) -> int:
    enc = _get_encoder()
    return len(enc.encode(text, disallowed_special=())) if enc else int(len(text.split()) * 1.5)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    info = MODEL_CATALOG.get(model)
    if not info:
        return None
    return input_tokens * info.price_in / 1e6 + output_tokens * info.price_out / 1e6


def list_transcripts(folder: str) -> list[Path]:
    return sorted(
        f for f in Path(folder).rglob("*")
        if not f.name.startswith("~$") and f.suffix.lower() in (".docx", ".txt", ".md")
    )


def count_only(folder: str, out_dir: str, models: list[str], variant: str = "baseline") -> pd.DataFrame:
    files = list_transcripts(folder)
    if not files:
        raise SystemExit(f"No se encontraron transcripciones en {folder}")
    prompt = build_prompt(variant, max_items=4)
    prompt_tokens = count_tokens(prompt) + PROMPT_OVERHEAD_TOKENS
    rows = []
    for f in tqdm(files, desc="Contando tokens"):
        text = preprocess(load_transcript(f))
        toks = count_tokens(text)
        row = {
            "Id_entrevista": parse_interview_id(f.name).Id_entrevista,
            "archivo": f.name,
            "palabras": len(text.split()),
            "tokens_transcripcion": toks,
            "tokens_entrada_total": toks + prompt_tokens,
        }
        for m in models:
            info = MODEL_CATALOG[m]
            row[f"pct_contexto_{m}"] = round(100 * (toks + prompt_tokens) / info.context, 1)
            row[f"costo_usd_{m}"] = round(estimate_cost(m, toks + prompt_tokens, EST_OUTPUT_TOKENS), 4)
        rows.append(row)
    df = pd.DataFrame(rows)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "token_report.csv", index=False, encoding="utf-8-sig")

    summary = []
    for m in models:
        info = MODEL_CATALOG[m]
        exceeds = int((df["tokens_entrada_total"] > CONTEXT_SAFETY * info.context).sum())
        summary.append({
            "modelo": m,
            "context_window": info.context,
            "precio_in_por_1M": info.price_in,
            "precio_out_por_1M": info.price_out,
            "max_pct_contexto": df[f"pct_contexto_{m}"].max(),
            "entrevistas_exceden_90pct": exceeds,
            "costo_total_usd_estimado": round(df[f"costo_usd_{m}"].sum(), 2),
        })
    sdf = pd.DataFrame(summary)
    sdf.to_csv(out / "model_cost_summary.csv", index=False, encoding="utf-8-sig")
    print(f"\nModo de conteo: {TOKEN_MODE}")
    print(f"Prompt variant: {variant} ({prompt_tokens} tokens de overhead)")
    print(f"Entrevistas: {len(df)} | palabras totales: {df['palabras'].sum():,}")
    print(sdf.to_string(index=False))
    print("\nPrecios de referencia: verificar en openai.com antes de correr.")
    return df

# ──────────────────────────────────────────────────────────────────────────────
# 5. Construcción del prompt con parámetros
# ──────────────────────────────────────────────────────────────────────────────
def build_prompt(variant: str, max_items: int = 4) -> str:
    """Devuelve el system prompt para la variante y max_items indicados."""
    if variant not in PROMPT_VARIANTS:
        raise ValueError(
            f"Variante '{variant}' no existe. Opciones: {list(PROMPT_VARIANTS)}"
        )
    return PROMPT_VARIANTS[variant].format(max_items=max_items)

# ──────────────────────────────────────────────────────────────────────────────
# 6. Cliente OpenAI con reintentos
# ──────────────────────────────────────────────────────────────────────────────
class UsageTracker:
    def __init__(self, model: str):
        self.model, self.input, self.output = model, 0, 0
        self._lock = threading.Lock()

    def add(self, usage) -> tuple[int, int]:
        i = getattr(usage, "prompt_tokens", 0) or 0
        o = getattr(usage, "completion_tokens", 0) or 0
        with self._lock:
            self.input += i
            self.output += o
        return i, o

    def cost(self) -> Optional[float]:
        return estimate_cost(self.model, self.input, self.output)


def _is_retryable(exc: BaseException) -> bool:
    import openai
    if getattr(exc, "code", None) == "insufficient_quota":
        return False
    if isinstance(exc, (openai.RateLimitError, openai.APIConnectionError,
                        openai.APITimeoutError, openai.InternalServerError)):
        return True
    msg = str(exc).lower()
    return any(k in msg for k in ("503", "429", "rate limit", "overloaded"))


def _call_structured(client, model: str, system: str, user: str, schema, temperature: Optional[float]):
    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(6),
        before_sleep=before_sleep_log(log, logging.WARNING),
        reraise=True,
    )
    def _do():
        kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            response_format=schema,
        )
        if temperature is not None and model.startswith(TEMPERATURE_PREFIXES):
            kwargs["temperature"] = temperature
        resp = client.chat.completions.parse(**kwargs)
        msg = resp.choices[0].message
        if getattr(msg, "refusal", None):
            raise RuntimeError(f"El modelo rechazó la solicitud: {msg.refusal}")
        if msg.parsed is None:
            raise ValueError("Respuesta sin contenido estructurado")
        return msg.parsed, resp.usage
    return _do()

# ──────────────────────────────────────────────────────────────────────────────
# 7. Postproceso: limpiar ítems nulos y el bug "Original"
# ──────────────────────────────────────────────────────────────────────────────
def clean_phrase(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s)).strip(" \t\n.;:,-•*")
    return s[:1].upper() + s[1:] if s else s


def _is_null_label(s: str) -> bool:
    """True si la etiqueta es un placeholder vacío o el bug 'Original'."""
    return s.strip().lower() in _NULL_LABELS


def dedupe(items: list[str]) -> list[str]:
    seen, out = set(), []
    for x in items:
        k = x.lower()
        if x and k not in seen:
            seen.add(k)
            out.append(x)
    return out


def postprocess_items(items: list[str]) -> list[str]:
    """Limpia frases: elimina nulls/placeholders, capitaliza, deduplica."""
    cleaned = [clean_phrase(x) for x in items]
    return dedupe([x for x in cleaned if x and not _is_null_label(x)])


def postprocess_actions(actions: list[dict]) -> list[dict]:
    """Limpia acciones: elimina las que tienen accion='Original' u otro null."""
    out = []
    for a in actions:
        actor  = clean_phrase(a.get("actor", "")) or "No especificado"
        accion = clean_phrase(a.get("accion", ""))
        if accion and not _is_null_label(accion):
            out.append({"actor": actor, "accion": accion})
    # Deduplicar por (actor, accion)
    seen, deduped = set(), []
    for a in out:
        key = (a["actor"].lower(), a["accion"].lower())
        if key not in seen:
            seen.add(key)
            deduped.append(a)
    return deduped

# ──────────────────────────────────────────────────────────────────────────────
# 8. Extracción por entrevista
# ──────────────────────────────────────────────────────────────────────────────
def analyze_transcript(
    text: str,
    metadata: InterviewMetadata,
    client,
    model: str,
    tracker: UsageTracker,
    prompt_variant: str = "baseline",
    max_items: int = 4,
    temperature: Optional[float] = 0.2,
) -> InterviewResult:
    result = InterviewResult(metadata=metadata, prompt_variant=prompt_variant)
    info = MODEL_CATALOG.get(model)
    system = build_prompt(prompt_variant, max_items)
    est_in = count_tokens(text) + count_tokens(system) + PROMPT_OVERHEAD_TOKENS
    if info and est_in > CONTEXT_SAFETY * info.context:
        result.error = (
            f"La entrada (~{est_in:,} tokens) excede el {int(CONTEXT_SAFETY*100)}% "
            f"del context window de {model} ({info.context:,})."
        )
        log.error(f"  {metadata.Id_entrevista}: {result.error}")
        return result
    try:
        parsed, usage = _call_structured(
            client, model, system,
            USER_TEMPLATE.format(transcript=text),
            QualitativeAnalysisSchema, temperature,
        )
        result.input_tokens, result.output_tokens = tracker.add(usage)

        # Postproceso: limpia nulls y el bug "Original"
        result.Preocupacion_principal_acerca_del_agua = postprocess_items(
            parsed.Preocupacion_principal_acerca_del_agua
        )
        result.Causas_principales = postprocess_items(parsed.Causas_principales)
        result.Consecuencias = postprocess_items(parsed.Consecuencias)
        result.Acciones_y_Actores_que_realizan_dichas_acciones = postprocess_actions(
            [{"actor": a.actor, "accion": a.accion}
             for a in parsed.Acciones_y_Actores_que_realizan_dichas_acciones]
        )
    except Exception as exc:
        log.error(f"  Error analizando {metadata.Id_entrevista}: {exc}")
        result.error = str(exc)
    return result

# ──────────────────────────────────────────────────────────────────────────────
# 9. Salidas: CSV consolidado
# ──────────────────────────────────────────────────────────────────────────────
COL_PREOC  = "Preocupacion_principal_acerca_del_agua"
COL_CAUSAS = "Causas_principales"
COL_CONSEC = "Consecuencias"
COL_ACC    = "Acciones_y_Actores_que_realizan_dichas_acciones"


def _fmt_action(a: dict) -> str:
    return f"{a['actor']}: {a['accion']}"


def results_to_dataframe(results: list[InterviewResult]) -> pd.DataFrame:
    rows = []
    for r in sorted(
        (r for r in results if not r.error),
        key=lambda r: r.metadata.Id_entrevista,
    ):
        rows.append({
            "Id_entrevista": r.metadata.Id_entrevista,
            "prompt_variant": r.prompt_variant,
            COL_PREOC:  "\n".join(r.Preocupacion_principal_acerca_del_agua),
            COL_CAUSAS: "\n".join(r.Causas_principales),
            COL_CONSEC: "\n".join(r.Consecuencias),
            COL_ACC:    "\n".join(_fmt_action(a)
                                  for a in r.Acciones_y_Actores_que_realizan_dichas_acciones),
        })
    return pd.DataFrame(rows)


_checkpoint_lock = threading.Lock()


def save_checkpoints(result: InterviewResult, all_results: list[InterviewResult], out_dir: Path):
    d = out_dir / "individual"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{result.metadata.Id_entrevista}.json").write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    df = results_to_dataframe(all_results)
    if not df.empty:
        df.to_csv(out_dir / "analysis_v7.csv", index=False, encoding="utf-8-sig")


def load_existing_result(out_dir: Path, interview_id: str) -> Optional[InterviewResult]:
    p = out_dir / "individual" / f"{interview_id}.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("error"):
            return None
        d["metadata"] = InterviewMetadata(**d["metadata"])
        return InterviewResult(**d)
    except Exception:
        return None

# ──────────────────────────────────────────────────────────────────────────────
# 10. Pipeline principal
# ──────────────────────────────────────────────────────────────────────────────
def _process_single_file(
    fpath: Path, client, model: str, out_path: Path,
    all_results: list[InterviewResult], tracker: UsageTracker,
    resume: bool, prompt_variant: str, max_items: int,
):
    metadata = parse_interview_id(fpath.name)
    if resume:
        prev = load_existing_result(out_path, metadata.Id_entrevista)
        if prev:
            log.info(f"Omitida (ya procesada): {fpath.name}")
            with _checkpoint_lock:
                all_results.append(prev)
            return
    log.info(f"Procesando: {fpath.name}")
    raw_text = load_transcript(fpath)
    if not raw_text.strip():
        result = InterviewResult(
            metadata=metadata, prompt_variant=prompt_variant, error="Archivo vacío"
        )
    else:
        result = analyze_transcript(
            preprocess(raw_text), metadata, client, model, tracker,
            prompt_variant=prompt_variant, max_items=max_items,
        )
    with _checkpoint_lock:
        all_results.append(result)
        save_checkpoints(result, all_results, out_path)


def make_client():
    from openai import OpenAI
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("Falta OPENAI_API_KEY en el archivo .env")
    return OpenAI(api_key=key)


def run_pipeline(
    input_folder: str,
    output_folder: str,
    model: str,
    prompt_variant: str = "baseline",
    max_items: int = 4,
    max_workers: int = 2,
    resume: bool = True,
    client=None,
) -> list[InterviewResult]:
    """
    Ejecuta la extracción sobre todas las transcripciones de `input_folder`
    y guarda los resultados en `output_folder/analysis_v7.csv`.
    """
    client = client or make_client()
    out_path = Path(output_folder)
    out_path.mkdir(parents=True, exist_ok=True)
    tracker = UsageTracker(model)

    log.info(f"Modelo: {model} | Prompt variant: {prompt_variant} | max_items: {max_items}")
    log.info(f"Salida: {out_path.resolve()}")

    # Guardar metadatos del experimento
    experiment_meta = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "modelo": model,
        "prompt_variant": prompt_variant,
        "max_items": max_items,
        "prompt_text": build_prompt(prompt_variant, max_items),
    }
    (out_path / "experiment_config.json").write_text(
        json.dumps(experiment_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    files = list_transcripts(input_folder)
    if not files:
        raise SystemExit(f"No se encontraron transcripciones en {input_folder}")
    log.info(f"Transcripciones encontradas: {len(files)}")

    all_results: list[InterviewResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_single_file, f, client, model, out_path,
                all_results, tracker, resume, prompt_variant, max_items,
            ): f
            for f in files
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"[{prompt_variant}]"):
            try:
                fut.result()
            except Exception as exc:
                log.error(f"Error fatal en {futures[fut].name}: {exc}")

    failed = [r.metadata.Id_entrevista for r in all_results if r.error]
    if failed:
        log.warning(f"{len(failed)} entrevistas con error: {failed}")

    df = results_to_dataframe(all_results)
    if not df.empty:
        out_csv = out_path / "analysis_v7.csv"
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        log.info(f"CSV consolidado: {out_csv.resolve()} ({len(df)} entrevistas)")
    else:
        log.warning("No se generaron resultados válidos; el CSV no fue creado.")

    # Registro acumulativo de uso
    usage = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "modelo": model,
        "prompt_variant": prompt_variant,
        "tokens_entrada": tracker.input,
        "tokens_salida": tracker.output,
        "costo_usd_estimado": tracker.cost(),
    }
    with open(out_path / "usage_log_v7.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(usage, ensure_ascii=False) + "\n")
    log.info(
        f"Uso de tokens y costo estimado:\n"
        + json.dumps(usage, ensure_ascii=False, indent=2)
    )
    return all_results


def run_all_variants(
    input_folder: str,
    base_out_dir: str,
    model: str,
    max_items: int = 4,
    max_workers: int = 2,
    client=None,
) -> dict[str, list[InterviewResult]]:
    """Ejecuta todas las variantes de prompt en secuencia. Cada una va a su propio subdirectorio."""
    client = client or make_client()
    results_by_variant = {}
    for variant in PROMPT_VARIANTS:
        out = str(Path(base_out_dir) / f"output_{variant}")
        log.info(f"\n{'='*60}\nVariante: {variant}\n{'='*60}")
        results_by_variant[variant] = run_pipeline(
            input_folder, out, model,
            prompt_variant=variant, max_items=max_items,
            max_workers=max_workers, resume=False, client=client,
        )
    # Resumen comparativo de costos
    print("\n" + "="*60)
    print("RESUMEN DE COSTO POR VARIANTE")
    print("="*60)
    for variant in PROMPT_VARIANTS:
        log_path = Path(base_out_dir) / f"output_{variant}" / "usage_log_v7.jsonl"
        if log_path.exists():
            last = json.loads(log_path.read_text(encoding="utf-8").strip().split("\n")[-1])
            print(f"  {variant:25s}  ${last.get('costo_usd_estimado', 0):.4f} USD")
    return results_by_variant

# ──────────────────────────────────────────────────────────────────────────────
# 11. CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Pipeline MEGADAPT v7 — framework de experimentación con variantes de prompt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Variantes de prompt disponibles:
  {chr(10).join(f'  {k:25s}  {PROMPT_VARIANTS[k].strip()[:60]}...' for k in PROMPT_VARIANTS)}

Ejemplos de uso:

  # Estimar costo sin llamar a la API:
  python megadapt_qualitative_pipeline_v7.py \\
      --folder ./transcripciones --count-only

  # Correr una variante:
  python megadapt_qualitative_pipeline_v7.py \\
      --folder ./transcripciones --model gpt-4o-mini \\
      --prompt-variant strict_grounding

  # Correr todas las variantes (una carpeta por variante):
  python megadapt_qualitative_pipeline_v7.py \\
      --folder ./transcripciones --model gpt-4o-mini --run-all-variants

  # Evaluar y comparar con métricas NLP:
  python megadapt_eval_nlp_2.py \\
      --human "Integracion.xlsx" \\
      --llm baseline=output_baseline/analysis_v7.csv \\
             strict_grounding=output_strict_grounding/analysis_v7.csv \\
      --metrics bleu chrf bertscore --xlsx
""",
    )
    ap.add_argument("--folder", help="Carpeta con transcripciones (.docx/.txt/.md)")
    ap.add_argument("--out", default=None,
                    help="Directorio de salida. Por defecto: ./output_<variante>/")
    ap.add_argument("--model", default="gpt-4o-mini",
                    help=f"Modelo de extracción. Opciones: {', '.join(MODEL_CATALOG)}")
    ap.add_argument("--prompt-variant", default="baseline",
                    choices=list(PROMPT_VARIANTS),
                    help="Variante de prompt a usar (default: baseline)")
    ap.add_argument("--max-items", type=int, default=4,
                    help="Número máximo de ítems por campo (default: 4)")
    ap.add_argument("--workers", type=int, default=2,
                    help="Hilos concurrentes (ajustar según rate limit, default: 2)")
    ap.add_argument("--count-only", action="store_true",
                    help="Solo cuenta tokens/costo; no llama a la API")
    ap.add_argument("--models", default=",".join(MODEL_CATALOG),
                    help="Modelos a comparar en --count-only (separados por coma)")
    ap.add_argument("--no-resume", action="store_true",
                    help="Reprocesar todas las entrevistas aunque ya existan sus JSONs")
    ap.add_argument("--run-all-variants", action="store_true",
                    help="Ejecutar todas las variantes en secuencia. --out es el directorio base.")
    args = ap.parse_args()

    if args.count_only:
        if not args.folder:
            ap.error("--count-only requiere --folder")
        unknown = [m for m in args.models.split(",") if m not in MODEL_CATALOG]
        if unknown:
            ap.error(f"Modelos fuera del catálogo: {unknown}")
        count_only(args.folder, args.out or "./count_report",
                   args.models.split(","), variant=args.prompt_variant)
    elif args.run_all_variants:
        if not args.folder:
            ap.error("--run-all-variants requiere --folder")
        run_all_variants(
            args.folder,
            args.out or ".",
            args.model,
            max_items=args.max_items,
            max_workers=args.workers,
        )
    else:
        if not args.folder:
            ap.error("Falta --folder con la ruta a las transcripciones")
        out_dir = args.out or f"./output_{args.prompt_variant}"
        run_pipeline(
            args.folder, out_dir, args.model,
            prompt_variant=args.prompt_variant,
            max_items=args.max_items,
            max_workers=args.workers,
            resume=not args.no_resume,
        )
