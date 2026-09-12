"""
Pipeline de análisis cualitativo automatizado  (v3)
Cambios respecto a v2:
  1. Prompt — no menciona el nombre del proyecto, ni Siqueiros, ni
     "modelos mentales". El LLM trabaja solo desde el texto.
  2. Eje de extracción: CAUSAS / ACCIONES / RESPONSABILIDADES (en lugar de
     "variables genéricas"). Sigue produciendo edgelist.
  3. Diccionario RICO: 1 665 entradas coloquiales → 150 términos canónicos
     (cargado desde dictionary_megadapt_rich.json).
  4. Meta-narrativas: score 0-1 por cada una de las 3 narrativas definidas
     por el equipo; los tres valores suman 1.0 (distribución exclusiva).
  5. Nuevo output: metanarratives_report.json con scores acumulados por
     entrevista y narrativa dominante del corpus.

Dependencias:
  pip install google-genai python-docx pandas tqdm tenacity pydantic python-dotenv openpyxl
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel, Field, model_validator
from tenacity import (
    RetryError,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════════════
# 0. Entorno
# ══════════════════════════════════════════════════════════════════════════════

load_dotenv()

# Evitar conflicto de credenciales con ADC en Windows
if "GEMINI_API_KEY" in os.environ:
    del os.environ["GEMINI_API_KEY"]


# 1. Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
# 2. Carga del diccionario 

def load_dictionary(path: str) -> tuple[dict[str, str], list[str]]:
    """
    Carga el diccionario rico desde un JSON {expresión_coloquial: término_canónico}.
    Devuelve (diccionario, lista_ordenada_de_150_términos_canónicos).

    El JSON se genera con build_rich_dictionary.py a partir del Excel
    Diccionario_New_Last_Version_010619.xlsx.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"No encontré el diccionario en '{path}'.\n"
            "Genera primero el JSON con build_rich_dictionary.py:\n"
            "  python build_rich_dictionary.py --excel Diccionario_New_Last_Version_010619.xlsx"
        )
    with open(p, encoding="utf-8") as f:
        rich_dict: dict[str, str] = json.load(f)

    canonical_terms = sorted(set(rich_dict.values()))
    log.info(
        f"Diccionario cargado: {len(rich_dict)} entradas → {len(canonical_terms)} términos canónicos"
    )
    return rich_dict, canonical_terms


def standardize(term: str, dictionary: dict[str, str]) -> str:
    """Normaliza un término usando el diccionario. Prueba con/sin guiones bajos."""
    t = term.strip()
    if t in dictionary:
        return dictionary[t]
    t_under = t.replace(" ", "_")
    if t_under in dictionary:
        return dictionary[t_under]
    t_space = t.replace("_", " ")
    if t_space in dictionary:
        return dictionary[t_space]
    return t  # sin cambio si no está en el diccionario

# 3. Estructuras de datos

@dataclass
class InterviewMetadata:
    interview_id: str
    activity_type: str          # int | wks | fg
    sector: str = ""
    delegation: str = ""
    source_file: str = ""


@dataclass
class InterviewResult:
    metadata: InterviewMetadata
    causas: list[str]               = field(default_factory=list)
    acciones: list[str]             = field(default_factory=list)
    responsabilidades: list[str]    = field(default_factory=list)
    edgelist: list[tuple[str, str]] = field(default_factory=list)
    meta_infraestructura: float     = 0.0
    meta_gobernanza: float          = 0.0
    meta_marginalidad: float        = 0.0
    meta_justificacion: str         = ""
    summary: str                    = ""
    error: Optional[str]            = None

# 4. Schemas Pydantic para Structured Outputs

class CausalEdge(BaseModel):
    cause: str = Field(..., description="Variable que actúa como causa")
    effect: str = Field(..., description="Variable que actúa como efecto")


class MetaNarrativeAffinity(BaseModel):
    infraestructura_inadecuada: float = Field(
        ..., ge=0.0, le=1.0,
        description="Afinidad con la narrativa de infraestructura inadecuada (0-1)"
    )
    falla_en_gobernanza: float = Field(
        ..., ge=0.0, le=1.0,
        description="Afinidad con la narrativa de falla en gobernanza (0-1)"
    )
    vulnerabilidad_y_marginalidad: float = Field(
        ..., ge=0.0, le=1.0,
        description="Afinidad con la narrativa de vulnerabilidad y marginalidad (0-1)"
    )
    justificacion: str = Field(
        ...,
        description="1-2 oraciones explicando los scores con evidencia del texto"
    )

    @model_validator(mode="after")
    def scores_sum_to_one(self) -> "MetaNarrativeAffinity":
        """Normaliza para que los tres scores sumen exactamente 1.0."""
        total = (
            self.infraestructura_inadecuada
            + self.falla_en_gobernanza
            + self.vulnerabilidad_y_marginalidad
        )
        if total > 0:
            self.infraestructura_inadecuada = round(self.infraestructura_inadecuada / total, 4)
            self.falla_en_gobernanza        = round(self.falla_en_gobernanza / total, 4)
            self.vulnerabilidad_y_marginalidad = round(
                1.0 - self.infraestructura_inadecuada - self.falla_en_gobernanza, 4
            )
        return self


class InterviewAnalysisSchema(BaseModel):
    causas: list[str] = Field(
        ..., min_length=3,
        description="Factores que el participante identifica como origen del problema"
    )
    acciones: list[str] = Field(
        ..., min_length=3,
        description="Respuestas o estrategias mencionadas (de cualquier actor)"
    )
    responsabilidades: list[str] = Field(
        ..., min_length=1,
        description="Actores o instituciones a quienes se atribuye responsabilidad"
    )
    edgelist: list[CausalEdge] = Field(
        ..., min_length=5,
        description="Relaciones causales causa→efecto"
    )
    meta_narrative_affinity: MetaNarrativeAffinity
    summary: str = Field(
        ...,
        description="Párrafo de 3-5 oraciones sobre la narrativa central del participante"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5. Lectura e identificación de transcripciones
# ══════════════════════════════════════════════════════════════════════════════

def read_docx(path: Path) -> str:
    from docx import Document
    doc = Document(str(path))
    return "\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())


def load_transcript(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".docx":
        return read_docx(path)
    elif ext in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="replace")
    else:
        log.warning(f"Formato no soportado: {path.name}")
        return ""


def parse_interview_id(filename: str) -> InterviewMetadata:
    stem = Path(filename).stem.upper()
    stem = re.sub(r"__T-.*", "", stem)

    activity = (
        "wks" if ("WKS" in stem or "TALLER" in stem)
        else "fg" if ("FG" in stem or "FOCAL" in stem)
        else "int"
    )

    sector_map = {
        "GOV": "Gobierno", "GOB": "Gobierno",
        "ACA": "Académico", "OTR": "Otro/ONG", "RES": "Residente",
    }
    sector = next(
        (v for k, v in sector_map.items() if k in stem), "Desconocido"
    )

    geo_map = {
        "FED": "Federal",
        "MC":  "Magdalena Contreras",
        "DF":  "Ciudad de México (gobierno)",
        "I":   "Iztapalapa",
        "X":   "Xochimilco",
    }
    delegation = next(
        (v for k, v in geo_map.items() if stem.startswith(k)), "Desconocida"
    )

    return InterviewMetadata(
        interview_id=stem,
        activity_type=activity,
        sector=sector,
        delegation=delegation,
        source_file=filename,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 6. Preprocesamiento de texto
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(text: str, max_chars: int = 30_000) -> str:
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"E:\s*", "Entrevistador: ", text)
    text = re.sub(r"R:\s*", "Respondente: ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > max_chars:
        log.warning(f"Texto truncado a {max_chars} caracteres.")
        text = text[:max_chars] + "\n[...TEXTO TRUNCADO...]"
    return text


# ══════════════════════════════════════════════════════════════════════════════
# 7. Prompts — CIEGOS (sin nombre de proyecto, sin Siqueiros, sin jerga de MM)
# ══════════════════════════════════════════════════════════════════════════════

def build_context(canonical_terms: list[str]) -> str:
    """
    Construye el system context a partir de los 150 términos canónicos.
    Completamente ciego al nombre del proyecto y a la literatura de MM.
    """
    terms_block = "\n".join(f"  - {t}" for t in canonical_terms)

    return f"""\
Eres un asistente especializado en análisis cualitativo de entrevistas
sobre gestión del agua y riesgo urbano en ciudades de América Latina.

Tu tarea es analizar la transcripción de una entrevista con un actor
(residente, funcionario público o académico) para identificar su visión
sobre los problemas de agua e inundaciones en su ciudad.

Extrae exactamente tres dimensiones de análisis:

  1. CAUSAS — ¿Qué factores, condiciones o procesos identifica el participante
     como origen de los problemas de agua e inundaciones?

  2. ACCIONES — ¿Qué respuestas, medidas o estrategias menciona el participante
     (propias, de vecinos, de gobierno u otras instituciones)?

  3. RESPONSABILIDADES — ¿A qué actores o instituciones atribuye el participante
     la obligación de resolver o gestionar estos problemas?

Identifica también las RELACIONES CAUSALES entre variables
(pares causa → efecto, explícitos o implícitos en el texto).

VOCABULARIO DE REFERENCIA
Al nombrar variables y conceptos, utiliza preferentemente los siguientes
términos estandarizados. Si el participante usa una expresión diferente para
el mismo concepto, normalízala al término correspondiente de esta lista:

{terms_block}

Si identificas un concepto que no aparece en la lista, usa la frase más
corta y descriptiva posible, en español, sin siglas ni nombres propios de proyectos.\
"""


PROMPT_INDIVIDUAL = """\
{context}

Transcripción de entrevista individual
Sector del participante: {sector} | Zona: {delegation}

=== TRANSCRIPCIÓN ===
{transcript}
=== FIN DE TRANSCRIPCIÓN ===

Analiza la transcripción y devuelve ÚNICAMENTE un objeto JSON con esta estructura:

{{
  "causas": [
    // 5-15 variables o condiciones que el participante identifica como ORIGEN del problema.
    // Usa los términos canónicos del vocabulario de referencia cuando aplique.
  ],
  "acciones": [
    // 5-15 respuestas, medidas o estrategias mencionadas por cualquier actor.
  ],
  "responsabilidades": [
    // 3-10 actores o instituciones a quienes el participante atribuye responsabilidad.
    // Solo el nombre del actor (ej. "SACMEX", "delegación", "vecinos", "gobierno federal").
  ],
  "edgelist": [
    // 10-30 relaciones causales. Usa términos del vocabulario de referencia.
    {{"cause": "variable_causa", "effect": "variable_efecto"}}
  ],
  "meta_narrative_affinity": {{
    // Distribución de la narrativa del participante entre las tres visiones.
    // Los tres valores deben sumar exactamente 1.0.
    "infraestructura_inadecuada": 0.0,      // La crisis viene de infraestructura deficiente
    "falla_en_gobernanza": 0.0,             // La crisis viene de corrupción o fracaso institucional
    "vulnerabilidad_y_marginalidad": 0.0,   // La crisis viene de desigualdad y desarrollo informal
    "justificacion": "1-2 oraciones con evidencia concreta del texto que justifiquen los scores."
  }},
  "summary": "Párrafo de 3-5 oraciones describiendo la narrativa central del participante:
              qué ve como causa raíz, qué acciones valora, a quién responsabiliza."
}}
"""


PROMPT_COLECTIVO = """\
{context}

Transcripción de {activity_label} (múltiples participantes)
Sector predominante: {sector} | Zona: {delegation}

=== TRANSCRIPCIÓN ===
{transcript}
=== FIN DE TRANSCRIPCIÓN ===

Al ser una sesión colectiva, presta especial atención a:
- CONSENSOS: qué visiones son compartidas por el grupo.
- TENSIONES: dónde hay desacuerdo sobre causas, acciones o responsabilidades.
- NARRATIVA DOMINANTE: qué postura emerge con más fuerza del colectivo.

Devuelve ÚNICAMENTE un objeto JSON con esta estructura:

{{
  "causas": [
    // 8-20 variables identificadas colectivamente como ORIGEN del problema.
  ],
  "acciones": [
    // 8-20 respuestas o estrategias mencionadas en la sesión.
  ],
  "responsabilidades": [
    // 5-12 actores o instituciones a quienes el grupo atribuye responsabilidad.
    // Solo el nombre del actor.
  ],
  "edgelist": [
    // 15-40 relaciones causales construidas o validadas en la sesión.
    {{"cause": "variable_causa", "effect": "variable_efecto"}}
  ],
  "meta_narrative_affinity": {{
    // Los tres valores deben sumar exactamente 1.0.
    "infraestructura_inadecuada": 0.0,
    "falla_en_gobernanza": 0.0,
    "vulnerabilidad_y_marginalidad": 0.0,
    "justificacion": "2-3 oraciones mencionando consensos y tensiones del grupo."
  }},
  "summary": "Párrafo de 4-6 oraciones sobre narrativas dominantes, consensos y tensiones del grupo."
}}
"""


# ══════════════════════════════════════════════════════════════════════════════
# 8. Exponential backoff con Tenacity
# ══════════════════════════════════════════════════════════════════════════════

def _is_retryable(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(kw in msg for kw in ("503", "unavailable", "429", "rate", "quota", "resource_exhausted"))


def _build_retry_decorator():
    return retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(6),
        before_sleep=before_sleep_log(log, logging.WARNING),
        reraise=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 9. Llamada a la API con Structured Outputs
# ══════════════════════════════════════════════════════════════════════════════

def _call_api(
    client: genai.Client,
    model_name: str,
    prompt: str,
) -> InterviewAnalysisSchema:
    retry_call = _build_retry_decorator()

    @retry_call
    def _do_request():
        return client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=InterviewAnalysisSchema,
                temperature=0.2,
            ),
        )

    response = _do_request()
    parsed: InterviewAnalysisSchema = response.parsed
    if parsed is None:
        raw_text = response.text or ""
        parsed = InterviewAnalysisSchema.model_validate_json(raw_text)
    return parsed


# ══════════════════════════════════════════════════════════════════════════════
# 10. Análisis de una transcripción
# ══════════════════════════════════════════════════════════════════════════════

def analyze_transcript(
    text: str,
    metadata: InterviewMetadata,
    client: genai.Client,
    model_name: str,
    dictionary: dict[str, str],
    canonical_terms: list[str],
) -> InterviewResult:

    is_collective = metadata.activity_type in ("fg", "wks")
    activity_label = (
        "grupo focal" if metadata.activity_type == "fg"
        else "taller participativo" if metadata.activity_type == "wks"
        else "entrevista individual"
    )

    context = build_context(canonical_terms)
    template = PROMPT_COLECTIVO if is_collective else PROMPT_INDIVIDUAL
    prompt = template.format(
        context=context,
        activity_label=activity_label,
        sector=metadata.sector,
        delegation=metadata.delegation,
        transcript=text,
    )

    result = InterviewResult(metadata=metadata)

    try:
        parsed = _call_api(client, model_name, prompt)

        # Estandarizar con diccionario rico
        result.causas         = [standardize(v, dictionary) for v in parsed.causas]
        result.acciones       = [standardize(v, dictionary) for v in parsed.acciones]
        result.responsabilidades = list(dict.fromkeys(parsed.responsabilidades))  # dedup

        result.edgelist = [
            (standardize(e.cause, dictionary), standardize(e.effect, dictionary))
            for e in parsed.edgelist
        ]

        aff = parsed.meta_narrative_affinity
        result.meta_infraestructura = aff.infraestructura_inadecuada
        result.meta_gobernanza      = aff.falla_en_gobernanza
        result.meta_marginalidad    = aff.vulnerabilidad_y_marginalidad
        result.meta_justificacion   = aff.justificacion
        result.summary              = parsed.summary

        log.info(
            f"✓ {metadata.interview_id} ({activity_label}): "
            f"{len(result.causas)} causas, {len(result.acciones)} acciones, "
            f"{len(result.edgelist)} relaciones | "
            f"INF={result.meta_infraestructura:.2f} "
            f"GOB={result.meta_gobernanza:.2f} "
            f"MAR={result.meta_marginalidad:.2f}"
        )

    except RetryError as exc:
        result.error = f"API no disponible tras reintentos: {exc}"
        log.error(f"✗ {metadata.interview_id}: {result.error}")
    except Exception as exc:
        result.error = f"Error inesperado: {exc}"
        log.error(f"✗ {metadata.interview_id}: {result.error}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 11. Exportación de resultados individuales
# ══════════════════════════════════════════════════════════════════════════════

def save_edgelist(result: InterviewResult, out_dir: Path) -> None:
    if not result.edgelist:
        return
    out_path = out_dir / "edgelists" / f"{result.metadata.interview_id}_edgelist.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for cause, effect in result.edgelist:
            f.write(f"{cause},{effect}\n")


def result_to_dict(r: InterviewResult) -> dict:
    d = asdict(r.metadata)
    d["causas"]            = r.causas
    d["acciones"]          = r.acciones
    d["responsabilidades"] = r.responsabilidades
    d["edgelist"]          = [{"cause": c, "effect": e} for c, e in r.edgelist]
    d["meta_narrative_affinity"] = {
        "infraestructura_inadecuada":  r.meta_infraestructura,
        "falla_en_gobernanza":         r.meta_gobernanza,
        "vulnerabilidad_y_marginalidad": r.meta_marginalidad,
        "justificacion":               r.meta_justificacion,
    }
    d["summary"] = r.summary
    d["error"]   = r.error
    return d


# ══════════════════════════════════════════════════════════════════════════════
# 12. Checkpointing progresivo
# ══════════════════════════════════════════════════════════════════════════════

def checkpoint_summary_json(result: InterviewResult, out_dir: Path) -> None:
    out_path = Path(out_dir) / "analysis_summary.json"
    existing: list[dict] = []
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = []
    new_entry = result_to_dict(result)
    existing = [e for e in existing if e.get("interview_id") != new_entry["interview_id"]]
    existing.append(new_entry)
    out_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")


def checkpoint_variables_csv(result: InterviewResult, out_dir: Path) -> None:
    """Guarda causas, acciones y responsabilidades en un CSV largo."""
    if not (result.causas or result.acciones or result.responsabilidades):
        return
    out_path = Path(out_dir) / "variables_table.csv"

    rows = (
        [{"interview_id": result.metadata.interview_id,
          "sector": result.metadata.sector,
          "delegation": result.metadata.delegation,
          "activity": result.metadata.activity_type,
          "dimension": "causa",
          "variable": v} for v in result.causas]
        + [{"interview_id": result.metadata.interview_id,
            "sector": result.metadata.sector,
            "delegation": result.metadata.delegation,
            "activity": result.metadata.activity_type,
            "dimension": "accion",
            "variable": v} for v in result.acciones]
        + [{"interview_id": result.metadata.interview_id,
            "sector": result.metadata.sector,
            "delegation": result.metadata.delegation,
            "activity": result.metadata.activity_type,
            "dimension": "responsabilidad",
            "variable": v} for v in result.responsabilidades]
    )
    new_rows = pd.DataFrame(rows)

    if out_path.exists():
        existing = pd.read_csv(out_path, encoding="utf-8-sig")
        existing = existing[existing["interview_id"] != result.metadata.interview_id]
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows
    combined.to_csv(out_path, index=False, encoding="utf-8-sig")


def checkpoint_global_edgelist(all_results: list[InterviewResult], out_dir: Path) -> None:
    edge_counts: dict[tuple[str, str], int] = {}
    for r in all_results:
        for edge in r.edgelist:
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    rows = [
        {"cause": c, "effect": e, "frequency": n}
        for (c, e), n in sorted(edge_counts.items(), key=lambda x: -x[1])
    ]
    if not rows:
        return
    out_path = Path(out_dir) / "global_edgelist.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")


def checkpoint_metanarratives(all_results: list[InterviewResult], out_dir: Path) -> None:
    """
    Construye y guarda metanarratives_report.json con:
    - Scores promedio de cada narrativa en el corpus completo.
    - Narrativa dominante por entrevista.
    - Narrativa dominante del corpus.
    - Top-10 relaciones causales de cada narrativa (las más frecuentes en
      entrevistas donde esa narrativa es dominante).
    """
    valid = [r for r in all_results if not r.error]
    if not valid:
        return

    # Scores por entrevista
    per_interview = []
    dominant_counts = {"infraestructura_inadecuada": 0, "falla_en_gobernanza": 0,
                       "vulnerabilidad_y_marginalidad": 0}

    edges_by_narrative: dict[str, dict[tuple[str, str], int]] = {
        "infraestructura_inadecuada": {},
        "falla_en_gobernanza": {},
        "vulnerabilidad_y_marginalidad": {},
    }

    for r in valid:
        scores = {
            "infraestructura_inadecuada":  r.meta_infraestructura,
            "falla_en_gobernanza":         r.meta_gobernanza,
            "vulnerabilidad_y_marginalidad": r.meta_marginalidad,
        }
        dominant = max(scores, key=scores.__getitem__)
        dominant_counts[dominant] += 1

        per_interview.append({
            "interview_id": r.metadata.interview_id,
            "sector":       r.metadata.sector,
            "delegation":   r.metadata.delegation,
            "scores":       scores,
            "dominant":     dominant,
            "justificacion": r.meta_justificacion,
        })

        # Acumular edges bajo la narrativa dominante de esta entrevista
        bucket = edges_by_narrative[dominant]
        for edge in r.edgelist:
            bucket[edge] = bucket.get(edge, 0) + 1

    # Promedios del corpus
    n = len(valid)
    corpus_avg = {
        "infraestructura_inadecuada":  round(sum(r.meta_infraestructura for r in valid) / n, 4),
        "falla_en_gobernanza":         round(sum(r.meta_gobernanza for r in valid) / n, 4),
        "vulnerabilidad_y_marginalidad": round(sum(r.meta_marginalidad for r in valid) / n, 4),
    }
    corpus_dominant = max(dominant_counts, key=dominant_counts.__getitem__)

    # Top-10 relaciones por narrativa
    top_edges = {}
    for narrative, bucket in edges_by_narrative.items():
        top = sorted(bucket.items(), key=lambda x: -x[1])[:10]
        top_edges[narrative] = [
            {"cause": c, "effect": e, "frequency": n_}
            for (c, e), n_ in top
        ]

    report = {
        "corpus_summary": {
            "total_interviews_analyzed": n,
            "average_scores": corpus_avg,
            "dominant_narrative_corpus": corpus_dominant,
            "dominant_counts": dominant_counts,
        },
        "top_causal_relations_by_narrative": top_edges,
        "per_interview": per_interview,
    }

    out_path = Path(out_dir) / "metanarratives_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"  → metanarratives_report.json actualizado ({n} entrevistas válidas)")


def save_all_checkpoints(
    result: InterviewResult,
    all_results: list[InterviewResult],
    out_dir: Path,
) -> None:
    save_edgelist(result, out_dir)
    checkpoint_summary_json(result, out_dir)
    if not result.error:
        checkpoint_variables_csv(result, out_dir)
        checkpoint_global_edgelist(all_results, out_dir)
        checkpoint_metanarratives(all_results, out_dir)


# ══════════════════════════════════════════════════════════════════════════════
# 13. Pipeline principal
# ══════════════════════════════════════════════════════════════════════════════

_checkpoint_lock = threading.Lock()


def _process_single_file(
    fpath: Path,
    client: genai.Client,
    model_name: str,
    dictionary: dict[str, str],
    canonical_terms: list[str],
    out_path: Path,
    all_results: list[InterviewResult],
) -> InterviewResult:
    log.info(f"Procesando: {fpath.name}")

    metadata = parse_interview_id(fpath.name)
    metadata.source_file = str(fpath)

    raw_text = load_transcript(fpath)
    if not raw_text.strip():
        log.warning(f"  Archivo vacío o no legible: {fpath.name}")
        result = InterviewResult(metadata=metadata, error="Archivo vacío o no legible")
    else:
        clean_text = preprocess(raw_text)
        result = analyze_transcript(
            clean_text, metadata, client, model_name, dictionary, canonical_terms
        )

    with _checkpoint_lock:
        all_results.append(result)
        save_all_checkpoints(result, all_results, out_path)

    return result


def run_pipeline(
    input_folder: str,
    output_folder: str,
    api_key: str,
    dictionary_path: str = "dictionary_megadapt_rich.json",
    model_name: str = "gemini-2.0-flash",
    limit: Optional[int] = None,
    max_workers: int = 2,
) -> None:
    client = genai.Client(api_key=api_key)
    log.info(f"Modelo: {model_name} | Hilos: {max_workers}")

    dictionary, canonical_terms = load_dictionary(dictionary_path)

    in_path = Path(input_folder)
    files = sorted(f for f in in_path.rglob("*") if f.suffix.lower() in (".docx", ".txt"))
    if limit:
        files = files[:limit]
    log.info(f"Transcripciones encontradas: {len(files)}")

    out_path = Path(output_folder)
    out_path.mkdir(parents=True, exist_ok=True)

    all_results: list[InterviewResult] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_single_file,
                fpath, client, model_name, dictionary, canonical_terms, out_path, all_results,
            ): fpath
            for fpath in files
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Analizando"):
            fpath = futures[future]
            try:
                future.result()
            except Exception as exc:
                log.error(f"Error fatal procesando {fpath.name}: {exc}")

    errors = [r for r in all_results if r.error]
    if errors:
        log.warning(f"\n{len(errors)} transcripciones con error:")
        for r in errors:
            log.warning(f"  • {r.metadata.interview_id}: {r.error}")

    log.info(
        f"\n✓ Pipeline completado. "
        f"{len(all_results) - len(errors)}/{len(all_results)} procesadas correctamente.\n"
        f"  Salida en: {out_path.resolve()}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 14. Punto de entrada
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Pipeline de análisis cualitativo automatizado (v3)"
    )
    parser.add_argument("--folder", "-f", required=True,
        help="Carpeta con transcripciones (.docx o .txt)")
    parser.add_argument("--out", "-o", default="./output_v3",
        help="Carpeta de salida (default: ./output_v3)")
    parser.add_argument("--api-key", "-k",
        default=os.environ.get("GEMINI_API_KEY", ""),
        help="API key de Gemini (o variable de entorno GEMINI_API_KEY)")
    parser.add_argument("--dictionary", "-d",
        default="dictionary_megadapt_rich.json",
        help="Ruta al diccionario rico JSON (default: dictionary_megadapt_rich.json)")
    parser.add_argument("--model",
        default="gemini-3.6-flash",
        help="Modelo Gemini a usar (default: gemini-3.6-flash)")
    parser.add_argument("--limit", type=int, default=None,
        help="Número máximo de transcripciones a procesar (para pruebas)")
    parser.add_argument("--workers", type=int, default=2,
        help="Número de hilos paralelos (default: 2)")

    args = parser.parse_args()

    if not args.api_key:
        parser.error(
            "Proporciona tu API key con --api-key o exporta GEMINI_API_KEY.\n"
            "  export GEMINI_API_KEY='tu_api_key_aqui'"
        )

    run_pipeline(
        input_folder   = args.folder,
        output_folder  = args.out,
        api_key        = args.api_key,
        dictionary_path= args.dictionary,
        model_name     = args.model,
        limit          = args.limit,
        max_workers    = args.workers,
    )


if __name__ == "__main__":
    main()
