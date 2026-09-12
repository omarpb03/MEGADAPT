"""
Pipeline de análisis cualitativo automatizado  (v4)
====================================================
Cambio de objetivo respecto a v3: ya NO se construyen modelos mentales
(ni edgelist, ni diccionario rico de 150 términos, ni meta-narrativas).

El nuevo objetivo es replicar el esquema de codificación cualitativa que
usó el equipo de la Dra. Hallie (ver Análisis_Cualitativo_de_Entrevistas_MC.xlsx),
para poder comparar —entrevista por entrevista— lo que produce el LLM contra
lo que codificaron a mano. Las 4 dimensiones (columnas H, I, J, K del Excel
original) son:

  1. PREOCUPACIÓN PRINCIPAL ACERCA DEL AGUA (texto corto)
  2. CAUSAS PRINCIPALES, categorizadas en 4 subcategorías:
        - Biofísicas
        - Socio-institucionales#pendiente de revisar
        - Uso de suelo
        - Infraestructurales
  3. MAYORES CONSECUENCIAS
  4. ACCIONES (especificando qué actor las realiza)

El resto de la infraestructura (lectura de .docx/.txt, inferencia de
metadata desde el nombre de archivo, reintentos con backoff, structured
outputs con Pydantic, checkpointing progresivo, paralelismo conservador)
se mantiene igual que en v2/v3.

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
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════════════
# 0. Entorno / logging
# ══════════════════════════════════════════════════════════════════════════════

load_dotenv()

if "GOOGLE_API_KEY" in os.environ:
    del os.environ["GOOGLE_API_KEY"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Estructuras de datos
# ══════════════════════════════════════════════════════════════════════════════

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
    preocupacion_principal: str            = ""
    causas_biofisicas: list[str]           = field(default_factory=list)
    causas_socio_institucionales: list[str] = field(default_factory=list)
    causas_uso_de_suelo: list[str]         = field(default_factory=list)
    causas_infraestructurales: list[str]   = field(default_factory=list)
    consecuencias: list[str]               = field(default_factory=list)
    acciones: list[str]                    = field(default_factory=list)
    error: Optional[str]                   = None


# ══════════════════════════════════════════════════════════════════════════════
# 2. Schema Pydantic para Structured Outputs
# ══════════════════════════════════════════════════════════════════════════════

class CausasCategorizadas(BaseModel):#modificar 
    biofisicas: list[str] = Field(
        default_factory=list,
        description="Causas relacionadas con lluvia, topografía, hidrología, clima, suelo natural.",
    )
    socio_institucionales: list[str] = Field(
        default_factory=list,
        description="Causas relacionadas con gobernanza, presupuesto, corrupción, gestión "
                     "institucional, desconfianza, competencia política, falta de coordinación.",
    )
    uso_de_suelo: list[str] = Field(
        default_factory=list,
        description="Causas relacionadas con urbanización, asentamientos formales/informales, "
                     "cambio de uso de suelo, crecimiento urbano, deforestación.",
    )
    infraestructurales: list[str] = Field(
        default_factory=list,
        description="Causas relacionadas con tuberías, drenaje, redes de distribución, "
                     "mantenimiento, infraestructura física deficiente o vieja.",
    )


class QualitativeAnalysisSchema(BaseModel):
    preocupacion_principal: str = Field(
        ...,
        description="1-3 oraciones: la preocupación central del participante acerca del agua "
                     "(escasez, calidad, inundaciones, gestión, etc.).",
    )
    causas: CausasCategorizadas = Field(
        ...,
        description="Causas principales que el participante identifica como origen del problema, "
                     "categorizadas en las 4 subcategorías. Puede haber 0 causas en una subcategoría "
                     "si el participante no la menciona, pero al menos una subcategoría debe tener contenido.",
    )
    consecuencias: list[str] = Field(
        ..., min_length=1,
        description="Mayores consecuencias o impactos mencionados (ambientales, sociales, económicos, de salud).",
    )
    acciones: list[str] = Field(
        ..., min_length=1,
        description="Acciones, respuestas o estrategias mencionadas, especificando SIEMPRE el actor "
                     "que las realiza al inicio de cada frase, ej. 'Vecinos: organizan limpiezas', "
                     "'Delegación: da mantenimiento a tuberías'.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. Lectura e identificación de transcripciones (sin cambios respecto a v3)
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


def normalize_interview_code(code: str) -> str:
    """Normaliza un código de entrevista para poder comparar formatos inconsistentes
    (con/sin guión bajo, mayúsculas/minúsculas) entre el Excel humano y los nombres
    de archivo. Ej: 'MC_022515_GOV' y 'MC022515_gov' → 'MC022515GOV'."""
    return re.sub(r"[^A-Za-z0-9]", "", code).upper()


# ══════════════════════════════════════════════════════════════════════════════
# 4. Preprocesamiento de texto (sin cambios)
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
# 5. Prompts — ciegos al nombre del proyecto / literatura de modelos mentales
# ══════════════════════════════════════════════════════════════════════════════

CONTEXT = """\
Eres un analista cualitativo especializado en entrevistas sobre gestión del
agua y riesgo urbano (inundaciones, escasez, contaminación) en ciudades de
América Latina.

Tu tarea es leer la transcripción de una entrevista con un actor (residente,
funcionario público, académico u otro) y codificarla siguiendo EXACTAMENTE
el mismo esquema que usaría un investigador humano en un análisis de campo:

  1. PREOCUPACIÓN PRINCIPAL ACERCA DEL AGUA
     ¿Cuál es, en 1-3 oraciones, la preocupación central del participante
     sobre el agua? (puede ser escasez, calidad, inundaciones, gestión,
     conflicto, etc.)

  2. CAUSAS PRINCIPALES, categorizadas en:
     - Biofísicas (lluvia, topografía, hidrología, clima, suelo natural)
     - Socio-institucionales (gobernanza, presupuesto, corrupción, gestión
       institucional, desconfianza, competencia política, descoordinación)
     - Uso de suelo (urbanización, asentamientos formales/informales,
       cambio de uso de suelo, crecimiento urbano, deforestación)
     - Infraestructurales (tuberías, drenaje, redes, mantenimiento,
       infraestructura física vieja o insuficiente)

  3. MAYORES CONSECUENCIAS
     Impactos ambientales, sociales, económicos o de salud que el
     participante atribuye al problema.

  4. ACCIONES
     Respuestas, medidas o estrategias mencionadas (propias, de vecinos,
     de gobierno u otras instituciones). Especifica SIEMPRE el actor al
     inicio de cada frase (ej. "Vecinos: ...", "SACMEX: ...", "Delegación: ...").

Usa frases cortas y concretas, en español, tal como las expresaría el
participante — no generalices ni añadas interpretación que no esté en el
texto. Si el participante no menciona nada para una subcategoría de causas,
déjala vacía; no inventes contenido.\
"""

PROMPT_INDIVIDUAL = """\
{context}

Transcripción de entrevista individual
Sector del participante: {sector} | Zona: {delegation}

=== TRANSCRIPCIÓN ===
{transcript}
=== FIN DE TRANSCRIPCIÓN ===

Devuelve ÚNICAMENTE un objeto JSON con esta estructura:

{{
  "preocupacion_principal": "...",
  "causas": {{
    "biofisicas": [...],
    "socio_institucionales": [...],
    "uso_de_suelo": [...],
    "infraestructurales": [...]
  }},
  "consecuencias": [...],
  "acciones": [...]
}}
"""

PROMPT_COLECTIVO = """\
{context}

Transcripción de {activity_label} (múltiples participantes)
Sector predominante: {sector} | Zona: {delegation}

=== TRANSCRIPCIÓN ===
{transcript}
=== FIN DE TRANSCRIPCIÓN ===

Al ser una sesión colectiva, prioriza las visiones compartidas por el grupo
y, si hay desacuerdos relevantes sobre causas, consecuencias o acciones,
inclúyelos igual (no los promedies ni los borres).

Devuelve ÚNICAMENTE un objeto JSON con esta estructura:

{{
  "preocupacion_principal": "...",
  "causas": {{
    "biofisicas": [...],
    "socio_institucionales": [...],
    "uso_de_suelo": [...],
    "infraestructurales": [...]
  }},
  "consecuencias": [...],
  "acciones": [...]
}}
"""


# ══════════════════════════════════════════════════════════════════════════════
# 6. Retry con backoff exponencial
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
# 7. Llamada a la API con Structured Outputs
# ══════════════════════════════════════════════════════════════════════════════

def _call_api(client: genai.Client, model_name: str, prompt: str) -> QualitativeAnalysisSchema:
    retry_call = _build_retry_decorator()

    @retry_call
    def _do_request():
        return client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=QualitativeAnalysisSchema,
                temperature=0.2,
            ),
        )

    response = _do_request()
    parsed: QualitativeAnalysisSchema = response.parsed
    if parsed is None:
        raw_text = response.text or ""
        parsed = QualitativeAnalysisSchema.model_validate_json(raw_text)
    return parsed


# ══════════════════════════════════════════════════════════════════════════════
# 8. Análisis de una transcripción
# ══════════════════════════════════════════════════════════════════════════════

def analyze_transcript(
    text: str,
    metadata: InterviewMetadata,
    client: genai.Client,
    model_name: str,
) -> InterviewResult:

    is_collective = metadata.activity_type in ("fg", "wks")
    activity_label = (
        "grupo focal" if metadata.activity_type == "fg"
        else "taller participativo" if metadata.activity_type == "wks"
        else "entrevista individual"
    )

    template = PROMPT_COLECTIVO if is_collective else PROMPT_INDIVIDUAL
    prompt = template.format(
        context=CONTEXT,
        activity_label=activity_label,
        sector=metadata.sector,
        delegation=metadata.delegation,
        transcript=text,
    )

    result = InterviewResult(metadata=metadata)

    try:
        parsed = _call_api(client, model_name, prompt)
        result.preocupacion_principal = parsed.preocupacion_principal
        result.causas_biofisicas = parsed.causas.biofisicas
        result.causas_socio_institucionales = parsed.causas.socio_institucionales
        result.causas_uso_de_suelo = parsed.causas.uso_de_suelo
        result.causas_infraestructurales = parsed.causas.infraestructurales
        result.consecuencias = parsed.consecuencias
        result.acciones = parsed.acciones
    except Exception as exc:
        log.error(f"  Error analizando {metadata.interview_id}: {exc}")
        result.error = str(exc)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 9. Checkpointing progresivo
# ══════════════════════════════════════════════════════════════════════════════

def save_individual_json(result: InterviewResult, out_dir: Path) -> None:
    d = out_dir / "individual"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{result.metadata.interview_id}.json"
    path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")


def checkpoint_consolidated_csv(all_results: list[InterviewResult], out_dir: Path) -> None:
    rows = []
    for r in all_results:
        if r.error:
            continue
        rows.append({
            "interview_id": r.metadata.interview_id,
            "sector": r.metadata.sector,
            "delegation": r.metadata.delegation,
            "activity_type": r.metadata.activity_type,
            "source_file": r.metadata.source_file,
            "preocupacion_principal": r.preocupacion_principal,
            "causas_biofisicas": "\n".join(r.causas_biofisicas),
            "causas_socio_institucionales": "\n".join(r.causas_socio_institucionales),
            "causas_uso_de_suelo": "\n".join(r.causas_uso_de_suelo),
            "causas_infraestructurales": "\n".join(r.causas_infraestructurales),
            "consecuencias": "\n".join(r.consecuencias),
            "acciones": "\n".join(r.acciones),
        })
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "analysis_summary_v4.csv", index=False, encoding="utf-8-sig")


def save_all_checkpoints(result: InterviewResult, all_results: list[InterviewResult], out_dir: Path) -> None:
    save_individual_json(result, out_dir)
    checkpoint_consolidated_csv(all_results, out_dir)


# ══════════════════════════════════════════════════════════════════════════════
# 10. Pipeline principal
# ══════════════════════════════════════════════════════════════════════════════

_checkpoint_lock = threading.Lock()


def _process_single_file(
    fpath: Path,
    client: genai.Client,
    model_name: str,
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
        result = analyze_transcript(clean_text, metadata, client, model_name)

    with _checkpoint_lock:
        all_results.append(result)
        save_all_checkpoints(result, all_results, out_path)

    return result


def run_pipeline(
    input_folder: str,
    output_folder: str,
    api_key: str,
    model_name: str = "gemini-2.0-flash",
    limit: Optional[int] = None,
    max_workers: int = 2,
) -> None:
    client = genai.Client(api_key=api_key)
    log.info(f"Modelo: {model_name} | Hilos: {max_workers}")

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
            executor.submit(_process_single_file, fpath, client, model_name, out_path, all_results): fpath
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
        f"  Salida en: {out_path.resolve()}\n"
        f"  Consolidado: {out_path / 'analysis_summary_v4.csv'}\n"
        f"  Siguiente paso: python compare_with_human_coding.py "
        f"--human-excel Análisis_Cualitativo_de_Entrevistas_MC.xlsx "
        f"--llm-csv {out_path / 'analysis_summary_v4.csv'}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 11. Punto de entrada
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline de análisis cualitativo automatizado (v4)")
    parser.add_argument("--folder", "-f", required=True,
        help="Carpeta con transcripciones (.docx o .txt)")
    parser.add_argument("--out", "-o", default="./output_v4",
        help="Carpeta de salida (default: ./output_v4)")
    parser.add_argument("--api-key", "-k",
        default=os.environ.get("GEMINI_API_KEY", ""),
        help="API key de Gemini (o variable de entorno GEMINI_API_KEY)")
    parser.add_argument("--model",
        default="gemini-2.0-flash",
        help="Modelo Gemini a usar (default: gemini-2.0-flash)")
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
        input_folder  = args.folder,
        output_folder = args.out,
        api_key       = args.api_key,
        model_name    = args.model,
        limit         = args.limit,
        max_workers   = args.workers,
    )


if __name__ == "__main__":
    main()
