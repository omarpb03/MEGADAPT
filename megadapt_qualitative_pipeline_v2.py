"""
MEGADAPT – Pipeline de análisis cualitativo automatizado  (v2 – refactorizado)
Mejoras sobre v1:
  1. Exponential Backoff con `tenacity` para errores 503 / 429
  2. Structured Outputs vía Pydantic + response_schema de google.genai
  3. Guardado progresivo (checkpointing) tras cada transcripción exitosa
  4. Paralelismo conservador con ThreadPoolExecutor (max_workers=2)

Dependencias:
  pip install google-genai python-docx pandas tqdm tenacity pydantic python-dotenv
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
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
    RetryError,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
from tqdm import tqdm

# 
# 0. Entorno
# 

load_dotenv()

# Evitar conflicto de credenciales con ADC en Windows
if "GOOGLE_API_KEY" in os.environ:
    del os.environ["GOOGLE_API_KEY"]

# 
# 1. Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# 
# 2. Constantes de dominio
# 

MEGADAPT_CONTEXT = """
Eres un asistente de investigación especializado en el análisis cualitativo del
proyecto MEGADAPT, que estudia la vulnerabilidad ante inundaciones, escasez de agua
y riesgo hídrico en la Ciudad de México. Los participantes son residentes de colonias
vulnerables (Iztapalapa, Xochimilco, Magdalena Contreras), funcionarios de gobierno
(SACMEX, SEDEMA, CONAGUA, Protección Civil, CENAPRED) y académicos.

Tu tarea es analizar transcripciones de entrevistas y talleres para construir
modelos mentales siguiendo el método propuesto por Siqueiros et al. (2019):
identificar variables/conceptos clave y las relaciones causales entre ellas.
"""

BASE_DICTIONARY: dict[str, str] = {
    "suministro de agua": "abastecimiento de agua",
    "agua potable": "abastecimiento de agua",
    "red hídrica": "abastecimiento de agua",
    "tandeo": "abastecimiento de agua",
    "pozos": "extracción de agua subterránea",
    "pozos de agua": "extracción de agua subterránea",
    "hundimiento": "subsidencia",
    "hundimientos del suelo": "subsidencia",
    "basura": "residuos sólidos",
    "desechos": "residuos sólidos",
    "inundaciones": "riesgo de inundación",
    "encharcamiento": "riesgo de inundación",
    "contaminación": "contaminación del agua",
    "agua sucia": "contaminación del agua",
    "cambio climático": "variabilidad climática",
    "lluvia": "precipitación",
    "lluvias": "precipitación",
    "densidad poblacional": "crecimiento urbano",
    "expansión urbana": "crecimiento urbano",
    "pobreza": "marginalidad socioeconómica",
    "marginación": "marginalidad socioeconómica",
    "drenaje": "infraestructura de drenaje",
    "alcantarillado": "infraestructura de drenaje",
    "mantenimiento": "mantenimiento de infraestructura",
    "inversión": "inversión en infraestructura",
    "gobierno": "capacidad institucional",
    "autoridades": "capacidad institucional",
}


# 3. Estructuras de datos
 

@dataclass
class InterviewMetadata:
    interview_id: str
    activity_type: str          # int | wks | fg
    sector: str = ""
    delegation: str = ""
    source_file: str = ""


@dataclass
class MentalModelResult:
    metadata: InterviewMetadata
    variables: list[str] = field(default_factory=list)
    edgelist: list[tuple[str, str]] = field(default_factory=list)
    summary: str = ""
    raw_response: str = ""
    error: Optional[str] = None


# 
# 4. Schemas Pydantic para Structured Outputs
#
# MEJORA 2: Al pasar estos schemas a `response_schema`, Gemini garantiza
# que la respuesta siempre tendrá exactamente estos campos y tipos.
# Esto elimina el JSONDecodeError y el limpiado de markdown fences.

class CausalEdge(BaseModel):
    """Un par causa → efecto extraído del texto."""
    cause: str = Field(..., description="Variable que actúa como causa")
    effect: str = Field(..., description="Variable que actúa como efecto")


class MentalModelSchema(BaseModel):
    """Salida estructurada del análisis de una transcripción."""
    variables: list[str] = Field(
        ...,
        description="Lista de conceptos/variables clave identificados en la transcripción",
    )
    edgelist: list[CausalEdge] = Field(
        ...,
        description="Lista de relaciones causales causa→efecto",
    )
    summary: str = Field(
        ...,
        description="Párrafo de 3-5 oraciones resumiendo la narrativa central del participante",
    )


# 5. Lectura e identificación de transcripciones

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
        "FED": "Federal",           # FED antes que DF para evitar falsos positivos
        "MC": "Magdalena Contreras",
        "DF": "Ciudad de México (gobierno)",
        "I": "Iztapalapa",
        "X": "Xochimilco",
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



# 6. Preprocesamiento de texto

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


def standardize_variable(var: str, dictionary: dict[str, str]) -> str:
    return dictionary.get(var.lower().strip(), var.strip())


# 7. Prompts

PROMPT_INDIVIDUAL = """\
{context}

A continuación se presenta la transcripción de una entrevista individual con ID {interview_id}.
El participante pertenece al sector: {sector}.

=== TRANSCRIPCIÓN ===
{transcript}
=== FIN DE TRANSCRIPCIÓN ===

Extrae el modelo mental de este participante:
- Entre 8 y 25 variables/conceptos clave (sustantivos o frases nominales cortas).
- Entre 10 y 30 pares causales (causa → efecto), explícitos o implícitos en el texto.
- Un resumen de 3-5 oraciones sobre la narrativa central del participante.
"""

PROMPT_COLECTIVO = """\
{context}

A continuación se presenta la transcripción de un {activity_label} (múltiples participantes) con ID {interview_id}.
Los participantes pertenecen al sector: {sector}.

=== TRANSCRIPCIÓN ===
{transcript}
=== FIN DE TRANSCRIPCIÓN ===

Extrae el modelo mental COLECTIVO del grupo, prestando atención a consensos y dinámicas:
- Entre 12 y 30 variables/conceptos clave discutidos por el grupo.
- Entre 15 y 40 pares causales construidos o validados durante la conversación.
- Un resumen de 3-5 oraciones sobre narrativas dominantes, acuerdos y preocupaciones principales.
"""

# 8. Exponential Backoff con Tenacity
#
# MEJORA 1: En lugar de un sleep fijo, usamos tenacity para reintentar
# automáticamente en errores 503 (servidor saturado) y 429 (rate limit),
# con espera que se duplica en cada intento (2s → 4s → 8s → 16s → 32s).
# Otros errores (ej. 400 Bad Request) no se reintentan porque no tienen solución
# reintentando.


def _is_retryable(exc: BaseException) -> bool:
    """Devuelve True sólo para errores transitorios de la API."""
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

# 9. Llamada a la API con Structured Outputs

def _call_gemini_api(
    client: genai.Client,
    model_name: str,
    prompt: str,
) -> MentalModelSchema:
    """
    Llama a la API de Gemini con response_schema de Pydantic.

    MEJORA 2: Al especificar response_schema=MentalModelSchema, el modelo
    devuelve JSON validado contra el schema. No hay markdown fences, no hay
    JSONDecodeError: si la respuesta no conforma el schema, la librería
    lanza una excepción antes de que el código la reciba.
    """
    retry_call = _build_retry_decorator()

    @retry_call
    def _do_request():
        return client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=MentalModelSchema,
            ),
        )

    response = _do_request()
    # Con response_schema, response.parsed devuelve directamente el objeto Pydantic
    parsed: MentalModelSchema = response.parsed
    if parsed is None:
        # Fallback: parseo manual si la librería no lo parsea automáticamente
        raw_text = response.text or ""
        parsed = MentalModelSchema.model_validate_json(raw_text)
    return parsed

# 10. Análisis principal de una transcripción

def analyze_transcript(
    text: str,
    metadata: InterviewMetadata,
    client: genai.Client,
    model_name: str,
    dictionary: dict[str, str],
) -> MentalModelResult:
    """Analiza una transcripción y devuelve el modelo mental extraído."""

    is_collective = metadata.activity_type in ("fg", "wks")
    activity_label = (
        "grupo focal" if metadata.activity_type == "fg"
        else "taller participativo" if metadata.activity_type == "wks"
        else "entrevista individual"
    )
    prompt_template = PROMPT_COLECTIVO if is_collective else PROMPT_INDIVIDUAL

    prompt = prompt_template.format(
        context=MEGADAPT_CONTEXT,
        activity_label=activity_label,
        interview_id=metadata.interview_id,
        sector=metadata.sector,
        transcript=text,
    )

    result = MentalModelResult(metadata=metadata)

    try:
        parsed = _call_gemini_api(client, model_name, prompt)

        # Estandarizar variables
        std_vars = [standardize_variable(v, dictionary) for v in parsed.variables]
        result.variables = list(dict.fromkeys(std_vars))  # dedup preservando orden

        # Estandarizar edgelist
        result.edgelist = [
            (
                standardize_variable(edge.cause, dictionary),
                standardize_variable(edge.effect, dictionary),
            )
            for edge in parsed.edgelist
        ]

        result.summary = parsed.summary
        log.info(
            f"✓ {metadata.interview_id} ({activity_label}): "
            f"{len(result.variables)} variables, {len(result.edgelist)} relaciones"
        )

    except RetryError as exc:
        result.error = f"API no disponible tras reintentos: {exc}"
        log.error(f"✗ {metadata.interview_id}: {result.error}")
    except Exception as exc:
        result.error = f"Error inesperado: {exc}"
        log.error(f"✗ {metadata.interview_id}: {result.error}")

    return result


# 11. Exportación de resultados

def save_edgelist(result: MentalModelResult, out_dir: Path) -> None:
    """Guarda el edgelist individual como .txt (causa,efecto por línea)."""
    if not result.edgelist:
        return
    out_path = out_dir / "edgelists" / f"{result.metadata.interview_id}_edgelist.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for cause, effect in result.edgelist:
            f.write(f"{cause},{effect}\n")
    log.info(f"  → Edgelist guardado: {out_path.name}")


def result_to_dict(r: MentalModelResult) -> dict:
    d = asdict(r.metadata)
    d["variables"] = r.variables
    d["edgelist"] = [list(e) for e in r.edgelist]
    d["summary"] = r.summary
    d["error"] = r.error
    return d


# 12. Checkpointing progresivo
#
# MEJORA 3: Estas funciones leen el estado previo del disco (si existe),
# lo actualizan con el nuevo resultado y lo sobreescriben, todo en cada
# iteración. Si el proceso muere en el archivo 34 de 35, los consolidados
# ya contienen los 33 anteriores.

def checkpoint_summary_json(result: MentalModelResult, out_dir: Path) -> None:
    """Añade o actualiza la entrada de `result` en analysis_summary.json."""
    out_path = out_dir / "analysis_summary.json"
    existing: list[dict] = []
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = []

    # Reemplazar si ya existe entrada con el mismo interview_id
    new_entry = result_to_dict(result)
    existing = [e for e in existing if e.get("interview_id") != new_entry["interview_id"]]
    existing.append(new_entry)

    out_path.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def checkpoint_variables_csv(result: MentalModelResult, out_dir: Path) -> None:
    """Añade las variables de `result` a variables_table.csv."""
    if not result.variables:
        return
    out_path = out_dir / "variables_table.csv"

    new_rows = pd.DataFrame([
        {
            "interview_id": result.metadata.interview_id,
            "sector": result.metadata.sector,
            "delegation": result.metadata.delegation,
            "activity": result.metadata.activity_type,
            "variable": var,
        }
        for var in result.variables
    ])

    if out_path.exists():
        existing = pd.read_csv(out_path, encoding="utf-8-sig")
        # Eliminar registros previos del mismo interview_id antes de añadir
        existing = existing[existing["interview_id"] != result.metadata.interview_id]
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows

    combined.to_csv(out_path, index=False, encoding="utf-8-sig")


def checkpoint_global_edgelist(results: list[MentalModelResult], out_dir: Path) -> None:
    """
    Recalcula el edgelist global a partir de todos los resultados en memoria
    y lo sobreescribe. Se llama tras cada iteración exitosa.
    """
    edge_counts: dict[tuple[str, str], int] = {}
    for r in results:
        for edge in r.edgelist:
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    rows = [
        {"cause": c, "effect": e, "frequency": n}
        for (c, e), n in sorted(edge_counts.items(), key=lambda x: -x[1])
    ]
    if not rows:
        return

    out_path = out_dir / "global_edgelist.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    log.debug(f"  → Global edgelist actualizado: {len(rows)} relaciones únicas")


def save_all_checkpoints(result: MentalModelResult, all_results: list[MentalModelResult], out_dir: Path) -> None:
    """Punto de entrada único para el guardado progresivo tras cada iteración."""
    save_edgelist(result, out_dir)
    checkpoint_summary_json(result, out_dir)
    if not result.error:
        checkpoint_variables_csv(result, out_dir)
        checkpoint_global_edgelist(all_results, out_dir)


# 13. Pipeline principal
#
# MEJORA 4: ThreadPoolExecutor con max_workers=2 procesa 2 transcripciones
# en paralelo. Es un valor conservador que respeta los rate limits de Gemini
# (por defecto 60 RPM en gemini-flash) sin sobrecargar la API. Aumentar a 3
# si el equipo tiene una cuenta de mayor cuota.
#
# IMPORTANTE: El guardado en disco se hace en el hilo principal (thread-safe)
# usando un lock, evitando condiciones de carrera al escribir los CSVs.

import threading

_checkpoint_lock = threading.Lock()


def _process_single_file(
    fpath: Path,
    client: genai.Client,
    model_name: str,
    dictionary: dict[str, str],
    out_path: Path,
    all_results: list[MentalModelResult],
) -> MentalModelResult:
    """Procesa un archivo y guarda checkpoints. Diseñado para ejecutarse en un hilo."""
    log.info(f"Procesando: {fpath.name}")

    metadata = parse_interview_id(fpath.name)
    metadata.source_file = str(fpath)

    raw_text = load_transcript(fpath)
    if not raw_text.strip():
        log.warning(f"  Archivo vacío o no legible: {fpath.name}")
        result = MentalModelResult(metadata=metadata, error="Archivo vacío o no legible")
    else:
        clean_text = preprocess(raw_text)
        result = analyze_transcript(clean_text, metadata, client, model_name, dictionary)

    # Guardado en disco bajo lock para evitar condiciones de carrera entre hilos
    with _checkpoint_lock:
        all_results.append(result)
        save_all_checkpoints(result, all_results, out_path)

    return result


def run_pipeline(
    input_folder: str,
    output_folder: str,
    api_key: str,
    model_name: str = "gemini-3.6-flash",
    limit: Optional[int] = None,
    dictionary_path: Optional[str] = None,
    max_workers: int = 2,
) -> None:
    client = genai.Client(api_key=api_key)
    log.info(f"Modelo Gemini: {model_name} | Hilos paralelos: {max_workers}")

    # Cargar diccionario
    dictionary = dict(BASE_DICTIONARY)
    if dictionary_path and Path(dictionary_path).exists():
        custom = json.loads(Path(dictionary_path).read_text(encoding="utf-8"))
        dictionary.update(custom)
        log.info(f"Diccionario cargado: {len(dictionary)} equivalencias")

    # Descubrir archivos
    in_path = Path(input_folder)
    files = sorted(f for f in in_path.rglob("*") if f.suffix.lower() in (".docx", ".txt"))
    if limit:
        files = files[:limit]
    log.info(f"Transcripciones encontradas: {len(files)}")

    out_path = Path(output_folder)
    out_path.mkdir(parents=True, exist_ok=True)

    # Lista compartida entre hilos (acceso protegido por _checkpoint_lock)
    all_results: list[MentalModelResult] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_single_file,
                fpath, client, model_name, dictionary, out_path, all_results,
            ): fpath
            for fpath in files
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Analizando transcripciones"):
            fpath = futures[future]
            try:
                future.result()
            except Exception as exc:
                log.error(f"Error fatal procesando {fpath.name}: {exc}")

    # Reporte final
    errors = [r for r in all_results if r.error]
    if errors:
        log.warning(f"\n{len(errors)} transcripciones con error:")
        for r in errors:
            log.warning(f"  • {r.metadata.interview_id}: {r.error}")

    log.info(
        f"\n✓ Pipeline completado. "
        f"{len(all_results) - len(errors)}/{len(all_results)} transcripciones procesadas correctamente.\n"
        f"  Salida en: {out_path.resolve()}"
    )


# 14. Punto de entrada

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="MEGADAPT – Análisis cualitativo automatizado con Gemini (v2)"
    )
    parser.add_argument("--folder", "-f", required=True,
        help="Carpeta con transcripciones (.docx o .txt).")
    parser.add_argument("--out", "-o", default="./output_megadapt",
        help="Carpeta de salida (default: ./output_megadapt)")
    parser.add_argument("--api-key", "-k",
        default=os.environ.get("GEMINI_API_KEY", ""),
        help="API key de Gemini (o variable GEMINI_API_KEY)")
    parser.add_argument("--model",
        default="gemini-3.6-flash",
        help="Modelo Gemini a usar (default: gemini-3.6-flash)")
    parser.add_argument("--limit", type=int, default=None,
        help="Nº máximo de transcripciones a procesar (para pruebas)")
    parser.add_argument("--dictionary", default=None,
        help="Ruta a JSON con equivalencias adicionales {término: estándar}")
    parser.add_argument("--workers", type=int, default=2,
        help="Nº de hilos paralelos (default: 2; recomendado: 2-3)")

    args = parser.parse_args()

    if not args.api_key:
        parser.error("Proporciona tu API key con --api-key o exporta GEMINI_API_KEY")

    run_pipeline(
        input_folder=args.folder,
        output_folder=args.out,
        api_key=args.api_key,
        model_name=args.model,
        limit=args.limit,
        dictionary_path=args.dictionary,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
