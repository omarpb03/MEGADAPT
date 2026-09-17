"""
Pipeline de análisis cualitativo automatizado (Edición NATIVA Google Gemini)
A prueba de fallos. Único proveedor: Google Gemini. Columnas Simplificadas.
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
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential, before_sleep_log
from tqdm import tqdm

load_dotenv()

# Solución para bug de Windows con la librería de Google
if "GOOGLE_API_KEY" in os.environ:
    os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]
    del os.environ["GOOGLE_API_KEY"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# 1. Estructuras de datos & Schemas Simplificados
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class InterviewMetadata:
    Id_entrevista: str

@dataclass
class InterviewResult:
    metadata: InterviewMetadata
    Preocupacion_principal_acerca_del_agua: str = ""
    Causas_principales: list[str] = field(default_factory=list)
    Consecuencias: list[str] = field(default_factory=list)
    Acciones_y_Actores_que_realizan_dichas_acciones: list[str] = field(default_factory=list)
    error: Optional[str] = None

class QualitativeAnalysisSchema(BaseModel):
    Preocupacion_principal_acerca_del_agua: str = Field(...)
    Causas_principales: list[str] = Field(..., min_length=1)
    Consecuencias: list[str] = Field(..., min_length=1)
    Acciones_y_Actores_que_realizan_dichas_acciones: list[str] = Field(..., min_length=1)

# ══════════════════════════════════════════════════════════════════════════════
# Funciones Auxiliares
# ══════════════════════════════════════════════════════════════════════════════
def load_transcript(path: Path) -> str:
    from docx import Document
    if path.suffix.lower() == ".docx":
        return "\n".join(p.text.strip() for p in Document(str(path)).paragraphs if p.text.strip())
    elif path.suffix.lower() in (".txt", ".md"):
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
            
    # Únicamente devolvemos el Id_entrevista
    return InterviewMetadata(Id_entrevista=interview_id)

def preprocess(text: str, max_chars: int = 30_000) -> str:
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"E:\s*", "Entrevistador: ", text)
    text = re.sub(r"R:\s*", "Respondente: ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[...TEXTO TRUNCADO...]"
    return text

def _is_retryable(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(kw in msg for kw in ("503", "unavailable", "429", "rate", "quota", "resource_exhausted"))

def _build_retry_decorator():
    return retry(retry=retry_if_exception(_is_retryable), wait=wait_exponential(multiplier=1, min=2, max=60), stop=stop_after_attempt(6), before_sleep=before_sleep_log(log, logging.WARNING), reraise=True)

# ══════════════════════════════════════════════════════════════════════════════
# Petición DIRECTA al LLM 
# ══════════════════════════════════════════════════════════════════════════════
CONTEXT = """
Eres un analista socio-ambiental cualitativo especializado en gestión del agua y riesgo hidrico en áreas urbanas.
Te voy a dar transcripciones de entrevistas con la siguiente estructura:Entrevistador 1 -> E1
Entrevistado 2 -> E2 (o una estructura similar). Igual pueden haber mas de dos participantes.
Tu tarea es que identifiques preocupaciones, causas, consecuencias y acciones realizadas por algún actor mencionado en la entrevista. 
Devuelve los datos en un JSON con las siguientes claves:
1. PREOCUPACIONES PRINCIPALES ACERCA DEL AGUA
2. CAUSAS PRINCIPALES
3. CONSECUENCIAS
4. ACCIONES Y ACTORES QUE REALIZAN DICHAS ACCIONES
Utiliza el lenguaje de la entrevista.
Todo el listado debe estar en español, sin traducciones.
No hagas resúmenes, ni interpretaciones, ni agregues información que no esté en la transcripción. 
No hagas parrafos, solo listas de ideas separadas por saltos de línea (Nota:una idea puede tener varias palabras, pero no es una frase ni un parrafo).
Todo dalo en español, devuelve la información en un JSON con las claves exactas que te di, sin agregar ni quitar ninguna clave.
"""

PROMPT_TEMPLATE = """
{context}

=== TRANSCRIPCIÓN ===
{transcript}
=== FIN DE TRANSCRIPCIÓN ===
"""

def _call_api(client: genai.Client, model_name: str, prompt: str) -> QualitativeAnalysisSchema:
    retry_call = _build_retry_decorator()
    @retry_call
    def _do_request():
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=QualitativeAnalysisSchema,
                temperature=0.2,
            ),
        )
        parsed = response.parsed
        if parsed is None:
            parsed = QualitativeAnalysisSchema.model_validate_json(response.text or "")
        return parsed
    return _do_request()

def analyze_transcript(text: str, metadata: InterviewMetadata, client: genai.Client, model_name: str) -> InterviewResult:
    # Generamos el prompt ya sin las variables extra que eliminamos
    prompt = PROMPT_TEMPLATE.format(context=CONTEXT, transcript=text)
    result = InterviewResult(metadata=metadata)
    try:
        parsed = _call_api(client, model_name, prompt)
        result.Preocupacion_principal_acerca_del_agua = parsed.Preocupacion_principal_acerca_del_agua
        result.Causas_principales = parsed.Causas_principales
        result.Consecuencias = parsed.Consecuencias
        result.Acciones_y_Actores_que_realizan_dichas_acciones = parsed.Acciones_y_Actores_que_realizan_dichas_acciones
    except Exception as exc:
        log.error(f"  Error analizando {metadata.Id_entrevista}: {exc}")
        result.error = str(exc)
    return result

# ══════════════════════════════════════════════════════════════════════════════
# Pipeline Principal
# ══════════════════════════════════════════════════════════════════════════════
_checkpoint_lock = threading.Lock()

def save_checkpoints(result: InterviewResult, all_results: list[InterviewResult], out_dir: Path):
    d = out_dir / "individual"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{result.metadata.Id_entrevista}.json").write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    
    rows = []
    for r in all_results:
        if not r.error:
            rows.append({
                "Id_entrevista": r.metadata.Id_entrevista,
                "Preocupacion_principal_acerca_del_agua": r.Preocupacion_principal_acerca_del_agua,
                "Causas_principales": "\n".join(r.Causas_principales),
                "Consecuencias": "\n".join(r.Consecuencias),
                "Acciones_y_Actores_que_realizan_dichas_acciones": "\n".join(r.Acciones_y_Actores_que_realizan_dichas_acciones),
            })
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "analysis_summary_v4.csv", index=False, encoding="utf-8-sig")

def _process_single_file(fpath: Path, client: genai.Client, model_name: str, out_path: Path, all_results: list[InterviewResult]):
    log.info(f"Procesando: {fpath.name}")
    metadata = parse_interview_id(fpath.name)
    raw_text = load_transcript(fpath)
    
    if not raw_text.strip():
        result = InterviewResult(metadata=metadata, error="Archivo vacío")
    else:
        result = analyze_transcript(preprocess(raw_text), metadata, client, model_name)

    with _checkpoint_lock:
        all_results.append(result)
        save_checkpoints(result, all_results, out_path)

def run_pipeline(input_folder: str, output_folder: str, model_name: str, max_workers: int = 2):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key: raise ValueError("Falta la llave GEMINI_API_KEY en el archivo .env")
    
    client = genai.Client(api_key=api_key)
    log.info(f"Conectado a GOOGLE GEMINI. Modelo: {model_name}")

    files = sorted(
        f for f in Path(input_folder).rglob("*")
        if not f.name.startswith("~$") and f.suffix.lower() in (".docx", ".txt", ".md")
    )
    out_path = Path(output_folder)
    out_path.mkdir(parents=True, exist_ok=True)
    all_results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_single_file, f, client, model_name, out_path, all_results): f for f in files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Analizando"):
            try: future.result()
            except Exception as exc: log.error(f"Error fatal: {exc}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--out", default="./output_completo")
    parser.add_argument("--model", default="gemini-3.8-flash") # Modelo actualizado por defecto
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    
    run_pipeline(input_folder=args.folder, output_folder=args.out, model_name=args.model, max_workers=args.workers)