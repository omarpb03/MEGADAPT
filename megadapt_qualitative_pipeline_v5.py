"""
MEGADAPT - Pipeline de análisis cualitativo automatizado, v5 (OpenAI)

Extrae cuatro campos cualitativos de cada transcripción de entrevista y
guarda el consolidado directamente como `analysis_v5.csv`.

La fase de estandarización fue eliminada: el CSV resultante contiene las
frases exactamente como las devuelve el LLM (output "crudo"), lo que
permite evaluar la extracción con BERTScore sin introducir sesgos derivados
de alterar el texto antes de la comparación.

Cambios respecto a v4 (Gemini):
  1. Proveedor OpenAI (GPT-4o y variantes) con salida estructurada (Pydantic).
  2. Prompt con definiciones explícitas de Preocupación, Causa, Consecuencia,
     Acción y Actor, con ejemplos tomados de la codificación humana.
  3. Cada elemento se devuelve resumido en 4-5 palabras (como lo hace el equipo humano).
  4. Todo en español (si la entrevista está en inglés, el modelo traduce).
  5. Sin truncado a 30,000 caracteres: se manda la transcripción completa y se
     verifica contra el context window del modelo.
  6. Modo --count-only: palabras, tokens, % del context window y costo estimado
     por entrevista y por modelo, SIN llamar a la API.
  7. Reanudación (--resume): no vuelve a pagar tokens por entrevistas ya procesadas.
  8. Registro de tokens reales usados y costo estimado.

Uso:
  python megadapt_qualitative_pipeline_v5.py --folder ./transcripciones --count-only
  python megadapt_qualitative_pipeline_v5.py --folder ./transcripciones --model gpt-4o-mini
  python megadapt_qualitative_pipeline_v5.py --folder ./transcripciones --model claude-haiku-3-5

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 0. Catálogo de modelos (context window y precios de REFERENCIA)
# ──────────────────────────────────────────────────────────────────────────────
# USD por 1M de tokens. Son valores de referencia: CONFIRMAR en la página de
# precios de OpenAI antes de una corrida grande. Editar aquí si cambian.
@dataclass(frozen=True)
class ModelInfo:
    context: int
    price_in: float
    price_out: float

MODEL_CATALOG: dict[str, ModelInfo] = {
    "gpt-4o":       ModelInfo(128_000,   2.50, 10.00),
    "gpt-4o-mini":  ModelInfo(128_000,   0.15,  0.60),
    "gpt-4.1":      ModelInfo(1_047_576, 2.00,  8.00),
    "gpt-4.1-mini": ModelInfo(1_047_576, 0.40,  1.60),
    "gpt-4.1-nano": ModelInfo(1_047_576, 0.10,  0.40),
}

# Solo estos prefijos aceptan el parámetro temperature (los modelos de
# razonamiento lo rechazan).
TEMPERATURE_PREFIXES = ("gpt-4o", "gpt-4.1")

PROMPT_OVERHEAD_TOKENS = 350   # esquema JSON + mensajes del sistema de la API (aprox.)
EST_OUTPUT_TOKENS = 1_000      # salida típica por entrevista (aprox.)
CONTEXT_SAFETY = 0.90          # se marca "excede" si la entrada usa >90% del contexto

# ──────────────────────────────────────────────────────────────────────────────
# 1. Estructuras de datos y schemas
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class InterviewMetadata:
    Id_entrevista: str

@dataclass
class InterviewResult:
    metadata: InterviewMetadata
    Preocupacion_principal_acerca_del_agua: list[str] = field(default_factory=list)
    Causas_principales: list[str] = field(default_factory=list)
    Consecuencias: list[str] = field(default_factory=list)
    Acciones_y_Actores_que_realizan_dichas_acciones: list[dict] = field(default_factory=list)  # {actor, accion}
    input_tokens: int = 0
    output_tokens: int = 0
    error: Optional[str] = None


class AccionActor(BaseModel):
    actor: str = Field(description="Quién realiza la acción (institución, grupo o persona), en 1 a 3 palabras.")
    accion: str = Field(description="La acción, resumida en 4 o 5 palabras.")


class QualitativeAnalysisSchema(BaseModel):
    Preocupacion_principal_acerca_del_agua: list[str] = Field(
        description="De 1 a 4 preocupaciones principales sobre el agua, cada una de 4 o 5 palabras.")
    Causas_principales: list[str] = Field(
        description="Causas del problema mencionadas, cada una de 4 o 5 palabras.")
    Consecuencias: list[str] = Field(
        description="Consecuencias o efectos mencionados, cada una de 4 o 5 palabras.")
    Acciones_y_Actores_que_realizan_dichas_acciones: list[AccionActor] = Field(
        description="Acciones realizadas o propuestas, con el actor que las realiza.")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Lectura y preprocesamiento
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
    """Limpieza ligera. No trunca: la transcripción completa se envía al modelo."""
    text = re.sub(r"\[.*?\]", "", text)   # notas del transcriptor, marcas de tiempo
    text = re.sub(r"\(.*?\)", "", text)   # acotaciones tipo (risas)
    # Etiquetas de hablante solo al inicio de línea
    text = re.sub(r"(?m)^\s*E:\s*", "Entrevistador: ", text)
    text = re.sub(r"(?m)^\s*R:\s*", "Respondente: ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

# ──────────────────────────────────────────────────────────────────────────────
# 3. Conteo de palabras y tokens
# ──────────────────────────────────────────────────────────────────────────────
_ENCODER = None          # None = sin intentar, False = no disponible
TOKEN_MODE = "sin inicializar"


def _get_encoder():
    global _ENCODER, TOKEN_MODE
    if _ENCODER is None:
        try:
            import tiktoken
            _ENCODER = tiktoken.get_encoding("o200k_base")   # GPT-4o, 4.1 y posteriores
            TOKEN_MODE = "exacto (tiktoken o200k_base)"
        except Exception as exc:
            log.warning(f"tiktoken no disponible ({type(exc).__name__}); se usa aproximación 1.5 tokens/palabra.")
            _ENCODER = False
            TOKEN_MODE = "APROXIMADO (1.5 tokens/palabra)"
    return _ENCODER


def count_tokens(text: str) -> int:
    enc = _get_encoder()
    if enc:
        return len(enc.encode(text, disallowed_special=()))
    return int(len(text.split()) * 1.5)


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


def count_only(folder: str, out_dir: str, models: list[str]) -> pd.DataFrame:
    """Palabras, tokens, % de context window y costo estimado por entrevista y modelo."""
    files = list_transcripts(folder)
    if not files:
        raise SystemExit(f"No se encontraron transcripciones en {folder}")
    prompt_tokens = count_tokens(SYSTEM_PROMPT) + PROMPT_OVERHEAD_TOKENS
    rows = []
    for f in tqdm(files, desc="Contando tokens"):
        text = preprocess(load_transcript(f))
        toks = count_tokens(text)
        row = {
            "Id_entrevista": parse_interview_id(f.name).Id_entrevista,
            "archivo": f.name,
            "palabras": len(text.split()),
            "caracteres": len(text),
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
            "entrevistas_que_exceden_90pct": exceeds,
            "costo_total_usd_estimado": round(df[f"costo_usd_{m}"].sum(), 2),
        })
    sdf = pd.DataFrame(summary)
    sdf.to_csv(out / "model_cost_summary.csv", index=False, encoding="utf-8-sig")

    print(f"\nModo de conteo de tokens: {TOKEN_MODE}")
    print(f"Entrevistas: {len(df)} | palabras totales: {df['palabras'].sum():,} | "
          f"tokens de entrada totales: {df['tokens_entrada_total'].sum():,}")
    print(f"Por entrevista -> palabras: mediana {int(df['palabras'].median()):,}, máx {df['palabras'].max():,} | "
          f"tokens: mediana {int(df['tokens_entrada_total'].median()):,}, máx {df['tokens_entrada_total'].max():,}")
    print(f"(costo asume {EST_OUTPUT_TOKENS} tokens de salida por entrevista)")
    print(sdf.to_string(index=False))
    print("\nPrecios de referencia: verificar en la página de precios de OpenAI antes de correr.")
    return df

# ──────────────────────────────────────────────────────────────────────────────
# 4. Prompt de extracción
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
Eres un analista socio-ambiental cualitativo, especializado en gestión del agua y riesgo hídrico en zonas urbanas de la Ciudad de México (inundaciones, escasez, calidad del agua, distribución).

Recibirás la transcripción completa de una entrevista con actores del sistema del agua (funcionarios, académicos, organizaciones civiles, vecinos). Los hablantes pueden venir como Entrevistador/Respondente, E1/E2, etc.; puede haber más de dos participantes. Analiza SOLO lo que dicen los participantes entrevistados, no las preguntas del entrevistador.

Tu tarea es codificar la entrevista en cuatro campos. Definiciones:

1. PREOCUPACIÓN PRINCIPAL ACERCA DEL AGUA: el problema del agua que el participante considera más importante o que motiva su trabajo o participación (por ejemplo escasez, mala calidad, inundaciones, tandeo o distribución desigual, hundimientos y grietas). Entrega de 1 a 4 preocupaciones, ordenadas de la más importante a la menos importante.

2. CAUSAS PRINCIPALES: los factores que el participante señala como ORIGEN del problema, ya sean biofísicos (lluvias intensas, sobreexplotación del acuífero), socio-institucionales (corrupción, falta de coordinación, falta de planeación), de uso de suelo (asentamientos irregulares, urbanización) o de infraestructura (fugas, drenaje insuficiente, tuberías viejas).

3. CONSECUENCIAS: los efectos o impactos que el participante dice que resultan del problema, sobre las personas, la salud, la economía, la vivienda, el ambiente o la organización social (por ejemplo daños a la salud, conflictos entre vecinos, costos por comprar agua, grietas en viviendas).

4. ACCIONES Y ACTORES: lo que se hace, o se propone hacer, para enfrentar el problema, indicando QUIÉN lo hace. Un ACTOR es una persona, institución u organización mencionada en la entrevista (por ejemplo SACMEX, Protección Civil, la delegación, CONAGUA, vecinos, una organización civil, ejidatarios, el entrevistado mismo). Si la entrevista no dice quién realiza la acción, usa "No especificado". No inventes actores.

FORMATO (muy importante):
- Cada elemento debe ser una frase corta de 4 o 5 palabras (máximo 6), en forma de etiqueta, no una oración completa ni un párrafo. Resume la idea con las palabras del entrevistado siempre que puedas.
- Ejemplos del estilo esperado: "Escasez de agua", "Tandeo en las colonias", "Inundaciones en la parte baja", "Fugas en la red", "Asentamientos irregulares", "Falta de mantenimiento en coladeras", "Daños a la salud", "Compra de agua en pipas", "Campañas de limpieza comunitaria", "Reforestación de las barrancas".
- Ejemplos de acciones con actor: {"actor": "Protección Civil", "accion": "Emite recomendaciones en zonas de riesgo"}, {"actor": "Vecinos", "accion": "Almacenan agua en cisternas"}.
- No repitas la misma idea dos veces dentro de un mismo campo.
- No agregues información que no esté en la transcripción ni interpretes más allá de lo que se dijo. Si un campo no se menciona, devuelve una lista vacía.
- Todo debe estar en español. Si la entrevista está en otro idioma, tradúcela al español.
- Devuelve únicamente el JSON con la estructura solicitada.
""".strip()

USER_TEMPLATE = "=== TRANSCRIPCIÓN ===\n{transcript}\n=== FIN DE TRANSCRIPCIÓN ==="

# ──────────────────────────────────────────────────────────────────────────────
# 5. Cliente OpenAI con reintentos
# ──────────────────────────────────────────────────────────────────────────────
class UsageTracker:
    """Acumula tokens reales y costo estimado (para vigilar el presupuesto)."""
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
    @retry(retry=retry_if_exception(_is_retryable),
           wait=wait_exponential(multiplier=1, min=2, max=60),
           stop=stop_after_attempt(6),
           before_sleep=before_sleep_log(log, logging.WARNING),
           reraise=True)
    def _do():
        kwargs = dict(model=model,
                      messages=[{"role": "system", "content": system},
                                {"role": "user", "content": user}],
                      response_format=schema)
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


def clean_phrase(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s)).strip(" \t\n.;:,-•*")
    return s[:1].upper() + s[1:] if s else s


def dedupe(items: list[str]) -> list[str]:
    seen, out = set(), []
    for x in items:
        k = x.lower()
        if x and k not in seen:
            seen.add(k)
            out.append(x)
    return out

# ──────────────────────────────────────────────────────────────────────────────
# 6. Extracción por entrevista
# ──────────────────────────────────────────────────────────────────────────────
def analyze_transcript(text: str, metadata: InterviewMetadata, client, model: str,
                       tracker: UsageTracker, temperature: Optional[float] = 0.2) -> InterviewResult:
    result = InterviewResult(metadata=metadata)
    info = MODEL_CATALOG.get(model)
    est_in = count_tokens(text) + count_tokens(SYSTEM_PROMPT) + PROMPT_OVERHEAD_TOKENS
    if info and est_in > CONTEXT_SAFETY * info.context:
        result.error = (f"La entrada (~{est_in:,} tokens) excede el {int(CONTEXT_SAFETY*100)}% del context "
                        f"window de {model} ({info.context:,}). Usar un modelo con más contexto.")
        log.error(f"  {metadata.Id_entrevista}: {result.error}")
        return result
    try:
        parsed, usage = _call_structured(client, model, SYSTEM_PROMPT,
                                         USER_TEMPLATE.format(transcript=text),
                                         QualitativeAnalysisSchema, temperature)
        result.input_tokens, result.output_tokens = tracker.add(usage)
        result.Preocupacion_principal_acerca_del_agua = dedupe([clean_phrase(x) for x in parsed.Preocupacion_principal_acerca_del_agua])
        result.Causas_principales = dedupe([clean_phrase(x) for x in parsed.Causas_principales])
        result.Consecuencias = dedupe([clean_phrase(x) for x in parsed.Consecuencias])
        result.Acciones_y_Actores_que_realizan_dichas_acciones = [
            {"actor": clean_phrase(a.actor) or "No especificado", "accion": clean_phrase(a.accion)}
            for a in parsed.Acciones_y_Actores_que_realizan_dichas_acciones if clean_phrase(a.accion)
        ]
    except Exception as exc:
        log.error(f"  Error analizando {metadata.Id_entrevista}: {exc}")
        result.error = str(exc)
    return result

# ──────────────────────────────────────────────────────────────────────────────
# 7. Salidas: CSV consolidado
# ──────────────────────────────────────────────────────────────────────────────
COL_PREOC  = "Preocupacion_principal_acerca_del_agua"
COL_CAUSAS = "Causas_principales"
COL_CONSEC = "Consecuencias"
COL_ACC    = "Acciones_y_Actores_que_realizan_dichas_acciones"


def _fmt_action(a: dict) -> str:
    return f"{a['actor']}: {a['accion']}"


def results_to_dataframe(results: list[InterviewResult]) -> pd.DataFrame:
    rows = []
    for r in sorted((r for r in results if not r.error), key=lambda r: r.metadata.Id_entrevista):
        rows.append({
            "Id_entrevista": r.metadata.Id_entrevista,
            COL_PREOC:  "\n".join(r.Preocupacion_principal_acerca_del_agua),
            COL_CAUSAS: "\n".join(r.Causas_principales),
            COL_CONSEC: "\n".join(r.Consecuencias),
            COL_ACC:    "\n".join(_fmt_action(a) for a in r.Acciones_y_Actores_que_realizan_dichas_acciones),
        })
    return pd.DataFrame(rows)


_checkpoint_lock = threading.Lock()


def save_checkpoints(result: InterviewResult, all_results: list[InterviewResult], out_dir: Path):
    """Guarda el JSON individual y regenera el CSV consolidado con los resultados disponibles hasta el momento."""
    d = out_dir / "individual"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{result.metadata.Id_entrevista}.json").write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    df = results_to_dataframe(all_results)
    if not df.empty:
        df.to_csv(out_dir / "analysis_v5.csv", index=False, encoding="utf-8-sig")


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
# 8. Pipeline principal
# ──────────────────────────────────────────────────────────────────────────────
def _process_single_file(fpath: Path, client, model: str, out_path: Path,
                         all_results: list[InterviewResult], tracker: UsageTracker, resume: bool):
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
        result = InterviewResult(metadata=metadata, error="Archivo vacío")
    else:
        result = analyze_transcript(preprocess(raw_text), metadata, client, model, tracker)
    with _checkpoint_lock:
        all_results.append(result)
        save_checkpoints(result, all_results, out_path)


def make_client():
    from openai import OpenAI
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("Falta OPENAI_API_KEY en el archivo .env")
    return OpenAI(api_key=key)


def run_pipeline(input_folder: str, output_folder: str, model: str,
                 max_workers: int = 2, resume: bool = True, client=None):
    """
    Ejecuta la extracción sobre todas las transcripciones en `input_folder`
    y guarda el CSV consolidado en `output_folder/analysis_v5.csv`.

    Parámetros
    ----------
    input_folder  : carpeta con archivos .docx / .txt / .md
    output_folder : directorio de salida
    model         : modelo OpenAI para la extracción
    max_workers   : hilos concurrentes (ajustar según rate limit de la API)
    resume        : si True, omite entrevistas cuyo JSON individual ya existe
    client        : instancia OpenAI opcional (se crea desde .env si no se pasa)
    """
    client = client or make_client()
    out_path = Path(output_folder)
    out_path.mkdir(parents=True, exist_ok=True)
    tracker = UsageTracker(model)
    log.info(f"Modelo de extracción: {model}")

    files = list_transcripts(input_folder)
    if not files:
        raise SystemExit(f"No se encontraron transcripciones en {input_folder}")
    log.info(f"Transcripciones encontradas: {len(files)}")

    all_results: list[InterviewResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_single_file, f, client, model, out_path,
                            all_results, tracker, resume): f
            for f in files
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Analizando"):
            try:
                fut.result()
            except Exception as exc:
                log.error(f"Error fatal en {futures[fut].name}: {exc}")

    failed = [r.metadata.Id_entrevista for r in all_results if r.error]
    if failed:
        log.warning(f"{len(failed)} entrevistas con error: {failed}")

    # CSV consolidado final (también se va escribiendo como checkpoint en save_checkpoints)
    df = results_to_dataframe(all_results)
    if not df.empty:
        out_csv = out_path / "analysis_v5.csv"
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        log.info(f"CSV consolidado guardado en: {out_csv.resolve()}")
        log.info(f"Entrevistas en el CSV: {len(df)}")
    else:
        log.warning("No se generaron resultados válidos; el CSV no fue creado.")

    # Registro de uso acumulativo (una línea por corrida)
    usage = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "extraccion": {
            "modelo": model,
            "tokens_entrada": tracker.input,
            "tokens_salida": tracker.output,
            "costo_usd_estimado": tracker.cost(),
        },
    }
    with open(out_path / "usage_log_v5.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(usage, ensure_ascii=False) + "\n")
    log.info("Uso de tokens y costo estimado de esta corrida:\n"
             + json.dumps(usage, ensure_ascii=False, indent=2))
    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline MEGADAPT v5 — extracción cualitativa de entrevistas (OpenAI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Estimar costo y tokens antes de lanzar:
  python megadapt_qualitative_pipeline_v5.py \\
      --folder ./transcripciones --count-only

  # Correr extracción completa con gpt-4o-mini:
  python megadapt_qualitative_pipeline_v5.py \\
      --folder ./transcripciones --model gpt-4o-mini --out ./output_v5

  # Re-procesar todo (ignorar checkpoints):
  python megadapt_qualitative_pipeline_v5.py \\
      --folder ./transcripciones --model gpt-4o-mini --no-resume

  # Salida: output_v5/analysis_v5.csv
""",
    )
    parser.add_argument("--folder", help="Carpeta con transcripciones (.docx/.txt/.md)")
    parser.add_argument("--out", default="./output_v5", help="Directorio de salida (default: ./output_v5)")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help=f"Modelo de extracción. Opciones: {', '.join(MODEL_CATALOG)}")
    parser.add_argument("--workers", type=int, default=2,
                        help="Hilos concurrentes (ajustar según rate limit, default: 2)")
    parser.add_argument("--count-only", action="store_true",
                        help="Solo cuenta palabras/tokens/costo estimado; no llama a la API")
    parser.add_argument("--models", default=",".join(MODEL_CATALOG),
                        help="Modelos a comparar en --count-only (separados por coma)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Reprocesar todas las entrevistas aunque ya exista su JSON individual")
    args = parser.parse_args()

    if args.count_only:
        if not args.folder:
            parser.error("--count-only requiere --folder")
        unknown = [m for m in args.models.split(",") if m not in MODEL_CATALOG]
        if unknown:
            parser.error(f"Modelos fuera del catálogo (agrégalos a MODEL_CATALOG): {unknown}")
        count_only(args.folder, args.out, args.models.split(","))
    else:
        if not args.folder:
            parser.error("Falta --folder con la ruta a las transcripciones")
        run_pipeline(
            args.folder, args.out, args.model,
            max_workers=args.workers,
            resume=not args.no_resume,
        )
