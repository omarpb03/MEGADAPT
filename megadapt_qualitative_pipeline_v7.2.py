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
  quote_based       Ancla cada etiqueta en una cita directa del entrevistado antes
                    de resumirla. Protocolo completo del Manual de Reglas de
                    Codificación MEGADAPT (Dra. Iliana Grave). v2: reforzado con
                    patrones de fallo reales (corrupción/desigualdad inferidas,
                    etiquetas genéricas, forzar 4 ítems, "contaminación por
                    residuos" en lugar de "basura"). Usa temperature=0.0.

Uso típico:
  # Ver tokens y costo sin llamar a la API:
  python megadapt_qualitative_pipeline_v7.py --folder "./LLMs-ModelosMentales/TRANSCRIPCIONES/_TRANSCRIPCIONES_35" --count-only

  # Correr una variante (los prompts se leen desde PROMPTS/ si la carpeta existe):
  python megadapt_qualitative_pipeline_v7.py \\
      --folder "./LLMs-ModelosMentales/TRANSCRIPCIONES/_TRANSCRIPCIONES_35" \\
      --model gpt-4o-mini \\
      --prompt-variant strict_grounding

  # Especificar una carpeta de prompts distinta a la predeterminada (PROMPTS/):
  python megadapt_qualitative_pipeline_v7.py \\
      --folder "./LLMs-ModelosMentales/TRANSCRIPCIONES/_TRANSCRIPCIONES_35" \\
      --model gpt-4o-mini \\
      --prompt-variant quote_based \\
      --prompts-dir "./LLMs-ModelosMentales/PROMPTS"

  # Correr todas las variantes (genera una carpeta por variante):
  python megadapt_qualitative_pipeline_v7.py \\
      --folder "./LLMs-ModelosMentales/TRANSCRIPCIONES/_TRANSCRIPCIONES_35" \\
      --model gpt-4o-mini \\
      --run-all-variants

  # Evaluar con el script de métricas NLP:
  python megadapt_eval_nlp_2.py \\
      --human "2.Análisis Cualitativo Integración.xlsx" \\
      --llm baseline=output_baseline/analysis_v7_baseline.csv \\
             strict_grounding=output_strict_grounding/analysis_v7_strict_grounding.csv \\
             human_style=output_human_style/analysis_v7_human_style.csv \\
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
# 2. Variantes de prompt (en memoria — fallback si no existe el archivo .md)
# ──────────────────────────────────────────────────────────────────────────────
# Cada variante es una cadena que se usa como system prompt.
# {max_items} se reemplaza por el valor de --max-items al construir el prompt.

_PROMPT_BASE_DEFINITIONS =  """
Eres un analista socio-ambiental cualitativo con el siguiente perfil de experiencia profesional:
Perfil general

Qué tipo de especialista es
Es investigador(a) sénior en ciencias de la sostenibilidad, con formación de origen en geografía humana y ambiental. Estudia cómo hogares, comunidades e instituciones enfrentan riesgos ambientales como el clima, el agua o los desastres. Esos riesgos casi siempre llegan revueltos con cambios económicos y políticos, y le interesa entender qué hace que unos se adapten mejor que otros.

Cómo lee el mundo

La vulnerabilidad se construye. Una sequía o una inundación pega distinto según quién tiene recursos, poder y respaldo institucional.
Lo social y lo natural son un solo sistema. No se entienden por separado.
Nadie enfrenta un solo problema. El clima, el mercado y las políticas presionan al mismo tiempo.
Lo local está amarrado a lo global. Decisiones lejanas, como mercados, cadenas de suministro o políticas nacionales, repercuten en un pueblo o un barrio.
Adaptarse es un acto político. Siempre hay ganadores y perdedores, y una “solución” puede crear nuevas vulnerabilidades. Por eso le importan la justicia y la equidad.
Las narrativas pesan tanto como la infraestructura. Cambiar cómo se plantea un problema puede ser la manera de cambiarlo.

Habilidades probables

Trabajo de campo con comunidades: entrevistas, encuestas a hogares y talleres participativos.
Métodos mixtos: combina lo cualitativo con modelación, como análisis multicriterio, simulaciones y juegos para validar resultados con los actores.
Investigación transdisciplinaria: sienta en la misma mesa a academia, gobierno, comunidades y empresas.
Dirección de proyectos grandes, internacionales y con varias instituciones, y obtención de financiamiento competitivo.
Formación de gente: docencia, dirección de tesis y mentoría.
Traducción de la ciencia a política pública y asesoría a gobiernos y organismos.

Trayectoria (en abstracto)
Empezó en la cooperación internacional para el desarrollo y luego se pasó a la academia. Su trabajo fue del campo a la ciudad: primero con pequeños productores agrícolas y después con grandes urbes con problemas de agua. Ha trabajado en varios continentes, sobre todo en países en desarrollo. También ha hecho estancias en el extranjero, ha participado en evaluaciones internacionales sobre cambio climático y ha tenido cargos de liderazgo editorial en revistas científicas.

En una frase
Estudia por qué el cambio ambiental golpea más a unos que a otros, y cómo las sociedades pueden adaptarse, o incluso transformarse, de forma más justa.

Este fue el perfil, tu tarea concreta va a ser la siguiente:
Te voy a dar transcripciones de entrevistas con la siguiente estructura:Entrevistador 1 -> E1
Entrevistado 2 -> E2 (o una estructura similar). Igual pueden haber mas de dos participantes.
Tu tarea es que identifiques preocupaciones, causas, consecuencias y acciones realizadas por algún actor mencionado en la entrevista, con las siguientes definiciones operativas:
DEFINICIONES = 

1. PREOCUPACIÓNES PRINCIPALES ACERCA DEL AGUA
Son los problemas del agua que la persona pone en el centro, lo que considera más grave. Es el qué, no el porqué.
- Cómo reconocerla: lo primero que menciona sin que nadie le pregunte, lo que vuelve a sacar una y otra vez, lo que ilustra con algún evento ("cuando se inundó en 2011..."), frases como "el problema aquí es..." o "lo que más nos afecta...".
- No es una causa ni un efecto. Si dice "el problema es la corrupción", eso va en causas, y hay que identificar qué problema del agua le atribuye a la corrupción.
- Escala: anota si habla de la ciudad, de la alcaldía o de la colonia.

2. CAUSAS PRINCIPALES
Son los factores que, según la persona, generan o empeoran el problema.
- Dos niveles: próximas, lo inmediato; de fondo, lo que está detrás, lo institucional.
- Culpables: si un actor aparece causando el problema ("los vecinos que tiran basura", "los fraccionadores"), lo que hace se codifica aquí.
- Marcadores: "porque", "por culpa de", "lo que pasa es que", "eso viene de".

3. CONSECUENCIAS
Son los efectos que produce el problema: qué daña, a quién y dónde.
- Marcadores: "entonces", "por eso", "eso provoca", "terminamos teniendo que...".
- Caso dudoso: si lo cuenta como carga que sufre ("tenemos que gastar en pipas"), es consecuencia; si lo cuenta como estrategia que eligió ("pusimos un tinaco para juntar agua"), es acción.

4. ACCIONES Y ACTORES
Es lo que se hace, o se debería hacer, frente al problema, y quién lo hace. De cada acción identificada anota:
- Quién: vecinos, alcaldía, gobierno de la ciudad, gobierno federal, sector privado, organizaciones civiles o academia. Si menciona la dependencia, anótala.
- Evaluación: si funciona o no, por qué se hace eso y no otra cosa, y qué lo limita.
En actores también van quienes la persona señala con mucho interés o mucha influencia sobre el agua, aunque no estén actuando.

En resumen y usando preguntas guia
| Término                                    | Pregunta guía                            |
|--------------------------------------------|------------------------------------------|
| Preocupaciónes principales acerca del agua | ¿Cuál es el problema del agua?           |
| Causas                                     | ¿Por qué pasa?                           |
| Consecuencias                              | ¿Qué provoca y a quién le pega?          |
| Acciones y actores                         | ¿Qué se hace, quién lo hace y qué falta? |

REGLAS GENERALES
1. Codifica la perspectiva de la persona, no lo que tú sabes que es cierto.
2. Registra los vínculos ("A provoca B"), no solo listas sueltas. Una misma idea puede ser consecuencia en una cadena y causa en otra (sobreexplotación -> hundimientos -> inundaciones).
3. Usa las palabras de la persona entrevistada; los términos se homologan después, no ahora.

Devuelve los datos en un JSON con las siguientes claves:
1. PREOCUPACIONES PRINCIPALES ACERCA DEL AGUA
2. CAUSAS PRINCIPALES
3. CONSECUENCIAS
4. ACCIONES Y ACTORES QUE REALIZAN DICHAS ACCIONES

"""

_PROMPT_FORMAT_STANDARD = """
FORMATO:
- Utiliza el lenguaje de la entrevista.
- Todo el listado debe estar en español, sin traducciones.
- No hagas resúmenes, ni interpretaciones, ni agregues información que no esté en la transcripción. 
- No hagas parrafos, solo listas de ideas separadas por saltos de línea (Nota:una idea puede tener varias palabras, pero no es una frase ni un parrafo).
- Todo dalo en español, devuelve la información en un JSON con las claves exactas que te di, sin agregar ni quitar ninguna clave.
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

    # ── P5: Quote-based (anclaje en citas + protocolo completo Dra. Iliana) ──
    # Metodología de dos pasos: localizar la cita antes de resumirla.
    # Incorpora todas las prohibiciones y reglas de vocabulario del Manual
    # de Reglas de Codificación MEGADAPT (Dra. Iliana Grave, oct 2026).
    # v2: reforzado con patrones de fallo observados en el primer CSV de salida.
    "quote_based": _PROMPT_BASE_DEFINITIONS + """
METODOLOGÍA PARA ESTA VARIANTE — razona en tres pasos (internamente):

  PASO 1 — PRUEBA DE CITA:
    Para cada ítem que consideres incluir, pregúntate: ¿puedo citar textualmente las
    palabras del entrevistado que respaldan esta etiqueta? Si la respuesta es NO,
    el ítem no va. No hay excepciones.

  PASO 2 — CONTEO HONESTO:
    Cuenta cuántos fragmentos DISTINTOS encontraste en el PASO 1 para cada campo.
    Ese número es el MÁXIMO de ítems que puedes devolver para ese campo.
    Si encontraste 2 fragmentos → devuelves máximo 2 ítems.
    Si encontraste 0 → devuelves lista vacía [].
    NUNCA añadas ítems extra para completar una lista de 4.

  PASO 3 — RESUMEN FIEL:
    Para cada fragmento, resume en 4-5 palabras usando el vocabulario LITERAL del
    entrevistado. Nunca sustituyas sus palabras por categorías analíticas externas.

══════════════════════════════════════════════════════════════════════
PROTOCOLO DE CODIFICACIÓN MEGADAPT — PROHIBICIONES ABSOLUTAS
(Manual de Reglas de Codificación, Dra. Iliana Grave)
══════════════════════════════════════════════════════════════════════

PROHIBICIÓN 1 — NO INFERIR CATEGORÍAS ANALÍTICAS ABSTRACTAS:
  La codificación registra lo que DIJO el entrevistado, no lo que implica su situación.
  Si el entrevistado describe una práctica o situación que "suena" a una categoría
  abstracta, codifica la práctica concreta, NUNCA el nombre de la categoría.

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  PALABRA PROHIBIDA           │  QUÉ CODIFICAR EN SU LUGAR                  │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │  "Corrupción"                │  El hecho concreto que describió:            │
  │  "Corrupción en permisos"    │  "Permisos otorgados sin revisión",          │
  │  "Corrupción en distribución"│  "Tráfico de influencias en contratos",      │
  │                              │  "Funcionarios favorecen ciertas colonias"   │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │  "Desigualdad"               │  La descripción concreta que usó:            │
  │  "Desigualdad en el acceso"  │  "Colonias sin agua diaria",                 │
  │  "Acceso desigual al agua"   │  "Tandeo solo en zonas pobres",              │
  │                              │  "Ricos tienen agua, pobres no"              │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │  "Cambio climático"          │  SOLO si el entrevistado pronunció esas      │
  │                              │  palabras. Si habló de lluvias intensas o    │
  │                              │  sequías, codifica eso: "Lluvias más         │
  │                              │  intensas que antes", "Sequías más largas"   │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │  "Inequidad"                 │  El caso específico que describió            │
  │  "Exclusión social"          │  Quién quedó fuera y por qué lo dijo él      │
  │  "Marginalización"           │  La situación concreta descrita              │
  │  "Marginalidad y pobreza"    │  Las condiciones específicas mencionadas     │
  │  "Clientelismo"              │  La práctica concreta que mencionó           │
  └─────────────────────────────────────────────────────────────────────────────┘

  ✗ ETIQUETAS GENÉRICAS PROHIBIDAS — detectadas como falsas en comparativa real:
    Las siguientes etiquetas aparecieron en CSV previos SIN respaldo textual.
    Están prohibidas a menos que el entrevistado las haya dicho con esas palabras:

      "Asentamientos irregulares"          (como causa genérica)
      "Afectaciones a la salud"            (demasiado genérico; usa la afectación específica)
      "Conflictos por agua"                (inferido; codifica el conflicto concreto)
      "Conflictos sociales por recursos"   (inferido)
      "Cambio climático"                   (causa por defecto; solo si lo nombró)
      "Desigualdad en el acceso"           (etiqueta de segundo orden; ver tabla arriba)
      "Impacto en la calidad de vida"      (demasiado genérico)
      "Degradación de servicios"           (demasiado genérico)
      "Riesgo hídrico en la ciudad"        (demasiado genérico)
      "Problemas de gobernanza"            (inferido; codifica la práctica concreta)

PROHIBICIÓN 2 — NO FABRICAR ACTORES EN ACCIONES:
  El actor en Acciones_y_Actores DEBE estar mencionado explícitamente en la
  transcripción como sujeto activo de esa acción concreta. Está prohibido inferir
  el actor por el tipo de acción o por el tema general de la entrevista.
  Instituciones PROHIBIDAS de agregar si el entrevistado no las nombró en ese contexto:
    SACMEX, CONAGUA, CENAPRED, PAOT, CORENA, Protección Civil.
  Si el entrevistado no nombra al actor de una acción → usa exactamente: "No especificado".
  Ejemplos de actores fabricados detectados (prohibidos):
    ✗ "CONAGUA: Desarrolla proyectos" (si CONAGUA no fue nombrada en esa acción)
    ✗ "Gobierno: Implementa programas" (demasiado vago; solo si hay actor explícito)

PROHIBICIÓN 3 — NUNCA FUERCES 4 ÍTEMS POR CAMPO:
  El número de ítems en cada campo viene del CONTEO DEL PASO 2, no de una plantilla.
  Una entrevista que solo toca 2 causas → Causas_principales tiene 2 ítems.
  Una entrevista sin acciones mencionadas → Acciones_y_Actores devuelve [].
  Nunca uses como valor de ningún campo: "Original", "Original action",
  "Acción original", "Acción no especificada", "Acción desconocida", ni ninguna variante.
  Si no hay contenido real → devuelve lista vacía [].

PROHIBICIÓN 4 — OBLIGATORIO PRESERVAR EL VOCABULARIO LOCAL Y TÉCNICO DEL PROYECTO:
  Los términos locales de la CDMX y los términos técnicos del proyecto MEGADAPT son
  analíticamente irreemplazables. DEBES conservarlos tal como los usa el entrevistado.

  TÉRMINOS CONFIRMADOS POR DRA. ILIANA — usar EXACTAMENTE estas formas:

    Término correcto                     │ NO usar estas formas
    ─────────────────────────────────────┼────────────────────────────────────────────
    tandeo                               │ distribución desigual, escasez por turnos,
                                         │ racionamiento, suministro intermitente
    basura                               │ residuos sólidos, desechos,
                                         │ contaminación por residuos  ← ERROR DETECTADO
    importación de agua de otras cuencas │ trasvase, suministro externo, agua de fuera
    Infraestructura insuficiente         │ deterioro de infraestructura,
                                         │ falta de mantenimiento, infraestructura deficiente
    pipas                                │ camiones cisterna, vehículos de abastecimiento
    barranca                             │ área natural protegida, zona de riesgo,
                                         │ zona de escurrimiento
    tinaco / tambo                       │ depósito, cisterna, contenedor de agua
    ejido / ejidatario / comunero        │ terreno comunitario, habitante, residente
    avecindado                           │ residente, habitante, inquilino
    sesgo presupuestal                   │ inequidad en la inversión, gasto desigual
    asbesto                              │ material de construcción, fibrocemento

  REGLA DE ORO: si el entrevistado dijo X, la etiqueta dice X resumida.
  Nunca sustituyas el término del entrevistado por un equivalente técnico o neutro.

PROHIBICIÓN 5 — NO INFERIR GEOLOCALIZACIÓN:
  No atribuyas delegación, alcaldía, colonia o localidad si la transcripción
  no lo indica de forma explícita. Si el entrevistado no mencionó el lugar,
  no lo incluyas como actor ni como contexto de ninguna etiqueta.

══════════════════════════════════════════════════════════════════════
NOMBRES INSTITUCIONALES — USAR SIEMPRE LA SIGLA OFICIAL
══════════════════════════════════════════════════════════════════════
  SACMEX   Sistema de Aguas de la Ciudad de México
  CONAGUA  Comisión Nacional del Agua
  CENAPRED Centro Nacional de Prevención de Desastres
  PAOT     Procuraduría Ambiental y del Ordenamiento Territorial
  CORENA   Comisión de Recursos Naturales
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
# 5. Construcción del prompt — carga desde archivo o desde memoria
# ──────────────────────────────────────────────────────────────────────────────
# Directorio de prompts activo. Se configura con set_prompts_dir() o --prompts-dir.
# None → usar los prompts definidos en PROMPT_VARIANTS (comportamiento v7 original).
_ACTIVE_PROMPTS_DIR: Optional[Path] = None

# Directorio predeterminado que se busca automáticamente si no se pasa --prompts-dir.
# Convención del proyecto: carpeta PROMPTS/ junto al script.
_DEFAULT_PROMPTS_DIR = Path(__file__).parent / "PROMPTS"


def set_prompts_dir(path: Optional[str]) -> None:
    """
    Configura el directorio del que se cargan los archivos .md de prompts.

    Si path es None, se intenta el directorio predeterminado (_DEFAULT_PROMPTS_DIR).
    Si ese tampoco existe, se usan los prompts en memoria sin advertencia.
    Si path es una ruta que no existe, se lanza FileNotFoundError.
    """
    global _ACTIVE_PROMPTS_DIR
    if path is not None:
        p = Path(path)
        if not p.is_dir():
            raise FileNotFoundError(
                f"Directorio de prompts no encontrado: {p.resolve()}\n"
                f"Asegúrate de que la carpeta existe y contiene archivos .md."
            )
        _ACTIVE_PROMPTS_DIR = p
        log.info(f"Prompts cargados desde directorio especificado: {p.resolve()}")
    else:
        if _DEFAULT_PROMPTS_DIR.is_dir():
            _ACTIVE_PROMPTS_DIR = _DEFAULT_PROMPTS_DIR
            log.info(f"Prompts cargados desde directorio predeterminado: {_DEFAULT_PROMPTS_DIR.resolve()}")
        else:
            _ACTIVE_PROMPTS_DIR = None
            log.debug("Directorio PROMPTS/ no encontrado; usando variantes en memoria.")


def build_prompt(variant: str, max_items: int = 4) -> str:
    """
    Devuelve el system prompt para la variante y max_items indicados.

    Orden de prioridad:
      1. {_ACTIVE_PROMPTS_DIR}/{variant}.md  — archivo editable del proyecto
      2. PROMPT_VARIANTS[variant]            — cadena en memoria (fallback)

    Si el archivo .md existe pero no contiene el marcador {max_items}, se
    devuelve el texto tal cual (el .md tiene control total sobre el prompt).
    Si el archivo no existe para la variante solicitada se usa el fallback y
    se emite un aviso.
    """
    if _ACTIVE_PROMPTS_DIR is not None:
        md_path = _ACTIVE_PROMPTS_DIR / f"{variant}.md"
        if md_path.exists():
            text = md_path.read_text(encoding="utf-8").strip()
            try:
                return text.format(max_items=max_items)
            except KeyError:
                # El .md no tiene {max_items} — lo devuelve sin formatear.
                return text
        else:
            log.warning(
                f"Archivo '{variant}.md' no encontrado en {_ACTIVE_PROMPTS_DIR}. "
                f"Usando variante en memoria como fallback."
            )

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
    temperature: Optional[float] = 0.0,
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


def _csv_name(prompt_variant: str) -> str:
    """Nombre canónico del CSV de salida para una variante dada."""
    return f"analysis_v7_{prompt_variant}.csv"


def save_checkpoints(result: InterviewResult, all_results: list[InterviewResult], out_dir: Path):
    d = out_dir / "individual"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{result.metadata.Id_entrevista}.json").write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    df = results_to_dataframe(all_results)
    if not df.empty:
        df.to_csv(out_dir / _csv_name(result.prompt_variant), index=False, encoding="utf-8-sig")


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
    y guarda los resultados en `output_folder/analysis_v7_{prompt_variant}.csv`.
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
        "prompts_dir": str(_ACTIVE_PROMPTS_DIR) if _ACTIVE_PROMPTS_DIR else "en memoria",
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
        out_csv = out_path / _csv_name(prompt_variant)
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        log.info(f"CSV consolidado: {out_csv.resolve()} ({len(df)} entrevistas)")
    else:
        log.warning("No se generaron resultados válidos; el CSV no fue creado.")

    # Registro acumulativo de uso
    usage = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "modelo": model,
        "prompt_variant": prompt_variant,
        "csv_salida": _csv_name(prompt_variant),
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
            print(f"  {variant:25s}  ${last.get('costo_usd_estimado', 0):.4f} USD"
                  f"  →  {last.get('csv_salida', '?')}")
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

  # Correr una variante (lee el prompt desde PROMPTS/quote_based.md si existe):
  python megadapt_qualitative_pipeline_v7.py \\
      --folder ./transcripciones --model gpt-4o-mini \\
      --prompt-variant quote_based
  # → salida: output_quote_based/analysis_v7_quote_based.csv

  # Especificar una carpeta de prompts distinta:
  python megadapt_qualitative_pipeline_v7.py \\
      --folder ./transcripciones --model gpt-4o-mini \\
      --prompt-variant quote_based \\
      --prompts-dir "./LLMs-ModelosMentales/PROMPTS"

  # Correr todas las variantes (una carpeta por variante):
  python megadapt_qualitative_pipeline_v7.py \\
      --folder ./transcripciones --model gpt-4o-mini --run-all-variants

  # Evaluar y comparar con métricas NLP:
  python megadapt_eval_nlp_2.py \\
      --human "Integracion.xlsx" \\
      --llm baseline=output_baseline/analysis_v7_baseline.csv \\
             strict_grounding=output_strict_grounding/analysis_v7_strict_grounding.csv \\
             quote_based=output_quote_based/analysis_v7_quote_based.csv \\
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
    ap.add_argument("--prompts-dir", default=None,
                    help=(
                        "Carpeta con archivos .md de prompts editables. "
                        "Si no se especifica, se busca automáticamente PROMPTS/ junto al script. "
                        "Si esa carpeta no existe, se usan los prompts definidos en el código."
                    ))
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

    # Configurar directorio de prompts (antes de cualquier llamada a build_prompt)
    set_prompts_dir(args.prompts_dir)

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
