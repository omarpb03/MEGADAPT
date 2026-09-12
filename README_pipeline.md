# MEGADAPT – Pipeline de Análisis Cualitativo Automatizado

## Qué hace este script

Toma transcripciones de entrevistas y talleres (`.docx` o `.txt`) y produce, para cada una:

- **Edgelist individual** (`.txt`): lista de pares causa→efecto, directamente importable en Cytoscape o NetworkX.
- **Tabla de variables** (`.csv`): todas las variables por entrevista, sector y delegación.
- **Edgelist global** (`.csv`): meta-modelo mental acumulado con frecuencias de co-ocurrencia.
- **Resumen JSON** (`.json`): metadatos + narrativa resumida de cada participante.

El flujo sigue la metodología descrita en **Siqueiros et al. (2019)**.

---

## Instalación de dependencias

```bash
pip install google-generativeai python-docx pandas tqdm
```

---

## Uso básico

### Opción A – Línea de comandos

```bash
# Con transcripciones en Google Drive montado localmente
export GEMINI_API_KEY="tu_api_key_aqui"

python megadapt_qualitative_pipeline.py \
  --folder "/ruta/google_drive/Transcripciones" \
  --out    "./output_megadapt"
```

### Opción B – Prueba con pocas transcripciones (recomendado para empezar)

```bash
python megadapt_qualitative_pipeline.py \
  --folder "/ruta/a/transcripciones" \
  --out    "./output_prueba" \
  --limit  3 \
  --api-key "tu_api_key_aqui"
```

### Opción C – Con diccionario personalizado

```bash
python megadapt_qualitative_pipeline.py \
  --folder      "/ruta/a/transcripciones" \
  --out         "./output_megadapt" \
  --dictionary  "./dictionary_megadapt.json"
```

---

## Estructura de salida

```
output_megadapt/
├── edgelists/
│   ├── DF012015_GOV_edgelist.txt      ← una por entrevista
│   ├── I021415_OTR_edgelist.txt
│   └── ...
├── variables_table.csv                ← todas las variables, con metadatos
├── global_edgelist.csv                ← meta-modelo mental con frecuencias
└── analysis_summary.json              ← resúmenes narrativos + todo lo anterior
```

### Formato del edgelist individual

```
abastecimiento de agua,calidad del agua
crecimiento urbano,riesgo de inundación
infraestructura de drenaje,riesgo de inundación
capacidad institucional,inversión en infraestructura
```

Este formato es compatible con Cytoscape (importar como *Network from file → Edge list*) y con NetworkX:

```python
import networkx as nx
G = nx.read_edgelist("DF012015_GOV_edgelist.txt", delimiter=",", create_using=nx.DiGraph())
```

---

## Convención de nombres de archivo

El script infiere metadatos del nombre de archivo siguiendo la convención del proyecto:

| Prefijo geográfico | Delegación / Alcaldía          |
|--------------------|-------------------------------|
| `DF`               | Ciudad de México (gobierno)   |
| `I`                | Iztapalapa                    |
| `X`                | Xochimilco                    |
| `MC`               | Magdalena Contreras           |
| `FED`              | Federal                       |

| Sufijo de sector | Sector                  |
|------------------|------------------------|
| `_GOV` / `_GOB`  | Gobierno               |
| `_ACA`           | Académico              |
| `_OTR`           | ONG / Otro             |
| `_RES`           | Residente              |

Ejemplo: `I021415_OTR.docx` → Iztapalapa, Otro/ONG, entrevista individual.

---

## Diccionario de equivalencias

El archivo `dictionary_megadapt.json` contiene equivalencias de variables
(términos libres → términos estandarizados). Amplíalo con las equivalencias
específicas que el equipo vaya identificando durante la codificación cualitativa.

Formato:
```json
{
  "término que usa el entrevistado": "Variable Estandarizada",
  "pipas": "abastecimiento alternativo de agua",
  "tandeo": "abastecimiento de agua"
}
```

---

## Parámetros completos

| Parámetro       | Descripción                                              | Default                  |
|-----------------|----------------------------------------------------------|--------------------------|
| `--folder`      | Carpeta con transcripciones (.docx / .txt)               | requerido                |
| `--out`         | Carpeta de salida                                        | `./output_megadapt`      |
| `--api-key`     | API key de Gemini (o variable `GEMINI_API_KEY`)           | requerido                |
| `--model`       | Modelo Gemini a usar                                     | `gemini-1.5-flash`       |
| `--limit`       | Número máximo de archivos a procesar (para pruebas)      | sin límite               |
| `--dictionary`  | Ruta a JSON con equivalencias adicionales                | usa diccionario base     |

---

## Pasos siguientes

1. **Revisar y corregir** los edgelists generados con criterio experto (proceso descrito en Siqueiros et al. 2019).
2. **Actualizar el diccionario** con nuevas equivalencias encontradas.
3. **Importar el edgelist global** en Cytoscape para análisis de comunidades (Map Equation).
4. **Calcular el Índice de Jaccard** entre el meta-modelo y cada modelo individual para identificar narrativas dominantes.
5. Considerar escalar a `gemini-1.5-pro` si se necesita mayor profundidad de análisis.

---

## Referencia metodológica

> Siqueiros-García, J.M., Lerner, A.M., García-Valdecasas, J.I., Hernández-Aguilar, B.,
> & Romero-Saldaña, S.I. (2019). A standardization process for mental model analysis
> in socioecological systems. *Environmental Modelling & Software*, 112, 108–111.
> https://doi.org/10.1016/j.envsoft.2018.11.016
