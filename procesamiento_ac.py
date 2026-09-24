import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
from openpyxl.styles import Alignment

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=20))
def call_openai(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[
            {"role": "system", "content": "Eres un traductor y estructurador de datos. Tu única tarea es traducir texto al español y separarlo en viñetas usando saltos de línea."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content.strip()

def format_text(text: str) -> str:
    if pd.isna(text) or not str(text).strip():
        return ""
    
    prompt = f"""
    Transforma el siguiente texto siguiendo estrictamente estas 3 reglas:
    1. TRADUCE absolutamente todo al español (Ejemplo: "Floods" -> "Inundaciones", "trash/garbage" -> "Basura").
    2. SEPARA cada idea distinta en un nuevo renglón (salto de línea).
    3. INICIA cada renglón con un guión corto y un espacio ("- ").
    
    Ejemplo de entrada:
    Sociales: 50% trash/garbage, Infraestructure: capacity of sawage is not enough
    
    Ejemplo de salida esperada:
    - Sociales: 50% basura
    - Infraestructura: la capacidad del drenaje no es suficiente

    Texto original a transformar:
    {text}
    
    Resultado (solo la lista, sin texto extra):
    """
    try:
        return call_openai(prompt)
    except Exception as e:
        print(f"\nError en celda: {e}")
        return text

print("Cargando el archivo original de humanos...")
df = pd.read_excel("2.Análisis Cualitativo Integración.xlsx", header=1)

# Búsqueda dinámica inmune a los espacios en blanco del Excel
columnas_objetivo = [col for col in df.columns if any(k in col for k in ['Preocupación', 'Causas', 'consecuencias', 'Acciones'])]

tqdm.pandas(desc="Traduciendo y formateando")

for col in columnas_objetivo:
    print(f"\nProcesando columna: {col.strip()[:30]}...")
    df[col] = df[col].progress_apply(format_text)

print("\nGuardando resultados con formato ajustado...")
archivo_salida = "Analisis_Cualitativo_Estandarizado.xlsx"

with pd.ExcelWriter(archivo_salida, engine='openpyxl') as writer:
    df.to_excel(writer, index=False, sheet_name="Analisis_Limpio")
    worksheet = writer.sheets["Analisis_Limpio"]
    
    for col in worksheet.columns:
        col_letter = col[0].column_letter
        worksheet.column_dimensions[col_letter].width = 50 
        for cell in col:
            cell.alignment = Alignment(wrap_text=True, vertical='top')

print(f"¡Éxito! El archivo '{archivo_salida}' ha sido generado 100% en español.")