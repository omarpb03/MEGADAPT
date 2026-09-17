import pandas as pd
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

def clean_and_translate(text: str) -> str:
    if pd.isna(text) or not str(text).strip():
        return ""
    
    prompt = f"""
    Eres un asistente de limpieza de datos. Tu tarea es arreglar el siguiente texto de un análisis cualitativo:
    1. Traduce todo al Español.
    2. Separa las ideas distintas por un salto de línea (enter) con un guión corto (-) al inicio. Reemplaza las comas o puntos y comas que separan ideas por estos saltos de línea.
    3. Corrige la ortografía y redacción sin cambiar el sentido original.
    
    Texto original:
    {text}
    
    Devuelve ÚNICAMENTE el texto limpio y formateado, sin introducciones.
    """
    try:
        response = client.models.generate_content(model="gemini-3.6-flash", contents=prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error procesando celda: {e}")
        return text
)
# Lee el Excel (ajusta el nombre si es necesario y nos saltamos la primera fila vacía)
df = pd.read_excel("2.Análisis Cualitativo Integración.xlsx", header=1)

# Columnas a limpiar
columnas_objetivo = [
    'Preocupación principal acerca del agua',
    'Causas principales \n\n(Biosíficos, Socio-institucional, Uso de Suelo, Infraestructural)',
    'Mayores consecuencias ',
    'Acciones (específicar quién)'
]

for col in columnas_objetivo:
    if col in df.columns:
        print(f"Limpiando columna: {col}")
        df[col] = df[col].apply(clean_and_translate)

df.to_excel("Analisis_Cualitativo_MC_Limpio.xlsx", index=False)
print("¡Listo! El archivo 'Analisis_Cualitativo_MC_Limpio.xlsx' ha sido generado.")