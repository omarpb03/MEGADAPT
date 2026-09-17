import pandas as pd

# 1. Cargar el Excel de los humanos ya LIMPIO
df_human = pd.read_excel("Analisis_Cualitativo_MC_Limpio.xlsx")
# Normalizar códigos (quitar guiones bajos y a mayúsculas) para que crucen exacto
df_human['ID_Normalizado'] = df_human['Código de la entrevista'].astype(str).str.replace('_', '').str.upper()

# 2. Cargar la salida del LLM (v4)
df_llm = pd.read_csv("output_MC/analysis_summary_v4.csv", encoding="utf-8-sig")
# Normalizamos el ID del LLM igual (quitando _GOB o _OTR para que coincidan)
df_llm['ID_Normalizado'] = df_llm['interview_id'].str.replace('_GOB', 'GOV').str.replace('_OTR', 'OTR').str.replace('_', '').str.upper()

# 3. Hacer el cruce (Merge)
df_comparativo = pd.merge(df_human, df_llm, on='ID_Normalizado', how='inner')

# 4. Seleccionar y ordenar las columnas para que queden lado a lado
columnas_finales = [
    'Código de la entrevista',
    # PREOCUPACIÓN PRINCIPAL
    'Preocupación principal acerca del agua', # Humano
    'preocupacion_principal',                 # LLM
    # CAUSAS
    'Causas principales \n\n(Biosíficos, Socio-institucional, Uso de Suelo, Infraestructural)', # Humano
    'causas_biofisicas',                      # LLM
    'causas_socio_institucionales',           # LLM
    'causas_uso_de_suelo',                    # LLM
    'causas_infraestructurales',              # LLM
    # CONSECUENCIAS
    'Mayores consecuencias ',                 # Humano
    'consecuencias',                          # LLM
    # ACCIONES
    'Acciones (específicar quién)',           # Humano
    'acciones'                                # LLM
]

df_reporte = df_comparativo[columnas_finales]

# 5. Guardar reporte final
df_reporte.to_excel("Comparativa_Humano_vs_LLM.xlsx", index=False)
print("¡Archivo 'Comparativa_Humano_vs_LLM.xlsx' generado con éxito!")