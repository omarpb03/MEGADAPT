import pandas as pd

print("Generando Excel comparativo Lado a Lado...")

# 1. Cargar el Excel Humano limpio
df_human = pd.read_excel("Analisis_Cualitativo_MC_Limpio.xlsx")
# Normalizar ID para cruce exacto (quitar guiones bajos)
df_human['ID_Normalizado'] = df_human['Código de la entrevista'].astype(str).str.replace('_', '').str.upper()

# 2. Cargar el CSV del LLM (El que acaban de generar con la v4_4)
df_llm = pd.read_csv("./output_completo/analysis_summary_v4.csv", encoding="utf-8-sig")
# Normalizar ID igualando las nomenclaturas del gobierno/otros
df_llm['ID_Normalizado'] = df_llm['Id_entrevista'].astype(str).str.replace('_GOB', 'GOV').str.replace('_OTR', 'OTR').str.replace('_', '').str.upper()

# 3. Cruzar la información (Merge)
df_merged = pd.merge(df_human, df_llm, on='ID_Normalizado', how='inner')

# 4. Seleccionar columnas exactamente Lado a Lado
columnas = [
    'Código de la entrevista',
    # PREOCUPACIONES
    'Preocupación principal acerca del agua', 
    'Preocupacion_principal_acerca_del_agua',
    # CAUSAS
    'Causas principales \n\n(Biosíficos, Socio-institucional, Uso de Suelo, Infraestructural)',
    'Causas_principales',
    # CONSECUENCIAS
    'Mayores consecuencias ',
    'Consecuencias',
    # ACCIONES
    'Acciones (específicar quién)',
    'Acciones_y_Actores_que_realizan_dichas_acciones'
]

# Filtrar para evitar errores si hay espacios extra en los nombres
cols_existentes = [c for c in columnas if c in df_merged.columns]
df_reporte = df_merged[cols_existentes].copy()

# 5. Renombrar columnas para la Dra. Helena
df_reporte.columns = [
    'Entrevista ID',
    'Preocupación (HUMANO)', 'Preocupación (LLM)',
    'Causas (HUMANO)', 'Causas (LLM)',
    'Consecuencias (HUMANO)', 'Consecuencias (LLM)',
    'Acciones (HUMANO)', 'Acciones (LLM)'
]

df_reporte.to_excel("Comparativa_Final_Humano_vs_LLM.xlsx", index=False)
print("¡Éxito! El archivo 'Comparativa_Final_Humano_vs_LLM.xlsx' está listo.")