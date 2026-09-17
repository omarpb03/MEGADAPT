import pandas as pd

print("Generando Excel comparativo Lado a Lado...")

# 1. Cargar el Excel Humano limpio
try:
    df_human = pd.read_excel("Analisis_Cualitativo_MC_Limpio.xlsx")
except Exception as e:
    print(f"Error al leer el Excel humano: {e}")
    exit()

print("\n[INFO] Columnas detectadas en el Excel Humano:")
print(df_human.columns.tolist())

# Normalizar ID
col_id_human = 'Código de la entrevista' if 'Código de la entrevista' in df_human.columns else df_human.columns[0]
df_human['ID_Normalizado'] = df_human[col_id_human].astype(str).str.replace('_', '').str.upper()

# 2. Cargar el CSV del LLM
try:
    df_llm = pd.read_csv("./output_completo/analysis_summary_v4.csv", encoding="utf-8-sig")
except Exception as e:
    print(f"Error al leer el CSV del LLM: {e}")
    exit()

print("\n[INFO] Columnas detectadas en el CSV de Gemini:")
print(df_llm.columns.tolist())

col_id_llm = 'Id_entrevista' if 'Id_entrevista' in df_llm.columns else df_llm.columns[0]
df_llm['ID_Normalizado'] = df_llm[col_id_llm].astype(str).str.replace('_GOB', 'GOV').str.replace('_OTR', 'OTR').str.replace('_', '').str.upper()

# 3. Cruzar la información (Merge)
df_merged = pd.merge(df_human, df_llm, on='ID_Normalizado', how='inner')
print(f"\n[INFO] Se cruzaron {len(df_merged)} entrevistas exitosamente.")

# 4. Mapa de renombramiento seguro (Diccionario)
# Esto es a prueba de errores. Si el nombre no coincide exacto, no truena, solo lo ignora.
rename_map = {
    col_id_human: 'Entrevista ID',
    # HUMANOS (Variantes comunes para atrapar espacios fantasma)
    'Preocupación principal acerca del agua': 'Preocupación (HUMANO)', 
    'Causas principales \n\n(Biosíficos, Socio-institucional, Uso de Suelo, Infraestructural)': 'Causas (HUMANO)',
    'Mayores consecuencias ': 'Consecuencias (HUMANO)',
    'Mayores consecuencias': 'Consecuencias (HUMANO)',
    'Acciones (específicar quién)': 'Acciones (HUMANO)',
    
    # LLM (Las 4 nuevas de tu pizarra)
    'Preocupacion_principal_acerca_del_agua': 'Preocupación (LLM)',
    'Causas_principales': 'Causas (LLM)',
    'Consecuencias': 'Consecuencias (LLM)',
    'Acciones_y_Actores_que_realizan_dichas_acciones': 'Acciones (LLM)'
}

# Renombrar dinámicamente
df_reporte = df_merged.rename(columns=rename_map)

# Ordenar las columnas para el reporte
columnas_finales_deseadas = [
    'Entrevista ID',
    'Preocupación (HUMANO)', 'Preocupación (LLM)',
    'Causas (HUMANO)', 'Causas (LLM)',
    'Consecuencias (HUMANO)', 'Consecuencias (LLM)',
    'Acciones (HUMANO)', 'Acciones (LLM)'
]

# Dejar solo las que existen para que nunca haya error
columnas_existentes = [col for col in columnas_finales_deseadas if col in df_reporte.columns]
df_final = df_reporte[columnas_existentes]

df_final.to_excel("Comparativa_Final_Humano_vs_LLM.xlsx", index=False)
print("\n¡Éxito! El archivo 'Comparativa_Final_Humano_vs_LLM.xlsx' está listo en la carpeta.")