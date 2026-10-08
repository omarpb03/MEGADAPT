import pandas as pd

print("Generando Excel comparativo Lado a Lado (Usando formato estandarizado de Hallie)...")

# 1. Cargar el Excel Humano Estandarizado
try:
    # Usamos header=1 porque la fila 0 tiene los encabezados reales ("Sector", "Código...", etc.)
    df_human = pd.read_excel("2_Análisis_Cualitativo_Integración_formato_estandarizado.xlsx", header=1)
    
    # La columna 1 es "Código de la entrevista"
    col_id_human = df_human.columns[1]
    df_human = df_human.dropna(subset=[col_id_human]) # Quitar filas sin código
    
except Exception as e:
    print(f"Error al leer el Excel humano: {e}")
    exit()

# Normalizar ID humano
df_human['ID_Normalizado'] = df_human[col_id_human].astype(str).str.replace('_', '').str.upper()

# 2. Cargar el CSV del LLM (analysis_v5.csv crudo)
try:
    df_llm = pd.read_csv("./output_v5/analysis_v5.csv", encoding="utf-8-sig")
except Exception as e:
    # Intento fallback si no está en carpeta
    try:
        df_llm = pd.read_csv("analysis_v5.csv", encoding="utf-8-sig")
    except Exception as e2:
        print(f"Error al leer el CSV del LLM: {e2}")
        exit()

col_id_llm = 'Id_entrevista' if 'Id_entrevista' in df_llm.columns else df_llm.columns[0]
df_llm['ID_Normalizado'] = df_llm[col_id_llm].astype(str).str.replace('_GOB', 'GOV').str.replace('_OTR', 'OTR').str.replace('_', '').str.upper()

# 3. Cruzar la información (Merge)
df_merged = pd.merge(df_human, df_llm, on='ID_Normalizado', how='inner')
print(f"\n[INFO] Se cruzaron {len(df_merged)} entrevistas exitosamente.")

# 4. Agrupar la codificación humana por entrevista (por si hay múltiples filas)
df_merged = df_merged.groupby('ID_Normalizado').first().reset_index()

# 5. Buscar dinámicamente las columnas para mapear
col_human_preoc = [c for c in df_human.columns if 'preocupaci' in str(c).lower()][0]
col_human_causa = [c for c in df_human.columns if 'causas principales' in str(c).lower()][0]
col_human_consec = [c for c in df_human.columns if 'consecuencias' in str(c).lower()][0]
col_human_accion = [c for c in df_human.columns if 'acciones' in str(c).lower()][0]

rename_map = {
    col_id_human: 'Entrevista ID',
    col_human_preoc: 'Preocupación (HUMANO)', 
    col_human_causa: 'Causas (HUMANO)',
    col_human_consec: 'Consecuencias (HUMANO)',
    col_human_accion: 'Acciones (HUMANO)',
    
    'Preocupacion_principal_acerca_del_agua': 'Preocupación (LLM)',
    'Causas_principales': 'Causas (LLM)',
    'Consecuencias': 'Consecuencias (LLM)',
    'Acciones_y_Actores_que_realizan_dichas_acciones': 'Acciones (LLM)'
}

df_reporte = df_merged.rename(columns=rename_map)

columnas_finales_deseadas = [
    'Entrevista ID',
    'Preocupación (HUMANO)', 'Preocupación (LLM)',
    'Causas (HUMANO)', 'Causas (LLM)',
    'Consecuencias (HUMANO)', 'Consecuencias (LLM)',
    'Acciones (HUMANO)', 'Acciones (LLM)'
]

columnas_existentes = [col for col in columnas_finales_deseadas if col in df_reporte.columns]
df_final = df_reporte[columnas_existentes]

df_final.to_excel("Comparativa_Final_Humano_vs_LLM_v5_Cruda.xlsx", index=False)
print("\n¡Éxito! El archivo 'Comparativa_Final_Humano_vs_LLM_v5_Cruda.xlsx' está listo.")