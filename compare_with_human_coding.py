import pandas as pd
from openpyxl.styles import Alignment

print("Generando Excel comparativo final (Humano vs LLM v5)...")

# 1. Cargar el Excel Humano estandarizado (NUEVO)
try:
    df_human = pd.read_excel("Analisis_Cualitativo_Estandarizado.xlsx")
    print("✓ Archivo humano cargado: Analisis_Cualitativo_Estandarizado.xlsx")
except Exception as e:
    print(f"❌ Error: No se encuentra 'Analisis_Cualitativo_Estandarizado.xlsx'.")
    exit()

# Tomar dinámicamente la columna del ID
col_id_human = df_human.columns[1] 
df_human['ID_Normalizado'] = df_human[col_id_human].astype(str).str.replace('_', '').str.upper()

# 2. Cargar el CSV del LLM (VERSIÓN 5)
ruta_llm = "./output_v5/analysis_standardized_v5.csv"
try:
    df_llm = pd.read_csv(ruta_llm, encoding="utf-8-sig")
    print("✓ Archivo LLM cargado: analysis_standardized_v5.csv")
except Exception as e:
    # Intento de rescate por si el archivo se guardó en la raíz de la carpeta
    try:
        df_llm = pd.read_csv("analysis_standardized_v5.csv", encoding="utf-8-sig")
        print("✓ Archivo LLM cargado: analysis_standardized_v5.csv")
    except:
        print("❌ Error: No se encuentra el CSV de la versión 5.")
        exit()

col_id_llm = 'Id_entrevista' if 'Id_entrevista' in df_llm.columns else df_llm.columns[0]
df_llm['ID_Normalizado'] = df_llm[col_id_llm].astype(str).str.replace(
    '_GOB', 'GOV').str.replace('_OTR', 'OTR').str.replace('_', '').str.upper()

# 3. Cruzar la información (Merge)
df_merged = pd.merge(df_human, df_llm, on='ID_Normalizado', how='inner')
print(f"\n[INFO] Se cruzaron {len(df_merged)} entrevistas exitosamente.")

# 4. Mapeo dinámico de columnas (Inmune a los espacios del Excel humano)
rename_map = {col_id_human: 'Entrevista ID'}

for col in df_human.columns:
    if "Preocupación" in col:
        rename_map[col] = 'Preocupación (HUMANO)'
    elif "Causas" in col:
        rename_map[col] = 'Causas (HUMANO)'
    elif "consecuencias" in col:
        rename_map[col] = 'Consecuencias (HUMANO)'
    elif "Acciones" in col:
        rename_map[col] = 'Acciones (HUMANO)'

# Mapeo exacto de las nuevas columnas de la versión 5 de OpenAI
rename_map.update({
    'Preocupacion_principal_acerca_del_agua': 'Preocupación (LLM)',
    'Causas_principales': 'Causas (LLM)',
    'Consecuencias': 'Consecuencias (LLM)',
    'Acciones_y_Actores_que_realizan_dichas_acciones': 'Acciones (LLM)'
})

df_reporte = df_merged.rename(columns=rename_map)

columnas_finales = [
    'Entrevista ID',
    'Preocupación (HUMANO)', 'Preocupación (LLM)',
    'Causas (HUMANO)', 'Causas (LLM)',
    'Consecuencias (HUMANO)', 'Consecuencias (LLM)',
    'Acciones (HUMANO)', 'Acciones (LLM)'
]

columnas_existentes = [col for col in columnas_finales if col in df_reporte.columns]
df_final = df_reporte[columnas_existentes]

# 5. Guardar el comparativo final
archivo_salida = "Comparativa_Final_Humano_vs_LLM_v5.xlsx"
with pd.ExcelWriter(archivo_salida, engine='openpyxl') as writer:
    df_final.to_excel(writer, index=False, sheet_name="Comparativa")
    worksheet = writer.sheets["Comparativa"]
    
    for col in worksheet.columns:
        col_letter = col[0].column_letter
        worksheet.column_dimensions[col_letter].width = 45 
        for cell in col:
            cell.alignment = Alignment(wrap_text=True, vertical='top')

print(f"\n¡Éxito! El archivo '{archivo_salida}' está listo.")
