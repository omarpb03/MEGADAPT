import pandas as pd
from pathlib import Path

print("¡Rescatando datos y fusionando edgelists!")

# 1. Ruta a la carpeta donde están todos los .txt individuales
edgelists_dir = Path("output_prueba_v2/edgelists")

all_edges = []

# 2. Leer cada archivo .txt en la carpeta manualmente para evitar errores de comas
for txt_file in edgelists_dir.glob("*_edgelist.txt"):
    try:
        causes = []
        effects = []
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(",")
                if len(parts) >= 2:
                    # Si Gemini metió comas extra, unimos todo menos el último elemento como la causa.
                    # Reemplazamos cualquier coma interna por un espacio para que Cytoscape no sufra.
                    cause = " ".join(parts[:-1]).replace('"', '').strip()
                    effect = parts[-1].replace('"', '').strip()
                    
                    causes.append(cause)
                    effects.append(effect)
        
        if causes:
            df = pd.DataFrame({'cause': causes, 'effect': effects})
            df['frequency'] = 1
            all_edges.append(df)
            
    except Exception as e:
        print(f"Error crítico leyendo {txt_file.name}: {e}")

if not all_edges:
    print("No se encontraron archivos .txt para fusionar.")
else:
    # 3. Unir todas las tablas individuales
    df_master = pd.concat(all_edges, ignore_index=True)

    # 4. Agrupar por causa y efecto, sumando las frecuencias
    df_master = df_master.groupby(['cause', 'effect'], as_index=False)['frequency'].sum()

    # 5. Ordenar de mayor a menor frecuencia para el análisis de redes
    df_master = df_master.sort_values(by='frequency', ascending=False)

    # 6. Guardar el meta-modelo mental definitivo
    out_path = Path("output_prueba_v2/MEGADAPT_master_edgelist.csv")
    df_master.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"¡Fusión completada con éxito! Se procesaron {len(all_edges)} archivos individuales.")
    print(f"El archivo '{out_path.name}' está listo para importarse a Cytoscape.")