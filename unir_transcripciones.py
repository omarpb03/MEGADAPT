import os
import glob
from docx import Document

# Ruta exacta a donde están tus 35 documentos
carpeta_transcripciones = "./LLMs-ModelosMentales/TRANSCRIPCIONES/_TRANSCRIPCIONES_35"

# Buscar todos los docx en esa carpeta
archivos_docx = glob.glob(os.path.join(carpeta_transcripciones, "*.docx"))

print(f"Encontrados {len(archivos_docx)} archivos para unir...")

# Archivo de salida en la raíz de tu proyecto
with open('todas_las_entrevistas_megadapt.txt', 'w', encoding='utf-8') as outfile:
    for filename in archivos_docx:
        # Extraer solo el nombre corto para el título
        nombre_corto = os.path.basename(filename)
        outfile.write(f"\n\n{'='*40}\nENTREVISTA ID: {nombre_corto}\n{'='*40}\n\n")
        
        try:
            # Leer el archivo Word correctamente
            doc = Document(filename)
            texto_entrevista = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            outfile.write(texto_entrevista)
        except Exception as e:
            print(f"Error procesando {nombre_corto}: {e}")

print("¡Listo! Se creó el archivo 'todas_las_entrevistas_megadapt.txt'")