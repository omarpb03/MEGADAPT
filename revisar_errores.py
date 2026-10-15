import json
import glob

print("Buscando entrevistas con error...")
for f in glob.glob('output_quote_based/individual/*.json'):
    with open(f, encoding='utf-8') as archivo:
        data = json.load(archivo)
        if data.get('error'):
            print(f"\nFalló {data['metadata']['Id_entrevista']}:")
            print(f"Motivo: {data['error']}")