"""
build_rich_dictionary.py
========================
Genera dictionary_megadapt_rich.json a partir del Excel del diccionario rico.

Uso:
    python build_rich_dictionary.py --excel Diccionario_New_Last_Version_010619.xlsx
    python build_rich_dictionary.py --excel ruta/al/archivo.xlsx --out mi_dict.json

El JSON resultante tiene la forma:
    {
      "expresion_coloquial": "termino_canonico",
      "expresion coloquial": "termino_canonico",   ← versión con espacios
      ...
    }

1 665 entradas → 150 términos canónicos.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def build(excel_path: str, out_path: str) -> None:
    df = pd.read_excel(excel_path, header=0, engine="openpyxl")

    # Validar columnas esperadas
    if list(df.columns[:2]) != ["TO", "TH"]:
        raise ValueError(
            f"El Excel debe tener las columnas 'TO' y 'TH' en las primeras dos posiciones. "
            f"Encontré: {df.columns.tolist()}"
        )

    rich_dict: dict[str, str] = {}

    for _, row in df.iterrows():
        raw_key   = str(row["TO"]).strip()
        std_value = str(row["TH"]).strip()

        # Versión con guiones bajos (tal como viene del Excel)
        rich_dict[raw_key] = std_value

        # Versión con espacios (para matchear lenguaje natural en los textos)
        key_spaces = raw_key.replace("_", " ")
        if key_spaces not in rich_dict:
            rich_dict[key_spaces] = std_value

    # Los términos TH que no tienen TO propio se mapean a sí mismos
    # (forma canónica: si el LLM la genera tal cual, pasa sin cambio)
    for _, row in df.iterrows():
        th = str(row["TH"]).strip()
        if th not in rich_dict:
            rich_dict[th] = th
        th_spaces = th.replace("_", " ")
        if th_spaces not in rich_dict:
            rich_dict[th_spaces] = th

    canonical_terms = sorted(set(rich_dict.values()))

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rich_dict, f, ensure_ascii=False, indent=2)

    print(f"✓ Diccionario generado: {out_path}")
    print(f"  Entradas totales  : {len(rich_dict)}")
    print(f"  Términos canónicos: {len(canonical_terms)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera el diccionario rico en JSON desde el Excel.")
    parser.add_argument("--excel", "-e", required=True,
        help="Ruta al archivo Excel del diccionario (columnas TO, TH)")
    parser.add_argument("--out", "-o", default="dictionary_megadapt_rich.json",
        help="Ruta de salida del JSON (default: dictionary_megadapt_rich.json)")
    args = parser.parse_args()

    build(args.excel, args.out)


if __name__ == "__main__":
    main()
