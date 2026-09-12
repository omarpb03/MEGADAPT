"""
compare_with_human_coding.py
=============================
Genera un Excel de comparación lado a lado entre la codificación humana
(equipo de la Dra. Hallie, columnas H-K de Análisis_Cualitativo_de_Entrevistas_*.xlsx)
y la salida del pipeline v4 (analysis_summary_v4.csv), alineadas por código
de entrevista.

El Excel humano puede tener varias hojas (una por delegación/zona); el
script las recorre todas. Los códigos de entrevista se normalizan
(mayúsculas, sin guiones/espacios) para poder emparejar formatos
inconsistentes como "MC_022515_GOV" vs "MC022515_gov".

Uso:
    python compare_with_human_coding.py \\
        --human-excel Análisis_Cualitativo_de_Entrevistas_MC.xlsx \\
        --llm-csv output_v4/analysis_summary_v4.csv \\
        --out comparacion_MC.xlsx
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

# Columnas del Excel humano (posición 0-indexed dentro de cada hoja)
COL_SECTOR = "Sector"
COL_CODIGO = "Código de la entrevista"
COL_PREOCUPACION = "Preocupación principal acerca del agua"
COL_CAUSAS = "Causas principales                                                  (Biosíficos, Socio-institucional, Uso de Suelo, Infraestructural)"
COL_CONSECUENCIAS = "Mayores consecuencias "
COL_ACCIONES = "Acciones (específicar quién)"


def normalize_code(code: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(code)).upper()


def load_human_coding(excel_path: str) -> pd.DataFrame:
    """Lee todas las hojas del Excel humano y las concatena en un solo DataFrame,
    quedándose solo con las columnas relevantes para la comparación."""
    xls = pd.ExcelFile(excel_path)
    frames = []
    for sheet in xls.sheet_names:
        # header=1 porque la fila 1 (Agua/Suelo) es un encabezado agrupador extra
        df = xls.parse(sheet, header=1)
        keep_cols = [c for c in [COL_SECTOR, COL_CODIGO, COL_PREOCUPACION,
                                  COL_CAUSAS, COL_CONSECUENCIAS, COL_ACCIONES] if c in df.columns]
        if COL_CODIGO not in keep_cols:
            continue
        df = df[keep_cols].copy()
        df = df[df[COL_CODIGO].notna()]
        # Descarta filas de resumen/pie de tabla (ej. "RESUMEN") que no son
        # códigos de entrevista reales — un código real siempre trae dígitos.
        df = df[df[COL_CODIGO].astype(str).str.contains(r"\d")]
        df["zona_hoja"] = sheet
        frames.append(df)
    if not frames:
        raise ValueError(
            f"No encontré la columna '{COL_CODIGO}' en ninguna hoja de {excel_path}. "
            "Revisa que los encabezados coincidan (fila 2 del Excel)."
        )
    human_df = pd.concat(frames, ignore_index=True)
    human_df["codigo_normalizado"] = human_df[COL_CODIGO].apply(normalize_code)
    return human_df


def load_llm_output(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["codigo_normalizado"] = df["interview_id"].apply(normalize_code)
    # Consolida las 4 subcategorías de causas del LLM en un solo bloque de texto
    # para que quede comparable, en una sola celda, contra la columna humana.
    def join_causas(row) -> str:
        bloques = []
        for label, col in [
            ("Biofísicas", "causas_biofisicas"),
            ("Socio-institucionales", "causas_socio_institucionales"),
            ("Uso de suelo", "causas_uso_de_suelo"),
            ("Infraestructurales", "causas_infraestructurales"),
        ]:
            val = row.get(col, "")
            if isinstance(val, str) and val.strip():
                bloques.append(f"[{label}]\n{val}")
        return "\n\n".join(bloques)

    df["causas_llm_consolidadas"] = df.apply(join_causas, axis=1)
    return df


def build_comparison(human_df: pd.DataFrame, llm_df: pd.DataFrame) -> pd.DataFrame:
    merged = human_df.merge(
        llm_df, on="codigo_normalizado", how="outer", suffixes=("_humano", "_llm"), indicator=True
    )

    def estado(row) -> str:
        if row["_merge"] == "both":
            return "Comparado"
        elif row["_merge"] == "left_only":
            return "Sin analizar por el LLM (falta transcripción o error)"
        else:
            return "Sin codificación humana (entrevista nueva)"

    merged["estado"] = merged.apply(estado, axis=1)

    out = pd.DataFrame({
        "codigo_entrevista": merged[COL_CODIGO].fillna(merged.get("interview_id")),
        "sector": merged[COL_SECTOR].fillna(merged.get("sector")),
        "estado": merged["estado"],
        "preocupacion_humano": merged.get(COL_PREOCUPACION),
        "preocupacion_llm": merged.get("preocupacion_principal"),
        "causas_humano": merged.get(COL_CAUSAS),
        "causas_llm": merged.get("causas_llm_consolidadas"),
        "consecuencias_humano": merged.get(COL_CONSECUENCIAS),
        "consecuencias_llm": merged.get("consecuencias"),
        "acciones_humano": merged.get(COL_ACCIONES),
        "acciones_llm": merged.get("acciones"),
    })
    return out


def write_formatted_excel(df: pd.DataFrame, out_path: str) -> None:
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Comparación")
        ws = writer.sheets["Comparación"]

        header_fill_humano = PatternFill("solid", fgColor="FCE4D6")
        header_fill_llm = PatternFill("solid", fgColor="DDEBF7")
        bold = Font(bold=True)

        for col_idx, col_name in enumerate(df.columns, start=1):
            cell = ws.cell(row=1, column=col_idx)
            cell.font = bold
            if col_name.endswith("_humano"):
                cell.fill = header_fill_humano
            elif col_name.endswith("_llm"):
                cell.fill = header_fill_llm

        widths = {
            "codigo_entrevista": 18, "sector": 16, "estado": 30,
            "preocupacion_humano": 40, "preocupacion_llm": 40,
            "causas_humano": 45, "causas_llm": 45,
            "consecuencias_humano": 40, "consecuencias_llm": 40,
            "acciones_humano": 40, "acciones_llm": 40,
        }
        for col_idx, col_name in enumerate(df.columns, start=1):
            letter = get_column_letter(col_idx)
            ws.column_dimensions[letter].width = widths.get(col_name, 25)

        for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        ws.freeze_panes = "D2"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compara la codificación humana vs la salida del LLM (v4)")
    parser.add_argument("--human-excel", required=True,
        help="Excel con la codificación humana (una o varias hojas, columnas H-K).")
    parser.add_argument("--llm-csv", required=True,
        help="analysis_summary_v4.csv generado por megadapt_qualitative_pipeline_v4.py")
    parser.add_argument("--out", default="comparacion_humano_vs_llm.xlsx",
        help="Ruta de salida del Excel de comparación.")
    args = parser.parse_args()

    human_df = load_human_coding(args.human_excel)
    llm_df = load_llm_output(args.llm_csv)
    comparison = build_comparison(human_df, llm_df)
    write_formatted_excel(comparison, args.out)

    n_both = (comparison["estado"] == "Comparado").sum()
    print(f"✓ Comparación generada: {args.out}")
    print(f"  Entrevistas comparadas: {n_both}")
    print(f"  Total de filas: {len(comparison)}")


if __name__ == "__main__":
    main()
