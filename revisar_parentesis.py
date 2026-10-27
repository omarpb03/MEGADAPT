"""
Lista todo lo que aparece entre paréntesis ( ) y entre corchetes [ ] en las
transcripciones, para decidir si el preproceso del pipeline debe borrarlo.

Usa la misma regla que preprocess(): solo detecta lo que abre y cierra en el
mismo renglón.

Uso:
    python revisar_parentesis.py <carpeta de transcripciones>

Salida:
    - En la terminal: lo más frecuente de cada tipo.
    - revision_parentesis.csv (en la carpeta desde donde lo corres): la lista
      completa, con cuántas veces aparece cada texto y un archivo de ejemplo.
"""
import csv
import re
import sys
from collections import Counter
from pathlib import Path


def load_text(path: Path) -> str:
    if path.suffix.lower() == ".docx":
        from docx import Document
        return "\n".join(p.text for p in Document(str(path)).paragraphs)
    return path.read_text(encoding="utf-8", errors="replace")


def main(folder: str) -> None:
    files = sorted(
        f for f in Path(folder).rglob("*")
        if not f.name.startswith("~$") and f.suffix.lower() in (".docx", ".txt", ".md")
    )
    if not files:
        sys.exit(f"No se encontraron transcripciones en {folder}")

    conteo = {"( )": Counter(), "[ ]": Counter()}
    ejemplo = {}
    patrones = {"( )": r"\((.*?)\)", "[ ]": r"\[(.*?)\]"}

    for f in files:
        text = load_text(f)
        for tipo, patron in patrones.items():
            for m in re.findall(patron, text):
                k = m.strip()
                conteo[tipo][k] += 1
                ejemplo.setdefault((tipo, k), f.name)

    print(f"Archivos revisados: {len(files)}\n")
    for tipo, cnt in conteo.items():
        print(f"=== ENTRE {tipo}: {sum(cnt.values())} apariciones, "
              f"{len(cnt)} textos distintos ===")
        for k, n in cnt.most_common(25):
            print(f"{n:5d}  {tipo[0]}{k[:80]}{tipo[2]}   ej. {ejemplo[(tipo, k)]}")
        print()

    out = Path("revision_parentesis.csv")
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["tipo", "texto", "apariciones", "archivo_ejemplo"])
        for tipo, cnt in conteo.items():
            for k, n in cnt.most_common():
                w.writerow([tipo, k, n, ejemplo[(tipo, k)]])
    print(f"Lista completa en: {out.resolve()}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Uso: python revisar_parentesis.py <carpeta de transcripciones>")
    main(sys.argv[1])
