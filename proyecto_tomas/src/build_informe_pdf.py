"""Genera informe/informe.pdf a partir de informe/informe.md.

Dependencias opcionales (no necesarias para la app):  pip install markdown weasyprint
En Windows, WeasyPrint requiere GTK; alternativa: abrir informe.md en VS Code o Word y exportar a PDF.
"""
from __future__ import annotations

import re
from pathlib import Path

import markdown
from weasyprint import HTML

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "informe" / "informe.md"
OUT = ROOT / "informe" / "informe.pdf"

CSS = """
@page { size: Letter; margin: 1.6cm 1.7cm; @bottom-center { content: counter(page) " / " counter(pages); font-size: 8pt; color: #666; } }
body { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 9.3pt; line-height: 1.32; color: #111; text-align: justify; }
h1 { font-size: 15pt; color: #12355b; margin: 0 0 2pt 0; }
h2 { font-size: 11pt; color: #12355b; border-bottom: 1px solid #12355b; margin: 9pt 0 3pt 0; }
p { margin: 3pt 0; }
ul { margin: 2pt 0 2pt 14pt; padding: 0; } li { margin: 1pt 0; }
table { border-collapse: collapse; width: 100%; font-size: 8.3pt; margin: 4pt 0; }
th, td { border: 1px solid #bbb; padding: 2pt 4pt; text-align: center; }
th { background: #e8eef5; } td:first-child { text-align: left; }
img { display: block; margin: 4pt auto; }
code { font-size: 8.3pt; background: #f3f3f3; padding: 0 2px; }
"""


def main() -> None:
    text = SRC.read_text(encoding="utf-8")
    text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.S)  # quitar encabezado YAML
    # ancho de imagen estilo pandoc {width=70%}
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)\{width=(\d+)%\}", r'<img alt="\1" src="\2" style="width:\3%">', text)
    html = markdown.markdown(text, extensions=["tables", "sane_lists"])
    doc = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html}</body></html>"
    HTML(string=doc, base_url=str(SRC.parent)).write_pdf(OUT)
    print("PDF generado:", OUT)


if __name__ == "__main__":
    main()
