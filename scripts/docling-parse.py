#!/usr/bin/env python3
"""
docling-parse.py — Parse PDFs with Docling, extracting section-aware text blocks.

Output: parsed-docs.json
  [ { "file": "001.pdf", "section": "Introduction", "text": "..." }, ... ]

Run:
  pip install docling
  python scripts/docling-parse.py
"""
import json
import sys
from pathlib import Path

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
except ImportError:
    print("ERROR: docling not installed.\n  pip install docling", file=sys.stderr)
    sys.exit(1)

DOCS_DIR = Path(__file__).parent / "docs"
OUTPUT   = Path(__file__).parent / "parsed-docs.json"

# Labels Docling assigns to document elements (lowercase string values)
HEADER_LABELS = {"section_header", "title", "heading", "page_header"}
BODY_LABELS   = {"text", "paragraph", "list_item", "caption", "footnote", "formula"}

def label_str(label) -> str:
    """Normalize a Docling DocItemLabel to a lowercase string."""
    if hasattr(label, "value"):
        return str(label.value).lower()
    return str(label).lower()


def parse_pdf(pdf_path: Path, converter: DocumentConverter) -> list[dict]:
    """Return [{section, text}] records for one PDF."""
    print(f"  Parsing {pdf_path.name} ...", flush=True)
    result = converter.convert(str(pdf_path))
    doc    = result.document

    records: list[dict] = []
    current_section = "General"   # catch-all before first header

    for item, _level in doc.iterate_items():
        text = getattr(item, "text", None)
        if not text or not text.strip():
            continue

        text  = text.strip()
        label = label_str(getattr(item, "label", "text"))

        if label in HEADER_LABELS:
            # Use the header text as the section name.
            # Skip very long "headers" – they're probably mis-classified body text.
            if len(text) <= 120:
                current_section = text
        elif any(label.startswith(l) for l in ("text", "paragraph", "list", "caption", "footnote", "formula")):
            records.append({
                "file":    pdf_path.name,
                "section": current_section,
                "text":    text,
            })

    print(f"    → {len(records)} text blocks, "
          f"{len({r['section'] for r in records})} sections detected")
    return records


def main() -> None:
    if not DOCS_DIR.exists():
        print(f"No docs directory at {DOCS_DIR}. Create it and add PDFs.", file=sys.stderr)
        sys.exit(1)

    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {DOCS_DIR}.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pdfs)} PDF(s). Initialising Docling (first run downloads models)...\n")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False            # skip OCR — faster for text-based PDFs
    pipeline_options.do_table_structure = False  # keep table text

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    all_records: list[dict] = []
    failed: list[str] = []

    for pdf in pdfs:
        try:
            records = parse_pdf(pdf, converter)
            all_records.extend(records)
        except Exception as exc:
            print(f"  WARNING: could not parse {pdf.name}: {exc}")
            failed.append(pdf.name)

    OUTPUT.write_text(json.dumps(all_records, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nDone. {len(all_records)} total text blocks → {OUTPUT}")
    if failed:
        print(f"Failed ({len(failed)}): {', '.join(failed)}")


if __name__ == "__main__":
    main()
