import json
import re
import uuid
from datetime import datetime
from pathlib import Path

from docling.document_converter import DocumentConverter


# ============================================
# Converter nur einmal initialisieren
# ============================================
converter = DocumentConverter()


# ============================================
# Markdown in semantische Chunks aufteilen
# ============================================
def create_chunks(markdown_text: str):

    sections = re.split(r'(?=^##\s)', markdown_text, flags=re.MULTILINE)

    chunks = []

    for index, section in enumerate(sections):

        section = section.strip()

        if not section:
            continue

        chunk_data = {
            "chunk_id": f"chunk_{index + 1}",
            "content": section
        }

        chunks.append(chunk_data)

    return chunks


# ============================================
# Hauptfunktion
# ============================================
def process_document(
    pdf_path: str,
    export_json: bool = False,
    export_markdown: bool = False,
    output_dir: str | None = None,
):

    # Dokument konvertieren
    result = converter.convert(pdf_path)

    # Dokumentobjekt
    doc = result.document

    # Markdown exportieren
    markdown_content = doc.export_to_markdown()

    # Chunks erzeugen
    chunks = create_chunks(markdown_content)

    # Dokument-ID erzeugen
    doc_id = str(uuid.uuid4())

    # Dateiname ohne Endung
    base_name = Path(pdf_path).stem

    # Strukturierte Daten
    document_data = {
        "doc_id": doc_id,
        "source_file": str(pdf_path),
        "source_name": Path(pdf_path).name,
        "processed_at": datetime.now().isoformat(),
        "markdown": markdown_content,
        "total_chunks": len(chunks),
        "chunks": chunks
    }

    export_dir = Path(output_dir) if output_dir else Path.cwd()
    export_dir.mkdir(parents=True, exist_ok=True)

    # ============================================
    # JSON speichern
    # ============================================
    if export_json:

        json_file = export_dir / f"{base_name}_{doc_id}.json"

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(document_data, f, ensure_ascii=False, indent=4)

        print(f"\nJSON gespeichert:")
        print(json_file.absolute())

    # ============================================
    # Markdown speichern
    # ============================================
    if export_markdown:

        markdown_file = export_dir / f"{base_name}_{doc_id}.md"

        with open(markdown_file, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        print(f"\nMarkdown gespeichert:")
        print(markdown_file.absolute())

    return document_data


def extrahiere_dokument(
    pdf_path: str,
    export_json: bool = False,
    export_markdown: bool = False,
    output_dir: str | None = None,
) -> dict:
    result = process_document(
        pdf_path,
        export_json=export_json,
        export_markdown=export_markdown,
        output_dir=output_dir,
    )
    path = Path(pdf_path)
    return {
        "text": result["markdown"],
        "dokument_key": result["doc_id"],
        "dateiname": path.name,
        "dateipfad": str(path.resolve()),
        "source_path": str(path.resolve()),
        "seiten": None,
        "chunks": result["chunks"],
        "total_chunks": result["total_chunks"],
        "processed_at": result["processed_at"],
        "quelle": "modul-b-docling",
    }


# ============================================
# Beispiele
# ============================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PDF mit Docling verarbeiten")
    parser.add_argument("pdf_file", help="Pfad zur PDF-Datei")
    parser.add_argument("--json", action="store_true", help="JSON exportieren")
    parser.add_argument("--markdown", action="store_true", help="Markdown exportieren")
    parser.add_argument("--output-dir", help="Exportverzeichnis")
    args = parser.parse_args()

    process_document(
        args.pdf_file,
        export_json=args.json,
        export_markdown=args.markdown,
        output_dir=args.output_dir,
    )
