from docling.document_converter import DocumentConverter
from pathlib import Path
import json
import uuid
import re
from datetime import datetime


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
    export_json: bool = True,
    export_markdown: bool = False
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
        "processed_at": datetime.now().isoformat(),
        "total_chunks": len(chunks),
        "chunks": chunks
    }

    # ============================================
    # JSON speichern
    # ============================================
    if export_json:

        json_file = Path(f"{base_name}_{doc_id}.json")

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(document_data, f, ensure_ascii=False, indent=4)

        print(f"\nJSON gespeichert:")
        print(json_file.absolute())

    # ============================================
    # Markdown speichern
    # ============================================
    if export_markdown:

        markdown_file = Path(f"{base_name}_{doc_id}.md")

        with open(markdown_file, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        print(f"\nMarkdown gespeichert:")
        print(markdown_file.absolute())

    return document_data


# ============================================
# Beispiele
# ============================================

pdf_file = r"C:\Users\Miral Ibrahim\OneDrive\Desktop\Hs\digitalisierung_docs\handgeschrieben.pdf"


# Nur JSON
process_document(
    pdf_file,
    export_json=True,
    export_markdown=False
)


# Nur Markdown
process_document(
    pdf_file,
    export_json=False,
    export_markdown=True
)


# Beide Formate
process_document(
    pdf_file,
    export_json=True,
    export_markdown=True
)