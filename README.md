# PDF Document AI Pipeline (Docling + RAG Ready)

Dieses Projekt ist eine lokale Dokumentenverarbeitungs-Pipeline, die PDFs in strukturierte Daten für spätere Nutzung in LLM- oder RAG-Systemen umwandelt.

Es nutzt **Docling** zur Layout- und Textextraktion und bietet Export in **Markdown**, **JSON** sowie ein einfaches Chunking für spätere Embedding-Pipelines.

---

## Features

- PDF → strukturierte Dokumentanalyse
- Layout- und Tabellen-Erkennung (Docling)
- Markdown-Export (LLM-freundlich)
- JSON-Export (maschinenlesbar)
- Automatisches Chunking nach Überschriften
- Optionaler Export:
  - JSON
  - Markdown
  - beide Formate

---

## Installation

### Voraussetzungen

- Python 3.10+
- pip

### Abhängigkeiten installieren

```bash
pip install docling