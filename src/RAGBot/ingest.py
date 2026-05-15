"""
ingest.py – Dokumente in die Vektordatenbank (ChromaDB) laden.


PDF-Extraktion:
  - pdfplumber  → Tabellen werden als lesbarer Text rekonstruiert (Hauptmethode)
  - PyPDF       → Fallback falls pdfplumber fehlschlägt

Unterstützte Formate: PDF, DOCX, TXT

"""

import argparse
import os
import shutil
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import track
from rich.panel import Panel

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

try:
    import pdfplumber
    PDFPLUMBER_VERFUEGBAR = True
except ImportError:
    PDFPLUMBER_VERFUEGBAR = False

console = Console()

# ──────────────────────────────────────────────
# Konfiguration
# ──────────────────────────────────────────────
CHROMA_PFAD    = "./vectordb"
COLLECTION     = "projektdokumente"

# Bewährt aus AWMF-Projekt: multilingual-e5-small, Deutsch-optimiert, nur 117 MB
EMBEDDING_MODELL = "intfloat/multilingual-e5-small"

# Bewährt aus AWMF-Projekt: 1000/200 erhält Kontext besser als 500/100
CHUNK_GROESSE       = 1000
CHUNK_UEBERLAPPUNG  = 200


def bereinige_text(text: str) -> str:
    """
    Einfache Textbereinigung 
    Entfernt Literaturverweise wie [1], [12, 45] und normalisiert Leerzeilen.
    """
    import re
    # Literaturverweise entfernen (z.B. aus Berichten mit Quellenangaben)
    text = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", text)
    # Mehrfache Leerzeilen auf eine reduzieren
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tabelle_zu_text(tabelle: list) -> str:
    """
    Wandelt eine pdfplumber-Tabelle in lesbaren Fließtext um.


    
      Schlüssel-Wert-Format bleibt beim Chunking immer zusammen.
    """
    if not tabelle or len(tabelle) < 2:
        return ""

    # Erste Zeile = Spaltenköpfe, None-Werte durch "-" ersetzen
    koepfe = [str(z).strip() if z else "-" for z in tabelle[0]]
    zeilen = []

    for zeile in tabelle[1:]:
        if not any(z for z in zeile):   # komplett leere Zeile überspringen
            continue
        werte = [str(z).strip() if z else "-" for z in zeile]
        # Jede Zeile als "Kopf: Wert | Kopf: Wert" formatieren
        paare = [f"{k}: {v}" for k, v in zip(koepfe, werte) if v != "-"]
        if paare:
            zeilen.append(" | ".join(paare))

    return "\n".join(zeilen)


def lade_pdf_mit_tabellen(pfad: Path) -> list:
    """
    Lädt ein PDF seitenweise mit pdfplumber.
    Pro Seite: erst Fließtext, dann Tabellen als Schlüssel-Wert-Text.
    Gibt Liste von LangChain Document-Objekten zurück.
    """
    docs = []
    tabellen_gesamt = 0

    with pdfplumber.open(str(pfad)) as pdf:
        for seiten_nr, seite in enumerate(pdf.pages, 1):

            # ── 1. Fließtext extrahieren ──────────────────────────
            fliesstext = seite.extract_text() or ""

            # ── 2. Tabellen extrahieren & als Text rekonstruieren ──
            tabellen_text_teile = []
            for tabelle in seite.extract_tables():
                t_text = tabelle_zu_text(tabelle)
                if t_text:
                    tabellen_text_teile.append(t_text)
                    tabellen_gesamt += 1

            # ── 3. Seiten-Text zusammenbauen ──────────────────────
            seiten_inhalt = fliesstext.strip()
            if tabellen_text_teile:
                # Tabellen-Block mit Trenner anhängen
                seiten_inhalt += "\n\n[TABELLE]\n" + "\n\n[TABELLE]\n".join(tabellen_text_teile)

            if len(seiten_inhalt.strip()) < 50:
                continue

            docs.append(Document(
                page_content=seiten_inhalt,
                metadata={
                    "dateiname":  pfad.name,
                    "dateipfad":  str(pfad.resolve()),
                    "seite":      seiten_nr,
                    "tabellen":   len(tabellen_text_teile),
                }
            ))

    if tabellen_gesamt > 0:
        console.print(f"   [cyan]↳ {tabellen_gesamt} Tabelle(n) als Text rekonstruiert[/cyan]")

    return docs


def lade_dokument(dateipfad: str) -> list:
    """
    Lädt ein einzelnes Dokument je nach Dateiendung.
    PDFs: pdfplumber (Tabellen-aware) mit PyPDF als Fallback.
    """
    pfad = Path(dateipfad)
    endung = pfad.suffix.lower()

    if endung == ".pdf":
        # pdfplumber bevorzugen (Tabellen-Unterstützung)
        if PDFPLUMBER_VERFUEGBAR:
            try:
                docs = lade_pdf_mit_tabellen(pfad)
                if docs:
                    for doc in docs:
                        doc.page_content = bereinige_text(doc.page_content)
                    return [d for d in docs if len(d.page_content.strip()) > 50]
            except Exception as e:
                console.print(f"   [yellow]pdfplumber Fehler ({e}) → Fallback auf PyPDF[/yellow]")

        # Fallback: PyPDF (kein Tabellen-Verständnis)
        console.print(f"   [dim]Verwende PyPDF (pdfplumber nicht verfügbar)[/dim]")
        loader = PyPDFLoader(str(pfad))
        docs   = loader.load()

    elif endung in (".docx", ".doc"):
        loader = Docx2txtLoader(str(pfad))
        docs   = loader.load()
    elif endung == ".txt":
        loader = TextLoader(str(pfad), encoding="utf-8")
        docs   = loader.load()
    else:
        console.print(f"[yellow]⚠ Unbekanntes Format: {pfad.name} – wird übersprungen.[/yellow]")
        return []

    for doc in docs:
        doc.page_content = bereinige_text(doc.page_content)
        doc.metadata["dateiname"] = pfad.name
        doc.metadata["dateipfad"] = str(pfad.resolve())

    return [d for d in docs if len(d.page_content.strip()) > 50]


def lade_ordner(ordnerpfad: str) -> list:
    """Lädt alle unterstützten Dokumente aus einem Ordner rekursiv."""
    ordner = Path(ordnerpfad)
    dateien = (
        list(ordner.rglob("*.pdf"))
        + list(ordner.rglob("*.docx"))
        + list(ordner.rglob("*.txt"))
    )

    if not dateien:
        console.print(f"[red]Keine unterstützten Dateien in '{ordnerpfad}' gefunden.[/red]")
        return []

    alle_docs = []
    for datei in track(dateien, description="📄 Dokumente laden..."):
        docs = lade_dokument(str(datei))
        alle_docs.extend(docs)
        console.print(f"   [dim]{datei.name}[/dim] → [green]{len(docs)} Seite(n)[/green]")

    return alle_docs


def chunking(dokumente: list) -> list:
    """
    Zerlegt Dokumente in überlappende Chunks.
    1000/200 bewährt aus AWMF-Projekt für langen Kontext
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_GROESSE,
        chunk_overlap=CHUNK_UEBERLAPPUNG,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(dokumente)

    # E5-Prefix hinzufügen – notwendig für multilingual-e5-small
    # Siehe: https://huggingface.co/intfloat/multilingual-e5-small
    for chunk in chunks:
        chunk.page_content = "passage: " + chunk.page_content

    return chunks


def speichere_in_chromadb(chunks: list) -> Chroma:
    """Erzeugt Embeddings und speichert Chunks in ChromaDB."""
    console.print("\n[cyan] Lade Embedding-Modell (multilingual-e5-small)...[/cyan]")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODELL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    console.print(f"[cyan] Speichere {len(chunks)} Chunks → '{CHROMA_PFAD}'...[/cyan]")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PFAD,
        collection_name=COLLECTION,
    )
    return db


def main():
    parser = argparse.ArgumentParser(
        description="Dokumente in die RAG-Vektordatenbank laden",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python ingest.py --ordner ./dokumente

        """
    )
    parser.add_argument("--ordner", type=str, help="Ordner mit Dokumenten (rekursiv)")
    parser.add_argument("--datei",  type=str, help="Einzelne Datei laden")
    parser.add_argument("--reset",  action="store_true",
                        help="Bestehende Datenbank löschen und neu aufbauen")
    args = parser.parse_args()

    if not args.ordner and not args.datei:
        parser.print_help()
        sys.exit(1)

    console.print(Panel.fit(
        "[bold blue]RAG-System – Dokument-Ingestion[/bold blue]\n"
        "Lokales KI-Dokumentenmanagementsystem · Projektmanagement",
        border_style="blue"
    ))

    # Datenbank zurücksetzen
    if args.reset and Path(CHROMA_PFAD).exists():
        shutil.rmtree(CHROMA_PFAD)
        console.print("[yellow]⚠ Bestehende Datenbank gelöscht.[/yellow]")

    # Dokumente laden
    if args.datei:
        console.print(f"\n[bold]📄 Lade Datei:[/bold] {args.datei}")
        dokumente = lade_dokument(args.datei)
    else:
        console.print(f"\n[bold]📁 Lade Ordner:[/bold] {args.ordner}")
        dokumente = lade_ordner(args.ordner)

    if not dokumente:
        console.print("[red]Keine Dokumente geladen. Abbruch.[/red]")
        sys.exit(1)

    console.print(f"\n✅ [green]{len(dokumente)} Seite(n)/Dokument(e) geladen.[/green]")

    # Chunking
    console.print("\n  [cyan]Zerlege in Chunks (1000 Zeichen, 200 Overlap)...[/cyan]")
    chunks = chunking(dokumente)
    console.print(f"✅ [green]{len(chunks)} Chunks erzeugt[/green]")

    # Speichern
    speichere_in_chromadb(chunks)

    console.print(Panel.fit(
        f"[bold green]✅ Ingestion abgeschlossen![/bold green]\n\n"
        f"• {len(dokumente)} Dokument(e) verarbeitet\n"
        f"• {len(chunks)} Chunks in ChromaDB gespeichert\n"
        f"• Pfad: [cyan]{CHROMA_PFAD}[/cyan]\n\n"
        "Chatbot starten:\n"
        "  [bold]python rag.py[/bold]        ← Llama 3.1 (Qualität)\n"
        "  [bold]python rag_phi.py[/bold]    ← Phi-3.5 (Geschwindigkeit)",
        border_style="green"
    ))


if __name__ == "__main__":
    main()