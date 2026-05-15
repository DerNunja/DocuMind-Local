"""
rag.py – RAG-Chatbot mit Llama 3.1 8B (Qualitätsmodus)


Konfiguration (bewährt aus AWMF-Projekt):
  Retrieval: MMR, k=4, fetch_k=12, lambda_mult=0.7
  LLM:       num_predict=250, num_thread=8, num_ctx=2048, temperature=0.1

Verwendung:
    python rag.py
"""

import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.table import Table
from rich import box

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

console = Console()

# ──────────────────────────────────────────────
# Konfiguration (aus AWMF-Projekt übernommen & angepasst)
# ──────────────────────────────────────────────
CHROMA_PFAD      = "./vectordb"
COLLECTION       = "projektdokumente"
EMBEDDING_MODELL = "intfloat/multilingual-e5-small"
OLLAMA_MODELL    = "llama3.1:8b"

# MMR-Parameter (bewährt aus AWMF-Projekt)
K           = 4    # Chunks die ans LLM übergeben werden
FETCH_K     = 12   # Kandidaten für MMR-Auswahl
LAMBDA_MULT = 0.7  # Relevanz vs. Diversität (0=Diversität, 1=Relevanz)

# LLM-Parameter (Laptop-optimiert: num_thread=8 statt 12 – verhindert Überhitzung)
NUM_PREDICT = 250
NUM_THREAD  = 8
NUM_CTX     = 2048
TEMPERATURE = 0.1


# ──────────────────────────────────────────────
# Prompt-Template
# ──────────────────────────────────────────────
PROMPT_TEMPLATE = """Du bist ein hilfreicher Projektmanagement-Assistent.
Beantworte die Frage auf Basis der folgenden Dokument-Auszüge aus dem internen Projektsystem.
Die Dokumente können Vorlagen mit Feldnamen oder ausgefüllte Projektberichte sein.
Beschreibe was die Dokumente zu diesem Thema enthalten – auch Feldnamen, Kategorien und Strukturen sind nützliche Informationen.
Wenn wirklich keine relevante Information vorhanden ist, sage dies klar.
Erfinde keine konkreten Werte oder Namen die nicht im Text stehen.
Antworte immer auf Deutsch, präzise und strukturiert.

Dokument-Auszüge:
{context}

Frage: {question}

Antwort:"""


def initialisiere_system():
    """Lädt Embeddings, ChromaDB, Ollama und baut die RAG-Chain auf."""

    # 1. Embeddings
    console.print("[cyan]🔢 Lade Embedding-Modell (multilingual-e5-small)...[/cyan]")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODELL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 2. ChromaDB
    console.print(f"[cyan]📂 Verbinde mit ChromaDB → '{CHROMA_PFAD}'...[/cyan]")
    if not Path(CHROMA_PFAD).exists():
        console.print(
            "[red]Datenbank nicht gefunden. Bitte zuerst Dokumente einlesen:[/red]\n"
            "[bold]  python ingest.py --ordner ./dokumente[/bold]"
        )
        sys.exit(1)

    db = Chroma(
        persist_directory=CHROMA_PFAD,
        embedding_function=embeddings,
        collection_name=COLLECTION,
    )
    anzahl = db._collection.count()
    if anzahl == 0:
        console.print(
            "[red]Datenbank ist leer. Bitte zuerst Dokumente einlesen:[/red]\n"
            "[bold]  python ingest.py --ordner ./dokumente[/bold]"
        )
        sys.exit(1)
    console.print(f"[green]✅ {anzahl} Chunks in der Datenbank.[/green]")

    # 3. MMR-Retriever (bewährt aus AWMF-Projekt: bessere Diversität als pure Similarity)
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": K,
            "fetch_k": FETCH_K,
            "lambda_mult": LAMBDA_MULT,
        },
    )

    # 4. LLM
    console.print(f"[cyan]🦙 Verbinde mit Ollama ({OLLAMA_MODELL})...[/cyan]")
    try:
        llm = OllamaLLM(
            model=OLLAMA_MODELL,
            temperature=TEMPERATURE,
            num_predict=NUM_PREDICT,
            num_thread=NUM_THREAD,
            num_ctx=NUM_CTX,
        )
        llm.invoke("Hallo")  # Verbindungstest
    except Exception as e:
        console.print(
            f"[red]Ollama nicht erreichbar oder Modell fehlt.[/red]\n"
            f"Bitte sicherstellen:\n"
            f"  1. [bold]ollama serve[/bold]\n"
            f"  2. [bold]ollama pull llama3.1:8b[/bold]"
        )
        sys.exit(1)

    # 5. RAG-Chain
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    return chain


def zeige_quellen(quell_docs: list):
    """Gibt verwendete Quell-Chunks als Tabelle aus."""
    tabelle = Table(
        title=" Verwendete Quellen",
        box=box.SIMPLE_HEAVY,
        show_lines=True,
        style="dim",
    )
    tabelle.add_column("#",      width=3)
    tabelle.add_column("Datei",  style="cyan", no_wrap=True)
    tabelle.add_column("Auszug", overflow="fold")

    for i, doc in enumerate(quell_docs, 1):
        dateiname = doc.metadata.get("dateiname", doc.metadata.get("source", "–"))
        # E5-Prefix aus Anzeige entfernen
        auszug = doc.page_content.removeprefix("passage: ")[:200].replace("\n", " ").strip()
        if len(doc.page_content) > 200:
            auszug += "…"
        tabelle.add_row(str(i), dateiname, auszug)

    console.print(tabelle)


def chatbot_schleife(chain):
    """Hauptschleife mit Chatverlauf (letzte 2 Paare, wie im AWMF-Projekt)."""

    console.print(Panel.fit(
        "[bold green]🤖 Projektmanagement-Assistent bereit![/bold green]\n\n"
        "[bold]Modell:[/bold] Llama 3.1 8B (Qualität) · Score: ~80%\n\n"
        "Befehle:\n"
        "  [bold]quellen[/bold]     → Quellanzeige ein/ausschalten\n"
        "  [bold]quit[/bold]        → Beenden",
        border_style="green"
    ))

    zeige_quellen_flag = True
    verlauf = []  # Format: [{"frage": ..., "antwort": ...}, ...]

    while True:
        try:
            frage = Prompt.ask("\n[bold blue]Du[/bold blue]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Auf Wiedersehen![/yellow]")
            break

        if not frage:
            continue

        if frage.lower() in ("quit", "exit", "beenden"):
            console.print("[yellow]Auf Wiedersehen![/yellow]")
            break

        if frage.lower() == "quellen":
            zeige_quellen_flag = not zeige_quellen_flag
            status = "aktiviert" if zeige_quellen_flag else "deaktiviert"
            console.print(f"[dim]Quellanzeige {status}.[/dim]")
            continue

        # Chatverlauf (letzte 2 Paare) in die Frage einbetten
        kontext_verlauf = ""
        for paar in verlauf[-2:]:
            kontext_verlauf += f"Frage: {paar['frage']}\nAntwort: {paar['antwort']}\n\n"

        anfrage = kontext_verlauf + frage if kontext_verlauf else frage

        # E5-Query-Prefix für die Suche (wichtig für multilingual-e5-small)
        console.print("\n[dim]🔍 Suche relevante Dokumente (MMR)...[/dim]")
        try:
            ergebnis = chain.invoke({"query": "query: " + anfrage})
        except Exception as e:
            console.print(f"[red]Fehler: {e}[/red]")
            continue

        antwort       = ergebnis.get("result", "Keine Antwort erhalten.")
        quell_docs    = ergebnis.get("source_documents", [])

        console.print(Panel(
            Markdown(antwort),
            title="[bold green]🤖 Assistent (Llama 3.1)[/bold green]",
            border_style="green",
            padding=(1, 2),
        ))

        if zeige_quellen_flag and quell_docs:
            zeige_quellen(quell_docs)

        verlauf.append({"frage": frage, "antwort": antwort})


def main():
    console.print(Panel.fit(
        "[bold blue]RAG-Dokumentenmanagementsystem[/bold blue]\n"
        "Llama 3.1 8B · Qualitätsmodus · lokal & datenschutzkonform",
        border_style="blue"
    ))

    chain = initialisiere_system()
    chatbot_schleife(chain)


if __name__ == "__main__":
    main()