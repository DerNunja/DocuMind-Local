"""
rag_phi.py – RAG-Chatbot mit Phi-3.5 Mini (Geschwindigkeitsmodus)


Konfiguration :
  Retrieval: MMR, k=2, fetch_k=6, lambda_mult=0.7
  LLM:       num_predict=150, num_thread=8, num_ctx=1024, temperature=0.1
  → ~2x schneller als Llama 3.1, Score ~50%

Verwendung:
    python rag_phi.py
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
# Konfiguration – Speed-Modus (aus AWMF-Projekt)
# ──────────────────────────────────────────────
CHROMA_PFAD      = "./vectordb"
COLLECTION       = "projektdokumente"
EMBEDDING_MODELL = "intfloat/multilingual-e5-small"
OLLAMA_MODELL    = "phi3.5"

# Weniger Chunks → schneller (k=2 statt 4)
K           = 2
FETCH_K     = 6
LAMBDA_MULT = 0.7

# Kürzere Ausgabe → schneller
NUM_PREDICT = 150
NUM_THREAD  = 8
NUM_CTX     = 1024
TEMPERATURE = 0.1


PROMPT_TEMPLATE = """Du bist ein hilfreicher Projektmanagement-Assistent.
Beantworte die Frage kurz und präzise auf Basis der folgenden Dokument-Auszüge.
Die Dokumente können Vorlagen mit Feldnamen oder ausgefüllte Berichte sein.
Beschreibe was die Dokumente zu diesem Thema enthalten – auch Feldnamen und Strukturen sind hilfreiche Informationen.
Erfinde keine konkreten Werte die nicht im Text stehen. Antworte auf Deutsch.

Dokument-Auszüge:
{context}

Frage: {question}

Antwort:"""


def initialisiere_system():
    """Lädt Embeddings, ChromaDB, Ollama und baut die RAG-Chain auf."""

    console.print("[cyan]🔢 Lade Embedding-Modell...[/cyan]")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODELL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    console.print(f"[cyan]📂 Verbinde mit ChromaDB → '{CHROMA_PFAD}'...[/cyan]")
    if not Path(CHROMA_PFAD).exists():
        console.print(
            "[red]Datenbank nicht gefunden.[/red]\n"
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
        console.print("[red]Datenbank ist leer.[/red]\n[bold]  python ingest.py --ordner ./dokumente[/bold]")
        sys.exit(1)
    console.print(f"[green]✅ {anzahl} Chunks in der Datenbank.[/green]")

    # MMR mit weniger Kandidaten (Speed)
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": K,
            "fetch_k": FETCH_K,
            "lambda_mult": LAMBDA_MULT,
        },
    )

    console.print(f"[cyan]⚡ Verbinde mit Ollama ({OLLAMA_MODELL})...[/cyan]")
    try:
        llm = OllamaLLM(
            model=OLLAMA_MODELL,
            temperature=TEMPERATURE,
            num_predict=NUM_PREDICT,
            num_thread=NUM_THREAD,
            num_ctx=NUM_CTX,
        )
        llm.invoke("Hallo")
    except Exception as e:
        console.print(
            f"[red]Ollama nicht erreichbar oder Modell fehlt.[/red]\n"
            f"  1. [bold]ollama serve[/bold]\n"
            f"  2. [bold]ollama pull phi3.5[/bold]"
        )
        sys.exit(1)

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
    tabelle = Table(title="📚 Quellen", box=box.SIMPLE_HEAVY, show_lines=True, style="dim")
    tabelle.add_column("#", width=3)
    tabelle.add_column("Datei", style="cyan", no_wrap=True)
    tabelle.add_column("Auszug", overflow="fold")

    for i, doc in enumerate(quell_docs, 1):
        dateiname = doc.metadata.get("dateiname", "–")
        auszug = doc.page_content.removeprefix("passage: ")[:200].replace("\n", " ").strip() + "…"
        tabelle.add_row(str(i), dateiname, auszug)

    console.print(tabelle)


def chatbot_schleife(chain):
    console.print(Panel.fit(
        "[bold yellow]⚡ Projektmanagement-Assistent bereit![/bold yellow]\n\n"
        "[bold]Modell:[/bold] Phi-3.5 Mini (Speed) · ~2× schneller als Llama\n\n"
        "Befehle: [bold]quellen[/bold] · [bold]quit[/bold]",
        border_style="yellow"
    ))

    zeige_quellen_flag = True
    verlauf = []

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
            console.print(f"[dim]Quellanzeige {'aktiviert' if zeige_quellen_flag else 'deaktiviert'}.[/dim]")
            continue

        kontext_verlauf = ""
        for paar in verlauf[-2:]:
            kontext_verlauf += f"Frage: {paar['frage']}\nAntwort: {paar['antwort']}\n\n"
        anfrage = kontext_verlauf + frage if kontext_verlauf else frage

        console.print("\n[dim]⚡ Suche (MMR, k=2)...[/dim]")
        try:
            ergebnis = chain.invoke({"query": "query: " + anfrage})
        except Exception as e:
            console.print(f"[red]Fehler: {e}[/red]")
            continue

        antwort    = ergebnis.get("result", "Keine Antwort.")
        quell_docs = ergebnis.get("source_documents", [])

        console.print(Panel(
            Markdown(antwort),
            title="[bold yellow]⚡ Assistent (Phi-3.5)[/bold yellow]",
            border_style="yellow",
            padding=(1, 2),
        ))

        if zeige_quellen_flag and quell_docs:
            zeige_quellen(quell_docs)

        verlauf.append({"frage": frage, "antwort": antwort})


def main():
    console.print(Panel.fit(
        "[bold blue]RAG-Dokumentenmanagementsystem[/bold blue]\n"
        "Phi-3.5 Mini · Geschwindigkeitsmodus · lokal & datenschutzkonform",
        border_style="blue"
    ))
    chain = initialisiere_system()
    chatbot_schleife(chain)


if __name__ == "__main__":
    main()