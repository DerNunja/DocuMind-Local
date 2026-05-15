"""
eval_simple.py – Keyword-basierte Evaluation für den RAG-Chatbot.

Methode: Keyword-Score als Proxy für Faithfulness (lokal, kein API-Key nötig)
         RAGAS wurde getestet → TimeoutError auf CPU 

Verwendung:
    python eval_simple.py              # Llama 3.1 evaluieren
    python eval_simple.py --phi        # Phi-3.5 evaluieren
    python eval_simple.py --ausgabe ergebnisse.json
"""

import argparse
import json
import time
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

console = Console()

# ──────────────────────────────────────────────
# Konfiguration
# ──────────────────────────────────────────────
CHROMA_PFAD      = "./vectordb"
COLLECTION       = "projektdokumente"
EMBEDDING_MODELL = "intfloat/multilingual-e5-small"

# Testfragen – angepasst an tatsächliche Dokumente:
# - Projektmanagement-Vorlagen (Änderungskontrolle, Abschlussbericht, Bewertungsformulare)
# - AWMF/MLOps Projektstatus (Phase 1–6, Llama/Phi Evaluation)
#
# Keywords als Synonym-Gruppen: ["wort1", "wort2"] = ein Treffer wenn EINES vorkommt
TESTFRAGEN = [
    {
        "frage": "Wie hoch ist das Gesamtbudget des Projekts und wie viel wurde bisher verbraucht?",
        "keywords": [
            ["113.000", "113000", "gesamtbudget"],
            ["57.000", "57000", "verbraucht"],
            ["budget", "kosten", "personalkosten"],
            ["50%", "puffer", "verbleibend"],
        ],
    },
    {
        "frage": "Welche technischen Probleme oder Risiken wurden im Projekt identifiziert?",
        "keywords": [
            ["risiko", "datenmigration", "dvs"],
            ["personalausfall", "krankheit", "schneider"],
            ["rot", "gelb", "score"],
            ["maßnahmen", "verantwortlich", "behoben"],
        ],
    },
    {
        "frage": "Wer ist für die Backend-Entwicklung verantwortlich und was wurde zuletzt abgeschlossen?",
        "keywords": [
            ["sarah müller", "sarah", "müller"],
            ["backend", "api", "rest"],
            ["postgresql", "datenbank", "datenbankschema"],
            ["abgeschlossen", "fertiggestellt", "implementiert"],
        ],
    },
    {
        "frage": "Was wurde im letzten Meeting entschieden und welche Aufgaben sind noch offen?",
        "keywords": [
            ["beschluss", "entscheidung", "genehmigt"],
            ["postgresql", "mysql", "datenmigration"],
            ["weber", "müller", "hoffmann"],
            ["fällig", "action", "offen"],
        ],
    },
    {
        "frage": "Bis wann soll das Projekt abgeschlossen sein und warum gibt es Verzögerungen?",
        "keywords": [
            ["29.05.2026", "mai 2026", "projektende"],
            ["verzögerung", "verschoben", "2 wochen"],
            ["krankheit", "krankgeschrieben", "tobias"],
            ["meilenstein", "zeitplan", "sprint"],
        ],
    },
]


def keyword_score(antwort: str, keywords: list) -> float:
    """
    Verbesserter Keyword-Score mit Synonymen.



    Format keywords:
      Einfach:    "budget"           → prüft ob "budget" in Antwort
      Synonyme:   ["budget","kosten"] → prüft ob EINES davon in Antwort (zählt als 1 Treffer)
    """
    antwort_lower = antwort.lower()
    treffer = 0
    for kw in keywords:
        if isinstance(kw, list):
            # Synonym-Gruppe: ein Treffer reicht
            if any(syn.lower() in antwort_lower for syn in kw):
                treffer += 1
        else:
            if kw.lower() in antwort_lower:
                treffer += 1
    return treffer / len(keywords) if keywords else 0.0


def baue_chain(modell: str, k: int, fetch_k: int, num_predict: int, num_ctx: int):
    """Baut die RAG-Chain auf."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODELL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    if not Path(CHROMA_PFAD).exists():
        console.print("[red]Datenbank nicht gefunden.[/red]\n[bold]python ingest.py --ordner ./dokumente[/bold]")
        sys.exit(1)

    db = Chroma(
        persist_directory=CHROMA_PFAD,
        embedding_function=embeddings,
        collection_name=COLLECTION,
    )

    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": 0.7},
    )

    llm = OllamaLLM(
        model=modell,
        temperature=0.1,
        num_predict=num_predict,
        num_thread=8,
        num_ctx=num_ctx,
    )

    prompt = PromptTemplate(
        template="""Du bist ein Projektmanagement-Assistent.
Beantworte die Frage auf Basis der folgenden Dokument-Auszüge.
Die Dokumente können Vorlagen mit Feldnamen oder ausgefüllte Berichte sein.
Beschreibe was die Dokumente zu diesem Thema enthalten – auch Feldnamen, Kategorien oder Strukturen sind hilfreiche Informationen.
Erfinde keine konkreten Werte die nicht im Text stehen.
Antworte auf Deutsch, strukturiert und vollständig.

Dokument-Auszüge:
{context}

Frage: {question}

Antwort:""",
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


def evaluiere(chain, modell_name: str, ausgabe_datei: str | None):
    """Führt alle Testfragen durch und gibt Ergebnisse aus."""
    console.print(f"\n[bold cyan] Starte Evaluation: {modell_name}[/bold cyan]")
    console.print(f"[dim]{len(TESTFRAGEN)} Testfragen · Keyword-Score-Methode[/dim]\n")

    ergebnisse = []

    for i, test in enumerate(TESTFRAGEN, 1):
        console.print(f"[dim]Frage {i}/{len(TESTFRAGEN)}: {test['frage'][:60]}...[/dim]")

        start = time.time()
        try:
            result = chain.invoke({"query": "query: " + test["frage"]})
            antwort = result.get("result", "")
        except Exception as e:
            console.print(f"[red]Fehler: {e}[/red]")
            antwort = ""

        dauer = time.time() - start
        score = keyword_score(antwort, test["keywords"])

        ergebnisse.append({
            "frage":   test["frage"],
            "antwort": antwort,
            "score":   score,
            "zeit_s":  round(dauer, 1),
        })

    # Ergebnistabelle
    tabelle = Table(
        title=f"Evaluation: {modell_name}",
        box=box.HEAVY_EDGE,
        show_lines=True,
    )
    tabelle.add_column("Frage",         style="white",  overflow="fold", max_width=45)
    tabelle.add_column("Score",         style="bold",   justify="center", width=8)
    tabelle.add_column("Zeit (s)",      justify="right", width=10)

    gesamt_score = 0.0
    gesamt_zeit  = 0.0

    for e in ergebnisse:
        score_pct = f"{e['score']*100:.0f}%"
        farbe = "green" if e["score"] >= 0.75 else ("yellow" if e["score"] >= 0.5 else "red")
        tabelle.add_row(
            e["frage"][:45] + ("…" if len(e["frage"]) > 45 else ""),
            f"[{farbe}]{score_pct}[/{farbe}]",
            str(e["zeit_s"]),
        )
        gesamt_score += e["score"]
        gesamt_zeit  += e["zeit_s"]

    avg_score = gesamt_score / len(ergebnisse)
    avg_zeit  = gesamt_zeit  / len(ergebnisse)

    tabelle.add_section()
    tabelle.add_row(
        "[bold]Durchschnitt[/bold]",
        f"[bold]{avg_score*100:.0f}%[/bold]",
        f"[bold]{avg_zeit:.1f}[/bold]",
    )

    console.print(tabelle)
    console.print(Panel.fit(
        f"[bold]Modell:[/bold]        {modell_name}\n"
        f"[bold]Ø Score:[/bold]       {avg_score*100:.0f}%\n"
        f"[bold]Ø Antwortzeit:[/bold] {avg_zeit:.1f}s",
        border_style="blue",
        title="Zusammenfassung"
    ))

    # Optional als JSON speichern
    if ausgabe_datei:
        daten = {
            "modell": modell_name,
            "avg_score_pct": round(avg_score * 100, 1),
            "avg_zeit_s": round(avg_zeit, 1),
            "ergebnisse": ergebnisse,
        }
        with open(ausgabe_datei, "w", encoding="utf-8") as f:
            json.dump(daten, f, ensure_ascii=False, indent=2)
        console.print(f"\n[green]💾 Ergebnisse gespeichert: {ausgabe_datei}[/green]")


def main():
    parser = argparse.ArgumentParser(description="RAG-Evaluation (Keyword-Score)")
    parser.add_argument("--phi",     action="store_true", help="Phi-3.5 evaluieren statt Llama 3.1")
    parser.add_argument("--ausgabe", type=str, default=None, help="Ergebnisse als JSON speichern")
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold blue]RAG-Evaluation[/bold blue]\n"
        "Keyword-Score · lokal · kein API-Key",
        border_style="blue"
    ))

    if args.phi:
        modell_name = "Phi-3.5 Mini (Speed)"
        chain = baue_chain("phi3.5", k=2, fetch_k=6, num_predict=150, num_ctx=1024)
    else:
        modell_name = "Llama 3.1 8B (Qualität)"
        chain = baue_chain("llama3.1:8b", k=4, fetch_k=12, num_predict=250, num_ctx=2048)

    evaluiere(chain, modell_name, args.ausgabe)


if __name__ == "__main__":
    main()