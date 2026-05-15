# Modul C – RAG-basierter Chatbot
**Projekt: Lokales KI-gestütztes Dokumentenmanagementsystem**
**Kurs: Projektmanagement | Semester 6**

---

## Was ist das hier?

Dieses Modul ist mein Teil des gemeinsamen Projekts. Die Idee dahinter: Man soll Fragen in normaler Sprache an Projektdokumente stellen können – also zum Beispiel *"Was wurde im letzten Meeting entschieden?"* oder *"Wer ist für die Backend-Entwicklung zuständig?"* – und eine sinnvolle Antwort bekommen. Das alles läuft lokal auf dem eigenen Rechner, ohne dass irgendwelche Daten in die Cloud geschickt werden.

Technisch gesehen ist das ein RAG-System (Retrieval-Augmented Generation): Dokumente werden in kleine Stücke aufgeteilt, als Vektoren gespeichert, und wenn eine Frage kommt, werden die relevantesten Textstellen rausgesucht und an ein lokales Sprachmodell weitergegeben, das dann die Antwort formuliert.

---

## Voraussetzungen

- Python 3.10 oder neuer
- [Ollama](https://ollama.com) – lokal installiert und gestartet
- Die Modelle müssen vorher heruntergeladen werden (siehe unten)
- Kein GPU nötig, läuft auf normaler CPU

---

## Installation

```powershell
# 1. Virtual Environment erstellen und aktivieren
python -m venv venv
venv\Scripts\activate

# 2. Alle Pakete installieren
pip install -r requirements.txt

# 3. Ollama-Modelle herunterladen (einmalig, dauert etwas)
ollama pull llama3.1:8b    # ca. 4.9 GB – für gute Antwortqualität
ollama pull phi3.5          # ca. 2.2 GB – für schnelle Antworten

# 4. Ollama starten (muss im Hintergrund laufen)
ollama serve
```

---

## Dateien in diesem Modul

| Datei | Beschreibung |
|---|---|
| `ingest.py` | Liest Dokumente ein und speichert sie in ChromaDB |
| `rag.py` | Chatbot mit Llama 3.1 (bessere Qualität) |
| `rag_phi.py` | Chatbot mit Phi-3.5 (schneller) |
| `eval_simple.py` | Bewertet die Antwortqualität automatisch |
| `app_gesamt.py` | Gemeinsame Web-App für alle 4 Module |
| `modul_b_stub.py` | Überbrückt Modul B solange es noch nicht fertig ist |
| `modul_d_stub.py` | Überbrückt Modul D solange es noch nicht fertig ist |
| `ingest_pipeline.py` | Verbindet alle Module in einer gemeinsamen Pipeline |

---

## Verwendung – Kommandozeile

### Schritt 1: Dokumente einlesen

```powershell
python ingest.py --ordner "C:\Pfad\zu\deinen\dokumenten"

# Einzelne Datei
python ingest.py --datei protokoll.pdf

# Alles neu aufbauen
python ingest.py --ordner ./dokumente --reset
```

Unterstützte Formate: PDF, DOCX, TXT. Bei PDFs werden Tabellen automatisch erkannt und als lesbarer Text gespeichert.

### Schritt 2: Chatbot starten

```powershell
# Llama 3.1 – bessere Antworten, ca. 90 Sekunden pro Frage
python rag.py

# Phi-3.5 – schneller (~40s), für einfachere Fragen ausreichend
python rag_phi.py
```

Im Chat gibt es zwei Befehle: `quellen` schaltet die Quellanzeige ein/aus, `quit` beendet den Chatbot.

### Schritt 3: Evaluation

```powershell
python eval_simple.py           # Llama 3.1 testen
python eval_simple.py --phi     # Phi-3.5 testen

# Ergebnisse als JSON speichern
python eval_simple.py --ausgabe ergebnisse_llama.json
python eval_simple.py --phi --ausgabe ergebnisse_phi.json
```

---

## Web-App starten

Für die Präsentation und den normalen Betrieb gibt es eine Web-Oberfläche mit allen 4 Modulen:

```powershell
streamlit run app_gesamt.py
```

Der Browser öffnet sich automatisch auf `http://localhost:8501`.

Die App hat vier Tabs – einen für jedes Modul. Tab C (RAG-Chatbot) ist voll funktionsfähig. Die anderen Tabs laufen im Stub-Modus solange die Kollegen ihre Module noch nicht fertiggestellt haben. Sobald die echten Module verfügbar sind, reicht es eine Zeile in `ingest_pipeline.py` zu ändern (siehe Abschnitt Integration weiter unten).

In der Sidebar sieht man den Status aller Module, wie viele Chunks in der Datenbank sind, und kann das Modell wechseln.

---

## Ergebnisse

Getestet mit 8 Projektdokumenten (Statusberichte, Meeting-Protokolle, Risikoregister, Änderungsanträge, PM-Vorlagen):

| Modell | Ø Score | Ø Antwortzeit | RAM |
|---|---|---|---|
| Llama 3.1 8B | 80% | 90.3s | ~4.9 GB |
| Phi-3.5 Mini | 85% | 48.0s | ~2.2 GB |

Phi-3.5 ist bei kurzen, strukturierten PM-Dokumenten überraschenderweise besser als Llama 3.1 und gleichzeitig fast doppelt so schnell. Llama 3.1 zeigt seinen Vorteil eher bei langen, komplexen Dokumenten.

Die Bewertung läuft über einen Keyword-Score – die Antwort wird auf thematisch relevante Wörter geprüft. RAGAS (eine bessere Evaluierungsmethode) hatte auf CPU einen Timeout und war deshalb nicht nutzbar.

---

## Technische Entscheidungen

Ein paar Dinge die ich bewusst so gemacht habe und warum:

**Embedding-Modell:** Ich verwende `intfloat/multilingual-e5-small` statt dem häufig verwendeten MiniLM. Das Modell ist speziell für Retrieval-Aufgaben trainiert und kommt mit Deutsch gut zurecht. Der E5-Prefix (`passage:` für Dokumente, `query:` für Suchanfragen) ist dabei wichtig – ohne ihn wird die Qualität deutlich schlechter.

**Chunking 1000/200:** Größere Chunks halten zusammenhängende Informationen (z.B. eine Entscheidung über mehrere Sätze) zusammen. Die 200 Zeichen Überlappung stellen sicher, dass nichts an den Grenzen verloren geht.

**MMR statt einfacher Suche:** Bei der normalen Ähnlichkeitssuche kann es passieren, dass alle zurückgegebenen Chunks aus dem gleichen Absatz stammen und das Modell dreimal denselben Text bekommt. MMR wählt stattdessen Chunks aus, die relevant *und* verschieden voneinander sind.

**Zwei Modelle:** Llama 3.1 für Qualität, Phi-3.5 für Geschwindigkeit. Beides über Ollama, beides läuft komplett lokal ohne Internet.

---

## Integration mit den anderen Modulen

Das System ist so gebaut, dass es mit den anderen Teilen des Projekts zusammenarbeitet – aber auch ohne sie funktioniert.

### Modul B (OCR/Digitalisierung) → Modul C

Modul B extrahiert Text aus gescannten Dokumenten und liefert diesen zusammen mit einem Dokument-Key. Mein Modul nimmt diesen Text direkt entgegen und spielt ihn in die Datenbank ein.

```python
# ingest_pipeline.py – Zeile 22 ändern wenn Modul B fertig ist:
from modul_b_stub import extrahiere_dokument   # aktuell
# from modul_b import extrahiere_dokument      # wenn fertig
```

Solange Modul B nicht fertig ist, übernimmt `modul_b_stub.py` diese Aufgabe – es liest PDFs direkt ein und gibt denselben Output zurück den Modul B liefern würde.

### Modul A (Kategorisierung) → Modul C

Modul A weist jedem Dokument eine Kategorie zu (z.B. "Protokoll", "Rechnung"). Diese Kategorie wird als Metadatum in ChromaDB gespeichert und kann beim Abrufen als Filter verwendet werden:

```python
# Nur Protokolle durchsuchen
retriever = db.as_retriever(
    search_kwargs={"filter": {"kategorie": "Protokoll"}}
)
```

Das macht den Chatbot präziser – eine Frage nach Meeting-Entscheidungen sucht dann nur in Protokollen, nicht in Rechnungen.

### Modul D (Übersetzer) → Modul C

Modul D übersetzt Dokumente in andere Sprachen. Die übersetzte Version wird ebenfalls in ChromaDB eingespielt (mit Metadatum `sprache: "de"`), sodass der Chatbot auch auf englischsprachige Originaldokumente auf Deutsch antworten kann.

```python
# ingest_pipeline.py – Zeile 23 ändern wenn Modul D fertig ist:
from modul_d_stub import uebersetze   # aktuell
# from modul_d import uebersetze      # wenn fertig
```

### Zusammenfassung

Der einzige Ort wo etwas geändert werden muss wenn die anderen Module fertig sind, sind **zwei Zeilen** in `ingest_pipeline.py`. Der Rest des Codes bleibt unverändert.

---

## Bekannte Einschränkungen

- Antwortzeiten von 40–90 Sekunden pro Frage – das liegt am Modell auf CPU ohne GPU, daran lässt sich ohne Hardware-Upgrade wenig ändern
- Bilder und Diagramme in PDFs werden nicht erfasst, nur Text
- Bei sehr langen Tabellen kann es vorkommen, dass Spalten beim Chunking getrennt werden
- RAGAS als Evaluierungsmethode funktioniert nicht auf CPU (Timeout)
- Gemma 4 E4B wurde getestet, ist aber aufgrund eines LangChain-Bugs und zu hohem RAM-Bedarf (9.6 GB) nicht einsetzbar

---

## Ordnerstruktur

```
rag/
├── app_gesamt.py          # Web-App für alle 4 Module
├── app.py                 # Web-App nur für Modul C
├── ingest.py              # Dokumente einlesen
├── rag.py                 # Chatbot Llama 3.1
├── rag_phi.py             # Chatbot Phi-3.5
├── eval_simple.py         # Evaluation
├── ingest_pipeline.py     # Integration aller Module
├── modul_b_stub.py        # Ersatz für Modul B
├── modul_d_stub.py        # Ersatz für Modul D
├── requirements.txt
├── PROJEKTDOKUMENTATION.md
├── vectordb/              # ChromaDB (wird automatisch erstellt)
└── venv/
```
