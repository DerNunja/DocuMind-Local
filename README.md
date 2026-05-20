# DocuMind Local – Lokales KI-gestütztes Dokumentenmanagementsystem

**Kurs:** Projektmanagement  
**Semester:** 6  
**Ziel:** Lokaler Prototyp für ein datenschutzfreundliches Dokumentenmanagementsystem mit KI-Unterstützung

---

## Was ist DocuMind Local?

DocuMind Local ist ein gemeinsames Projekt zur prototypischen Entwicklung eines lokal betriebenen, KI-gestützten Dokumentenmanagementsystems. Der Fokus liegt darauf, typische Aufgaben der Dokumentenverwaltung im Projektmanagement zu unterstützen, ohne sensible Unternehmensdaten an externe Cloud-Dienste zu übertragen.

Das System besteht aus mehreren Funktionsmodulen:

- OCR/Digitalisierung gescannter Dokumente
- automatische Dokumentenkategorisierung
- RAG-basierter Chatbot für Fragen an Dokumente
- lokale Übersetzung technischer Texte

Alle Module verfolgen denselben Grundgedanken: Dokumente sollen lokal verarbeitet, strukturiert, auffindbar und besser nutzbar gemacht werden.

---

## Projektkontext

Die Fallstudie orientiert sich an einem mittelständischen Unternehmen mit sensiblen Projektdokumenten und begrenzter IT-Infrastruktur. Cloud-basierte KI-Dienste sind in diesem Szenario problematisch, weil Dokumente Verträge, technische Spezifikationen, Statusberichte, Risikoinformationen oder personenbezogene Daten enthalten können.

Leitfrage des Projekts:

> Wie kann ein lokal betriebenes, KI-gestütztes Dokumentenmanagementsystem die Effizienz der Dokumentenverwaltung im Projektmanagement steigern – unter Berücksichtigung von Datenschutz und Kosteneffizienz?

---

## Module

| Modul | Pfad | Zweck |
|---|---|---|
| Modul A – Kategorisierung | `src/categorise/` | Ordnet Dokumente lokalen Kategorien zu und schlägt neue Kategorien vor |
| Modul B – OCR/Digitalisierung | `src/OCR/` | Extrahiert Text aus gescannten oder bildbasierten Dokumenten |
| Modul C – RAG-Chatbot | `src/RAGBot/` | Beantwortet Fragen auf Basis eingelesener Projektdokumente |
| Modul D – Übersetzung | `src/translator/` | Übersetzt technische Dokumenttexte lokal von Deutsch nach Englisch |

---

## Modul A – Kategorisierung

Das Kategorisierungsmodul klassifiziert bereits extrahierte Textdokumente in stabile geschäftliche Kategorien. Es nutzt lokale Modelle über LM Studio, PostgreSQL und pgvector.

Aktuelle Funktionen:

- deutsche PM-spezifische Prompts
- Seed-Kategorien für ein Anlagenbau-/Projektmanagement-Szenario
- pgvector-basierte Ähnlichkeitssuche
- Kategorieempfehlungen mit Begründung
- `needs_review` statt automatischer finaler Ablage
- Vorschläge für neue Kategorien bei `needs_taxonomy_review`
- CLI mit Fortschrittsanzeige

Weitere Details:

```text
src/categorise/README.md
src/categorise/usage.md
```

---

## Modul B – OCR/Digitalisierung

Das OCR-Modul ist dafür vorgesehen, Papierdokumente, Scans und bildbasierte PDFs in nutzbaren Text umzuwandeln. Dieser Text kann anschließend von den anderen Modulen weiterverarbeitet werden.

Rolle in der Gesamtpipeline:

1. Dokument wird digitalisiert oder eingelesen.
2. OCR extrahiert Text.
3. Der Text wird an Kategorisierung, RAG oder Übersetzung weitergegeben.

---

## Modul C – RAG-Chatbot

Der RAG-Chatbot ermöglicht Fragen in natürlicher Sprache an den Dokumentenbestand. Dokumente werden in Chunks zerlegt, als Vektoren gespeichert und bei einer Frage semantisch durchsucht.

Beispiele:

- „Was wurde im letzten Meeting entschieden?“
- „Welche Risiken wurden im Projektstatusbericht genannt?“
- „Wer ist für die Inbetriebnahme verantwortlich?“

Weitere Details:

```text
src/RAGBot/README.md
```

---

## Modul D – Übersetzung

Das Übersetzungsmodul übersetzt technische deutsche Texte lokal ins Englische. Es nutzt ein spezialisiertes neuronales Übersetzungsmodell und bereinigt OCR-Texte vor der Übersetzung.

Weitere Details:

```text
src/translator/README.md
```

---

## Lokale Ausführung

Das Projekt nutzt Python und `uv` für die Paketverwaltung.

Abhängigkeiten installieren:

```bash
uv sync
```

Falls neue Pakete benötigt werden:

```bash
uv add paketname
```

Ein Modul ausführen, Beispiel Kategorisierung:

```bash
uv run python -m src.categorise --help
```

---

## Voraussetzungen

Allgemein:

- Python 3.11 oder neuer
- `uv`
- Docker, falls PostgreSQL/pgvector genutzt wird
- lokale KI-Laufzeit je nach Modul, z. B. LM Studio oder Ollama

Für Modul A:

- LM Studio mit lokalem OpenAI-kompatiblen Server
- lokales Chat-Modell
- lokales Embedding-Modell
- PostgreSQL mit pgvector

Für Modul C:

- Ollama
- lokale LLM- und Embedding-Modelle

Für Modul D:

- lokales Übersetzungsmodell über `transformers`

---

## PostgreSQL + pgvector für Modul A

Für die Kategorisierung wird PostgreSQL mit pgvector verwendet.

Container starten:

```bash
docker run --name documind-postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=documind \
  -p 5432:5432 \
  -d pgvector/pgvector:pg16
```

Falls der Container bereits existiert:

```bash
docker start documind-postgres
```

Kategorien anlegen:

```bash
uv run python -m src.categorise seed-categories
```

Dokumente kategorisieren:

```bash
uv run python -m src.categorise categorise-folder src/categorise/test_docs
```

---

## Projektstruktur

```text
DocuMind Local/
├── README.md
├── pyproject.toml
├── uv.lock
├── Ruff_Plan.md
└── src/
    ├── categorise/
    │   ├── README.md
    │   ├── usage.md
    │   ├── cli.py
    │   ├── service.py
    │   ├── lm_studio.py
    │   ├── store.py
    │   ├── models.py
    │   ├── prompts.py
    │   └── test_docs/
    ├── OCR/
    ├── RAGBot/
    │   └── README.md
    └── translator/
        └── README.md
```

---

## Beispielhafter Gesamtprozess

Ein möglicher Ablauf im späteren Gesamtsystem:

1. Ein Dokument wird hochgeladen oder eingescannt.
2. Modul B extrahiert den Text per OCR oder Parser.
3. Modul A schlägt eine Kategorie vor.
4. Modul D übersetzt den Text bei Bedarf.
5. Modul C indexiert den Text und macht ihn per Chatbot durchsuchbar.
6. Der Nutzer kann Dokumente finden, Fragen stellen oder Ergebnisse prüfen.

---

## Datenschutz und lokale Verarbeitung

Ein zentrales Projektziel ist die Verarbeitung sensibler Dokumente ohne Cloud-Abhängigkeit.

Umgesetzt bzw. vorgesehen:

- lokale LLM-Inferenz
- lokale Embeddings
- lokale Datenbanken
- keine externen KI-APIs für Dokumentinhalte
- menschliche Prüfung kritischer Klassifikationen

---

## Aktueller Entwicklungsstand

Der aktuelle Stand ist ein funktionierender Prototyp mit getrennt entwickelten Feature-Modulen.

Besonders weit umgesetzt ist die Kategorisierung mit:

- CLI
- LM Studio Integration
- PostgreSQL + pgvector
- deutschen Prompts
- Testdokumenten
- Taxonomie-Vorschlägen

Andere Module wurden parallel entwickelt und sind in ihren jeweiligen Unterordnern dokumentiert.

---

## Bekannte Einschränkungen

- Die Module sind noch nicht vollständig zu einer gemeinsamen End-to-End-Pipeline verbunden.
- Es gibt noch keine einheitliche zentrale Weboberfläche für alle finalen Module.
- Review-Workflows sind teilweise nur als Konzept oder CLI-Prototyp vorhanden.
- Die Qualität hängt stark von den lokal verfügbaren Modellen und der Hardware ab.
- Große Dokumente benötigen noch bessere Chunking- bzw. MapReduce-Strategien.
- Für produktiven Einsatz fehlen Authentifizierung, Rechteverwaltung, Logging und robuste Fehlerbehandlung.

---

## Nächste Schritte

Sinnvolle nächste Entwicklungsschritte:

1. Module über eine gemeinsame Pipeline verbinden.
2. Einheitliche Datenstruktur für Dokumente zwischen OCR, Kategorisierung, RAG und Übersetzung definieren.
3. Review-Workflow für Kategorievorschläge umsetzen.
4. Weboberfläche für Upload, Review, Suche und Übersetzung integrieren.
5. Evaluationsmetriken für Klassifikation und RAG-Antwortqualität ergänzen.
6. Testdatensatz erweitern und Ergebnisse systematisch auswerten.

---

## Zusammenfassung

DocuMind Local zeigt, dass ein lokal betriebenes KI-Dokumentenmanagementsystem technisch machbar ist. Die Kombination aus OCR, Kategorisierung, RAG und Übersetzung kann die Dokumentenverwaltung im Projektmanagement unterstützen, ohne sensible Inhalte an externe Cloud-Dienste zu übertragen.

Der Prototyp ist noch kein produktives DMS, bildet aber eine belastbare Grundlage für weitere Integration, Evaluation und Ausbau.
