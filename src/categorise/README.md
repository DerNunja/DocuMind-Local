# Modul A – Lokale Dokumentenkategorisierung

**Projekt:** Lokales KI-gestütztes Dokumentenmanagementsystem  
**Kurs:** Projektmanagement  
**Semester:** 6

---

## Was ist das hier?

Dieses Modul ist für die automatische Kategorisierung von Projektdokumenten zuständig. Ziel ist es, Dokumente lokal auf dem eigenen Rechner einer passenden Ablagekategorie zuzuordnen, ohne Daten an externe Cloud-Dienste zu senden.

Der aktuelle Prototyp arbeitet bewusst review-basiert: Das System schlägt eine Kategorie vor, speichert Begründung und Tags und markiert das Dokument anschließend mit `needs_review`. Eine finale automatische Freigabe findet noch nicht statt.

Technisch kombiniert das Modul ein lokales Sprachmodell über LM Studio mit Embeddings und PostgreSQL + pgvector. Die Vektorsuche findet passende Kandidatenkategorien, danach entscheidet das Sprachmodell anhand des Dokumentprofils und der Kandidaten über die beste Kategorie oder schlägt eine neue Kategorie zur Taxonomieprüfung vor.

---

## Voraussetzungen

- Python 3.11 oder neuer
- `uv` für Abhängigkeiten und Ausführung
- [LM Studio](https://lmstudio.ai/) mit aktiviertem lokalen OpenAI-kompatiblen Server
- Ein lokales Chat-Modell in LM Studio
- Ein lokales Embedding-Modell in LM Studio
- Docker für PostgreSQL + pgvector

Getestete lokale Modelle:

| Zweck | Modell |
|---|---|
| Chat / Klassifikation | `google/gemma-4-e4b` |
| Embeddings | `text-embedding-qwen3-embedding-4b` |

---

## Installation

Die benötigten Pakete wurden mit `uv` installiert:

```bash
uv add requests pydantic typer "psycopg[binary]" pgvector
```

Die CLI kann danach so gestartet werden:

```bash
uv run python -m src.categorise --help
```

---

## PostgreSQL + pgvector starten

Das Modul benötigt PostgreSQL mit installierter pgvector-Erweiterung.

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

Verbindung testen:

```bash
docker exec documind-postgres pg_isready -U postgres -d documind
```

Standard-Datenbank-URL:

```text
postgresql://postgres:postgres@localhost:5432/documind
```

Optional kann eine andere Datenbank gesetzt werden:

```bash
export DOCUMIND_DATABASE_URL="postgresql://user:password@localhost:5432/documind"
```

---

## LM Studio konfigurieren

LM Studio muss mit lokalem Server laufen. Die API wird standardmäßig hier erwartet:

```text
http://127.0.0.1:1234/v1
```

Standardwerte im Modul:

```bash
LM_STUDIO_CHAT_MODEL="google/gemma-4-e4b"
LM_STUDIO_EMBEDDING_MODEL="text-embedding-qwen3-embedding-4b"
LM_STUDIO_BASE_URL="http://127.0.0.1:1234/v1"
```

Falls LM Studio andere Modellnamen anzeigt:

```bash
export LM_STUDIO_CHAT_MODEL="dein-chat-modell"
export LM_STUDIO_EMBEDDING_MODEL="dein-embedding-modell"
export LM_STUDIO_BASE_URL="http://127.0.0.1:1234/v1"
```

Wichtig: Chat-Modell und Embedding-Modell sind zwei verschiedene Modelle.

---

## Dateien in diesem Modul

| Datei | Beschreibung |
|---|---|
| `cli.py` | Kommandozeilenoberfläche mit Typer |
| `service.py` | Kernlogik für Profilbildung, Kandidatensuche und Klassifikation |
| `lm_studio.py` | Client für LM Studio Chat- und Embedding-API |
| `store.py` | PostgreSQL- und pgvector-Persistenz |
| `models.py` | Pydantic-Datenmodelle für Kategorien, Dokumente und Entscheidungen |
| `prompts.py` | Deutsche PM-spezifische Prompts und Seed-Kategorien |
| `vector.py` | Hilfsfunktionen für Fallback-Embeddings |
| `usage.md` | Ausführliche technische Nutzungsdokumentation |
| `critc.md` | Notizen/Korrekturen zur schriftlichen Projektdokumentation |
| `test_docs/` | Testdokumente mit Lösung bzw. ältere Testdaten |

---

## Verwendung – Kommandozeile

### Schritt 1: Kategorien anlegen

Die deutschen PM-spezifischen Seed-Kategorien werden aus `prompts.py` geladen:

```bash
uv run python -m src.categorise seed-categories
```

Aktuelle Kategorien sind unter anderem:

- Eingangsrechnungen
- Verträge und Vereinbarungen
- Angebote und Auftragsbestätigungen
- Schriftverkehr und Korrespondenz
- Reisekostenabrechnungen
- Projektstatusberichte
- Besprechungsprotokolle
- Technische Zeichnungen und Spezifikationen
- Risikoregister und Risikoberichte
- Abnahme- und Prüfprotokolle
- Lieferscheine und Versanddokumente
- Stunden- und Leistungsnachweise

### Schritt 2: Kategorien anzeigen

```bash
uv run python -m src.categorise list-categories
```

### Schritt 3: Einzelnes Dokument kategorisieren

```bash
uv run python -m src.categorise categorise src/categorise/test_docs/dok_01.txt
```

Das Modul erwartet aktuell `.txt`-Dateien mit bereits extrahiertem Text.

### Schritt 4: Ganzen Ordner kategorisieren

```bash
uv run python -m src.categorise categorise-folder src/categorise/test_docs
```

Dabei wird ein Fortschrittsbalken angezeigt mit:

- aktueller Datei
- Prozentfortschritt
- vergangener Zeit
- geschätzter Restzeit

### Schritt 5: Ergebnisse anzeigen

```bash
uv run python -m src.categorise list-documents
```

Die Ausgabe zeigt Dokumentname, Status, vorgeschlagene Kategorie und Kategorie-ID.

### Schritt 6: Taxonomie-Vorschläge anzeigen

Wenn das Modell keine passende bestehende Kategorie findet, wird das Dokument mit `needs_taxonomy_review` gespeichert. Vorschläge für neue Kategorien sieht man mit:

```bash
uv run python -m src.categorise list-taxonomy-reviews
```

---

## Datenbank zurücksetzen

Für saubere Testläufe kann die Datenbank geleert werden:

```bash
docker exec documind-postgres psql -U postgres -d documind \
  -c "TRUNCATE TABLE documents, category_embeddings, categories CASCADE;"
```

Danach Kategorien erneut anlegen:

```bash
uv run python -m src.categorise seed-categories
```

---

## Wie funktioniert die Kategorisierung?

1. Eine `.txt`-Datei wird eingelesen.
2. Das Sprachmodell erzeugt ein strukturiertes Klassifikationsprofil.
3. Dieses Profil wird mit dem lokalen Embedding-Modell vektorisiert.
4. PostgreSQL + pgvector sucht ähnliche Kategorie-Embeddings.
5. Die besten Kandidaten werden zusammen mit dem Dokumentprofil an das Sprachmodell gegeben.
6. Das Modell entscheidet zwischen bestehender Kategorie und `none_fits`.
7. Das Ergebnis wird in PostgreSQL gespeichert.

Die Kategorie-Embeddings liegen in der Tabelle `category_embeddings` als echter pgvector-`vector`-Datentyp. Die Kandidatensuche läuft über Vektordistanz:

```sql
ORDER BY e.embedding <=> query_embedding
```

---

## Datenbanktabellen

Die Tabellen werden automatisch beim ersten Start angelegt:

| Tabelle | Zweck |
|---|---|
| `categories` | Kategorien und vollständige JSONB-Payloads |
| `category_embeddings` | Kategorie-Vektoren für pgvector-Suche |
| `documents` | Verarbeitete Dokumente, Profile und Entscheidungen |

---

## Technische Entscheidungen

**Lokale Modelle:** Alle LLM- und Embedding-Aufrufe laufen über LM Studio lokal auf dem Rechner. Dadurch werden keine Dokumentinhalte an externe Anbieter übertragen.

**Review-first statt Auto-Ablage:** Auch bei hoher Modell-Sicherheit wird nicht automatisch final zugeordnet. Das System speichert Vorschläge mit `needs_review`.

**PostgreSQL + pgvector:** Statt einer separaten Vektordatenbank wird PostgreSQL als zentrale Datenbank genutzt. Das reduziert Synchronisationsprobleme und reicht für den Prototyp aus.

**Deutsche PM-Prompts:** Die Prompts sind auf deutschsprachige Projektdokumente eines mittelständischen Anlagenbauers angepasst.

**Taxonomie-Vorschläge:** Wenn keine Kategorie passt, kann das Modell eine neue Kategorie vorschlagen. Diese wird aber noch nicht automatisch übernommen.

---

## Integration mit den anderen Modulen

### Modul B (OCR/Digitalisierung) → Modul A

Modul B extrahiert Text aus gescannten Dokumenten, PDFs oder Bildern. Modul A erwartet aktuell genau diesen extrahierten Text als `.txt`-Datei oder später als standardisierte Übergabe aus der Pipeline.

### Modul A (Kategorisierung) → Modul C (RAG)

Die erkannte Kategorie kann später als Metadatum an den RAG-Chatbot weitergegeben werden. Dadurch kann Modul C gezielter suchen, zum Beispiel nur in `Besprechungsprotokolle` oder `Projektstatusberichte`.

### Modul D (Übersetzung) → Modul A

Übersetzte oder bereinigte Texte können ebenfalls kategorisiert werden, solange sie als Text an Modul A übergeben werden.

---

## Bekannte Einschränkungen

- Es gibt noch keine grafische Review-Oberfläche.
- Es gibt noch keine CLI-Befehle für `approve`, `change-category` oder `reject`.
- Neue vom LLM vorgeschlagene Kategorien werden noch nicht automatisch als aktive Kategorien übernommen.
- Sehr lange Dokumente werden noch nicht per MapReduce verarbeitet.
- Aktuell werden nur die ersten 12.000 Zeichen eines Dokuments an das Profil-Prompt gesendet.
- Prompt- und Modellversionen werden noch nicht pro Dokument gespeichert.
- Die Qualität hängt deutlich vom lokal geladenen Chat- und Embedding-Modell ab.

---

## Zusammenfassung

Das Kategorisierungsmodul zeigt, dass eine lokale, LLM-gestützte Dokumentenkategorisierung technisch machbar ist. Dokumente werden ohne Cloud-Dienste verarbeitet, über pgvector mit bestehenden Kategorien verglichen und anschließend durch ein lokales Sprachmodell eingeordnet.

Der aktuelle Stand ist ein funktionsfähiger Prototyp für Kategorieempfehlungen mit menschlicher Nachprüfung. Für einen produktiveren Einsatz fehlen vor allem Review-Workflow, Auswertung bestätigter Entscheidungen und bessere Unterstützung für lange Dokumente.
