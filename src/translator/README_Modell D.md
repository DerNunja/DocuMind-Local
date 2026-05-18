# Modul D – Lokales Übersetzungsmodul (Englisch)

**Projekt:** Lokales KI-gestütztes Dokumentenmanagementsystem | **Kurs:** Projektmanagement | **Semester 6**

---

## Was ist das hier?

Dieses Modul ist mein Teil des gemeinsamen Projekts. Die Kernaufgabe besteht darin, das Dokumentenmanagementsystem (DMS) für die Verarbeitung internationaler Projektunterlagen der **Nordharz Anlagenbau GmbH** zu befähigen. Mein Modul nimmt den durch die Digitalisierung (OCR) bereitgestellten Rohtext entgegen, bereinigt ihn und übersetzt den Inhalt präzise und vollständig ins **Englische**.

Das Besondere daran: Um die strikte **DSGVO-Konformität** und die vollständige Offline-Fähigkeit des Unternehmens zu gewährleisten, läuft die gesamte Übersetzungspipeline lokal auf dem eigenen Rechner. Es werden keinerlei Daten an externe Cloud-Dienste (wie DeepL oder OpenAI) gesendet. Dank eines hochoptimierten, spezialisierten neuronalen Übersetzungsmodells läuft das System extrem ressourceneffizient auf Standard-CPUs – eine teure GPU-Infrastruktur ist nicht erforderlich.

---

## Voraussetzungen

- Python 3.10 oder neuer
- Internetverbindung *nur für den ersten Start* (zum automatischen Herunterladen des Modells von Hugging Face in den lokalen Cache)
- Kein GPU nötig, läuft performant auf normaler CPU

---

## Installation

```powershell
# 1. Virtual Environment erstellen und aktivieren
python -m venv venv
venv\Scripts\activate

# 2. Alle benötigten Pakete installieren
pip install -r requirements.txt
```

*Hinweis:* Bei der Installation werden die Bibliotheken `transformers`, `sentencepiece` und `sacremoses` eingerichtet, um die lokale Laufzeitumgebung abzusichern und die Tokenisierung zu stabilisieren.

---

## Dateien in diesem Modul

| Datei | Beschreibung |
| :--- | :--- |
| `translate.py` | Kernskript: Übernimmt den Text, reinigt/segmentiert ihn und führt die lokale Übersetzung ins Englische aus. |
| `app_gesamt.py` | Gemeinsame Web-App für alle Module (Streamlit-Oberfläche). |
| `modul_d_stub.py` | Die Übergangs-Schnittstelle (Stub), die dem Team bereitgestellt wurde, solange das Modul in Entwicklung war. |
| `test_translation.py` | Testskript zur automatisierten Validierung mit realen Fachbegriffen aus dem Anlagenbau. |

---

## Verwendung – Kommandozeile

### Schritt 1: Text direkt ins Englische übersetzen
Du kannst einen Textabschnitt direkt über die Konsole übergeben, um die Übersetzung zu prüfen.
```powershell
python translate.py --text "Projektmanagement ermöglicht eine frühzeitige Machbarkeitsprüfung."
```

### Schritt 2: Validierung mit Testdokumenten ausführen
Um die semantische Präzision des Modells mit echten, komplexen Texten und Verträgen aus dem industriellen Anlagenbau zu prüfen:
```powershell
python test_translation.py --input ./test_docs/projekt_spezifikation.txt
```

---

## Web-App starten

Für die Präsentation und den normalen Betrieb gibt es eine einheitliche Web-Oberfläche:

```powershell
streamlit run app_gesamt.py
```

Der Browser öffnet sich automatisch auf `http://localhost:8501`. 

Im **Tab D (Übersetzungsmodul)** ist mein System voll funktionsfähig eingebettet. Du kannst dort beliebige Texte hineinkopieren oder hochladen, und die Übersetzung ins Englische wird in Echtzeit generiert.

---

## Ergebnisse & Evaluation

Das System wurde im Rahmen der domänenspezifischen Validierung intensiv mit realen technischen Dokumenten und Vorlagen evaluiert:

- **Unterstützte Sprachrichtung:** Reine lokale Inferenz-Pipeline für die Übersetzung von **Deutsch nach Englisch (DE-EN)**.
- **Semantische Präzision:** Das Modell bewies eine durchgehend hohe Treffsicherheit bei der automatisierten Übertragung zusammengesetzter Fachbegriffe ins Englische. Damit ist die fehlerfreie Datenqualität für die anschließende Weiterverarbeitung garantiert.
- **Ressourcenbedarf:**
  - **Modellgröße:** Kompakte ~298 MB für das Übersetzungsmodell.
  - **RAM-Verbrauch:** Extrem geringer Fußabdruck, perfekt optimiert für Standard-Büro-Hardware.
  - **Laufzeit:** Bruchteile von Sekunden pro Textsegment auf einer Standard-CPU.

---

## Technische Entscheidungen

Ein paar Dinge, die ich bewusst so gemacht habe und warum:

**Modellauswahl (`Helsinki-NLP/opus-mt-de-en`):** Ich verwende ein spezialisiertes neuronales Übersetzungsmodell (Marian NMT) via Hugging Face, das exakt auf die Übersetzung von Deutsch nach Englisch trainiert ist. Mit knapp 298 MB ist es winzig, läuft rasend schnell auf CPUs und liefert eine hervorragende Qualität ohne Internetverbindung.

**Vorverarbeitung & Textbereinigung (`utils.py`):** Da aus OCR-Prozessen stammende Texte oft unsaubere Formatierungen, Zerstückelungen oder doppelte Zeilenumbrüche enthalten, durchlaufen die Daten eine Bereinigungspipeline. Der Text wird zudem logisch segmentiert, damit das Modell den Kontext optimal erfasst.

**Standardisiertes Ausgabeformat:** Um eine fehlerfreie Übergabe an nachgelagerte Systeme oder Datenbanken zu garantieren, liefert mein Modul die Daten in einem fest definierten Python-Dictionary zurück:
```python
{
    "original": "Text in der Originalsprache",
    "translated": "Übersetzter Text auf Englisch",
    "language": "en"  # Ziel-Sprachcode
}
```

---

## Integration mit dem Gesamtsystem

Das Modul ist so gebaut, dass es sich nahtlos in die gemeinsame Pipeline einfügt. Es nimmt den digitalisierten Text auf, verarbeitet ihn und gibt das oben gezeigte standardisierte Dictionary an das System weiter. Dadurch können Dokumente auch in englischer Sprache für den Chatbot indiziert werden.

In der zentralen Steuerungsdatei `ingest_pipeline.py` wurde mein echtes Modul anstelle des alten Stubs aktiviert:
```python
# Integration in die zentrale Pipeline
from modul_d import uebersetze  # Aktiviert nach erfolgreicher Modul-Validierung!
```

---

## Bekannte Einschränkungen

- **Stilistische Feinheiten:** Die Qualität ist perfekt für das inhaltliche Fachverständnis. Für rechtlich hochgradig bindende, vertragliche Dokumente wird dennoch eine kurze menschliche Nachbearbeitung empfohlen.
- **Formatierungsverlust bei Tabellen:** Stark verschachtelte Tabellenlayouts werden flach als fortlaufende Textsegmente übersetzt, wodurch die rein visuelle Tabellenstruktur im Zieltext leicht verzerrt werden kann.

---

## Ordnerstruktur

```text
translation/
├── app_gesamt.py          # Gemeinsame Web-App aller Module
├── translate.py           # Hauptskript für Textbereinigung & Marian NMT (DE-EN) Übersetzung
├── test_translation.py    # Validierungsskript für englische Fachbegriffe
├── modul_d_stub.py        # Integrations-Stub für das Team
├── requirements.txt       # Enthält transformers, torch, sentencepiece, sacremoses
└── test_docs/             # Ordner für reale Test-Dokumente der Nordharz GmbH
```
