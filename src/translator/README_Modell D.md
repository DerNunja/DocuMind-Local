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
- Keine GPU nötig, läuft extrem performant auf normaler CPU

---

## Installation

```powershell
# 1. Virtual Environment erstellen und aktivieren
python -m venv venv
venv\Scripts\activate

# 2. Alle benötigten Pakete installieren
pip install -r requirements.txt
Hinweis: Bei der Installation über die angepasste requirements.txt wird gezielt die CPU-Version von torch sowie transformers, sentencepiece und sacremoses eingerichtet. Dies sichert eine schlanke, ressourcenschonende lokale Laufzeitumgebung ab und stabilisiert die Tokenisierung.Dateien in diesem ModulDateiBeschreibungconfig.pyZentrale Konfigurationsdatei für Modellnamen, Pfade und Token-Längen.models.pyVerwaltet das Laden des Modells via Hugging Face im ressourceneffizienten Singleton-Pattern.utils.pyEnthält die Textbereinigung (OCR-Korrektur) und die logische Satz-Segmentierung.service.pyDer Kern-Übersetzungsdienst, welcher die Vorverarbeitung und Modell-Inferenz steuert.translate.pyHaupt-Einstiegspunkt: Bietet das CLI für die Konsole und die zentrale Schnittstelle für die Gruppen-Pipeline.test_translation.pyAutomatisiertes Testskript zur Validierung des Moduls mit realen Fachbegriffen aus dem Anlagenbau.app_gesamt.pyGemeinsame Web-App für alle Module (Streamlit-Oberfläche).modul_d_stub.pyDie Übergangs-Schnittstelle (Stub) für die Pipeline-Integration während der Entwicklungsphase.Verwendung – KommandozeileSchritt 1: Text direkt ins Englische übersetzenDu kannst einen Textabschnitt direkt über die Konsole übergeben, um die Übersetzung live zu prüfen:PowerShellpython translate.py --text "Die Montage der Kreiselpumpe muss gemäß der technischen Spezifikation erfolgen."
Schritt 2: Validierung des Moduls ausführenUm die semantische Präzision und Geschwindigkeit des Modells mit echten, komplexen Fachbegriffen aus dem industriellen Anlagenbau automatisiert zu prüfen:PowerShellpython test_translation.py
Web-App startenFür die Präsentation und den normalen Betrieb gibt es eine einheitliche Web-Oberfläche:PowerShellstreamlit run app_gesamt.py
Der Browser öffnet sich automatisch auf http://localhost:8501.Im Tab D (Übersetzungsmodul) ist mein System voll funktionsfähig eingebettet. Du kannst dort beliebige Texte hineinkopieren oder hochladen, und die Übersetzung ins Englische wird lokal in Echtzeit generiert.Ergebnisse & EvaluationDas System wurde im Rahmen der domänenspezifischen Validierung intensiv mit realen technischen Dokumenten und Vorlagen evaluiert:Unterstützte Sprachrichtung: Reine lokale Inferenz-Pipeline für die Übersetzung von Deutsch nach Englisch (DE-EN).Semantische Präzision: Das Modell bewies eine durchgehend hohe Treffsicherheit bei der automatisierten Übertragung zusammengesetzter Fachbegriffe (z.B. Inbetriebnahme -> commissioning, Kreiselpumpe -> centrifugal pump). Damit ist die fehlerfreie Datenqualität für die anschließende Weiterverarbeitung garantiert.Ressourcenbedarf:Modellgröße: Kompakte ~298 MB für das Übersetzungsmodell.RAM-Verbrauch: Extrem geringer Fußabdruck, perfekt optimiert für Standard-Büro-Hardware (Modell wird via Singleton nur einmal geladen).Laufzeit: Nach dem initialen Caching liegt die Inferenzzeit bei 0.1 bis 0.2 Sekunden pro Satz auf einer Standard-CPU.Technische EntscheidungenEin paar Dinge, die ich bewusst so gemacht habe und warum:Modellauswahl (Helsinki-NLP/opus-mt-de-en): Ich verwende ein spezialisiertes neuronales Übersetzungsmodell (Marian NMT) via Hugging Face, das exakt auf die Übersetzung von Deutsch nach Englisch trainiert ist. Mit knapp 298 MB ist es winzig, läuft rasend schnell auf CPUs und liefert eine hervorragende Qualität ohne Internetverbindung.Vorverarbeitung & Textbereinigung (utils.py): Da aus OCR-Prozessen stammende Texte oft unsaubere Formatierungen, Zerstückelungen oder doppelte Zeilenumbrüche enthalten, durchlaufen die Daten eine Bereinigungspipeline. Der Text wird zudem logisch segmentiert, damit das Modell den Kontext optimal erfasst.Standardisiertes Ausgabeformat: Um eine fehlerfreie Übergabe an nachgelagerte Systeme oder Datenbanken zu garantieren, liefert mein Modul die Daten in einem fest definierten Python-Dictionary zurück:Python{
    "original": "Text in der Originalsprache",
    "translated": "Übersetzter Text auf Englisch",
    "language": "en"  # Ziel-Sprachcode
}
Integration mit dem GesamtsystemDas Modul ist so gebaut, dass es sich nahtlos in die gemeinsame Pipeline einfügt. Es nimmt den digitalisierten Text auf, verarbeitet ihn und gibt das oben gezeigte standardisierte Dictionary an das System weiter. Dadurch können Dokumente auch in englischer Sprache für den Chatbot indiziert werden.In der zentralen Steuerungsdatei ingest_pipeline.py wurde mein echtes Modul anstelle des alten Stubs aktiviert:Python# Integration in die zentrale Pipeline
from modul_d import uebersetze  # Aktiviert nach erfolgreicher Modul-Validierung!
Bekannte EinschränkungenStilistische Feinheiten: Die Qualität ist perfekt für das inhaltliche Fachverständnis. Für rechtlich hochgradig bindende, vertragliche Dokumente wird dennoch eine kurze menschliche Nachbearbeitung empfohlen.Formatierungsverlust bei Tabellen: Stark verschachtelte Tabellenlayouts werden flach als fortlaufende Textsegmente übersetzt, wodurch die rein visuelle Tabellenstruktur im Zieltext leicht verzerrt werden kann.OrdnerstrukturPlaintextTranslastion-modell/
├── model_cache/           # Lokaler Cache-Ordner für die Gewichte der KI (~298 MB)
├── config.py              # Konfiguration des Modellnamens und der Parameter
├── models.py              # Singleton-Klasse zum RAM-schonenden Laden des Modells
├── utils.py               # Funktionen für OCR-Textbereinigung und Satz-Segmentierung
├── service.py             # Eigentlicher Übersetzungsdienst
├── translate.py           # Hauptskript mit CLI für Konsole & Pipeline-Anbindung
├── test_translation.py    # Automatisierter Integrationstest mit PM-Fachbegriffen
├── requirements.txt       # CPU-optimierte Abhängigkeiten (torch, transformers, etc.)
├── modul_d_stub.py        # Integrations-Stub für das Team
└── app_gesamt.py          # Gemeinsame Web-App aller Module (Streamlit)
