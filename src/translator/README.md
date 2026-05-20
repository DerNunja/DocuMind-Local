# Modul D – Lokales Übersetzungsmodul (DE → EN)

## Projektinformationen

**Projekt:** Lokales KI-gestütztes Dokumentenmanagementsystem  
**Kurs:** Projektmanagement  
**Semester:** 6  

---

# Überblick

Dieses Modul ist verantwortlich für die lokale Übersetzung technischer Dokumente aus dem Deutschen ins Englische innerhalb des gemeinsamen Dokumentenmanagementsystems (DMS).

Das System verarbeitet OCR-generierte Rohtexte, bereinigt typische OCR-Fehler und übersetzt den Inhalt vollständig offline mit einem spezialisierten neuronalen Übersetzungsmodell.

Ziel ist die DSGVO-konforme Verarbeitung sensibler Unternehmensdokumente ohne externe Cloud-Dienste.

---

# Hauptfunktionen

- Vollständig lokale Übersetzung (keine Cloud-APIs)
- OCR-Textbereinigung
- Semantisch präzise Fachübersetzung
- CPU-optimierte Ausführung
- Integration in die gemeinsame Pipeline
- Streamlit-Weboberfläche

---

# Voraussetzungen

- Python 3.10 oder neuer
- Internetverbindung nur beim ersten Modell-Download
- Keine GPU erforderlich

---

# Installation

## 1. Virtuelle Umgebung erstellen

```powershell
python -m venv venv
```

## 2. Umgebung aktivieren

```powershell
venv\Scripts\activate
```

## 3. Abhängigkeiten installieren

```powershell
pip install -r requirements.txt
```

### Hinweis

Die `requirements.txt` installiert gezielt:

- CPU-Version von `torch`
- `transformers`
- `sentencepiece`
- `sacremoses`

Dadurch bleibt das System ressourcenschonend und stabil für lokale Übersetzungen.

---

# Projektstruktur

```plaintext
Translation-modell/
├── model_cache/           # Lokaler Modell-Cache (~298 MB)
├── config.py              # Zentrale Konfiguration
├── models.py              # Singleton-Modellverwaltung
├── utils.py               # OCR-Bereinigung & Segmentierung
├── service.py             # Übersetzungsdienst
├── translate.py           # CLI & Pipeline-Schnittstelle
├── test_translation.py    # Automatisierter Test
├── requirements.txt       # CPU-optimierte Abhängigkeiten
├── modul_d_stub.py        # Team-Integrationsstub
└── app_gesamt.py          # Gemeinsame Streamlit-Web-App
```

---

# Dateien im Detail

| Datei | Beschreibung |
|---|---|
| `config.py` | Modellnamen, Pfade und Parameter |
| `models.py` | Lädt das Modell ressourcenschonend im Singleton-Pattern |
| `utils.py` | OCR-Korrektur und Satzsegmentierung |
| `service.py` | Kernlogik der Übersetzung |
| `translate.py` | CLI-Einstiegspunkt und Pipeline-Schnittstelle |
| `test_translation.py` | Validierung mit technischen Fachbegriffen |
| `app_gesamt.py` | Gemeinsame Streamlit-Web-App |
| `modul_d_stub.py` | Temporäre Pipeline-Schnittstelle |

---

# Verwendung

## Direkte Übersetzung über die Konsole

```powershell
python translate.py --text "Die Montage der Kreiselpumpe muss gemäß der technischen Spezifikation erfolgen."
```

---

## Modul testen

```powershell
python test_translation.py
```

Das Testskript überprüft:

- Übersetzungsqualität
- Fachbegriffserkennung
- Laufzeit
- Stabilität der Pipeline

---

# Streamlit-Web-App starten

```powershell
streamlit run app_gesamt.py
```

Die Anwendung startet anschließend unter:

```plaintext
http://localhost:8501
```

Im Tab **„Modul D – Übersetzung“** kann Text direkt eingegeben oder hochgeladen werden.

---

# Beispielausgabe

Das Modul liefert standardisierte Datenstrukturen zurück:

```python
{
    "original": "Text in der Originalsprache",
    "translated": "Translated English text",
    "language": "en"
}
```

---

# Technische Entscheidungen

## Modellwahl

Verwendetes Modell:

```plaintext
Helsinki-NLP/opus-mt-de-en
```

Vorteile:

- Speziell für Deutsch → Englisch trainiert
- Sehr kompakt (~298 MB)
- Schnell auf CPU
- Offline nutzbar
- Gute Fachbegriffserkennung

---

## OCR-Textbereinigung

Da OCR-Texte häufig:

- Zeilenumbrüche
- Trennfehler
- doppelte Leerzeichen
- zerstückelte Sätze

enthalten, werden die Inhalte vor der Übersetzung automatisch bereinigt.

---

## Singleton-Modellverwaltung

Das Modell wird nur einmal geladen und anschließend wiederverwendet.

Vorteile:

- geringer RAM-Verbrauch
- schnellere Inferenz
- stabile Laufzeit

---

# Evaluation

## Unterstützte Sprachrichtung

```plaintext
Deutsch → Englisch (DE-EN)
```

---

## Fachliche Präzision

Das Modell erkennt technische Begriffe zuverlässig:

| Deutsch | Englisch |
|---|---|
| Inbetriebnahme | commissioning |
| Kreiselpumpe | centrifugal pump |
| Rohrleitungsanlage | piping system |
| Druckbehälter | pressure vessel |

---

## Performance

| Eigenschaft | Wert |
|---|---|
| Modellgröße | ~298 MB |
| RAM-Verbrauch | gering |
| GPU erforderlich | Nein |
| Inferenzzeit | ~0.1–0.2 Sekunden pro Satz |

---

# Integration in die Gesamtpipeline

Nach erfolgreicher Validierung wurde das echte Modul aktiviert:

```python
from modul_d import uebersetze
```

Das Modul verarbeitet OCR-Texte und liefert standardisierte Ergebnisse an die zentrale Pipeline zurück.

---

# Bekannte Einschränkungen

## Stilistische Feinheiten

Für rechtlich bindende Verträge wird weiterhin eine menschliche Nachkontrolle empfohlen.

---

## Tabellenformatierung

Komplexe Tabellenstrukturen können nach der Übersetzung visuell vereinfacht dargestellt werden.

---

# Zusammenfassung

Dieses Modul bietet eine schnelle, DSGVO-konforme und vollständig lokale Übersetzungslösung für technische Dokumente im industriellen Umfeld.

Die Kombination aus:

- OCR-Bereinigung
- CPU-optimierter KI
- lokaler Inferenz
- sauberer Pipeline-Integration

macht das System besonders geeignet für sensible Unternehmensdaten ohne Cloud-Abhängigkeit.
