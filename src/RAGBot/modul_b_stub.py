"""
modul_b_stub.py – Ersatz für Modul B (OCR-Digitalisierung)

Solange Modul B nicht fertig ist, übernimmt diese Datei dessen Aufgabe:
  - Liest PDFs/DOCX/TXT direkt mit PyPDF/Docx2txt
  - Gibt exakt das gleiche Format zurück, das Modul B liefern würde:
    {"text": "...", "dokument_key": "...", "dateiname": "..."}

Wenn Modul B fertiggestellt wird, einfach den Import in ingest_pipeline.py
von `modul_b_stub` auf `modul_b` ändern – der Rest bleibt identisch.
"""

import hashlib
import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader


def _generiere_key(dateiname: str) -> str:
    """
    Erzeugt einen reproduzierbaren Dokument-Key aus dem Dateinamen.
    Gleiche Logik wie Modul B verwenden, sobald bekannt.
    Format: DOC-{8 Zeichen Hash}
    """
    hash_wert = hashlib.md5(dateiname.encode()).hexdigest()[:8].upper()
    return f"DOC-{hash_wert}"


def _bereinige_text(text: str) -> str:
    """Minimale Bereinigung – analog zu dem, was Modul B liefern würde."""
    text = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", text)   # Literaturverweise
    text = re.sub(r"\n{3,}", "\n\n", text)              # Mehrfache Leerzeilen
    return text.strip()


def extrahiere_dokument(dateipfad: str) -> dict:
    """
    Simuliert die Ausgabe von Modul B für eine einzelne Datei.

    Rückgabe (identisch mit Modul B Interface):
        {
            "text":          str,   # extrahierter Reintext
            "dokument_key":  str,   # eindeutiger Identifier
            "dateiname":     str,   # ursprünglicher Dateiname
            "dateipfad":     str,   # vollständiger Pfad
            "seiten":        int,   # Anzahl Seiten (0 bei TXT/DOCX)
        }

    Gibt None zurück bei nicht unterstütztem Format oder Fehler.
    """
    pfad = Path(dateipfad)
    if not pfad.exists():
        print(f"[modul_b_stub] Datei nicht gefunden: {dateipfad}")
        return None

    endung = pfad.suffix.lower()

    try:
        if endung == ".pdf":
            loader = PyPDFLoader(str(pfad))
            docs = loader.load()
            text = "\n\n".join(d.page_content for d in docs)
            seiten = len(docs)
        elif endung in (".docx", ".doc"):
            loader = Docx2txtLoader(str(pfad))
            docs = loader.load()
            text = docs[0].page_content if docs else ""
            seiten = 0
        elif endung == ".txt":
            loader = TextLoader(str(pfad), encoding="utf-8")
            docs = loader.load()
            text = docs[0].page_content if docs else ""
            seiten = 0
        else:
            print(f"[modul_b_stub] Nicht unterstütztes Format: {pfad.name}")
            return None
    except Exception as e:
        print(f"[modul_b_stub] Fehler beim Lesen von {pfad.name}: {e}")
        return None

    text = _bereinige_text(text)
    if len(text.strip()) < 50:
        print(f"[modul_b_stub] Zu wenig Text in {pfad.name} – übersprungen.")
        return None

    return {
        "text":         text,
        "dokument_key": _generiere_key(pfad.name),
        "dateiname":    pfad.name,
        "dateipfad":    str(pfad.resolve()),
        "seiten":       seiten,
    }


def extrahiere_ordner(ordnerpfad: str) -> list[dict]:
    """
    Simuliert Modul B für einen ganzen Ordner.
    Gibt Liste von Dokument-Dicts zurück (None-Einträge gefiltert).
    """
    ordner = Path(ordnerpfad)
    dateien = (
        list(ordner.rglob("*.pdf"))
        + list(ordner.rglob("*.docx"))
        + list(ordner.rglob("*.txt"))
    )

    ergebnisse = []
    for datei in dateien:
        ergebnis = extrahiere_dokument(str(datei))
        if ergebnis:
            ergebnisse.append(ergebnis)

    print(f"[modul_b_stub] {len(ergebnisse)}/{len(dateien)} Dokumente extrahiert.")
    return ergebnisse


# ── Direktaufruf zum Testen ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Verwendung: python modul_b_stub.py <datei_oder_ordner>")
        sys.exit(1)

    pfad = Path(sys.argv[1])
    if pfad.is_dir():
        results = extrahiere_ordner(str(pfad))
        for r in results:
            print(f"  KEY: {r['dokument_key']} | {r['dateiname']} | {r['seiten']} Seiten | {len(r['text'])} Zeichen")
    else:
        result = extrahiere_dokument(str(pfad))
        if result:
            print(json.dumps({k: v if k != "text" else v[:300]+"..." for k, v in result.items()}, ensure_ascii=False, indent=2))
