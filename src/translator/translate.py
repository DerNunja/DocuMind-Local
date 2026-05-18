# translate.py
import argparse
import json
import sys
from service import TranslationService

def uebersetze(text: str) -> dict:
    """
    Diese Funktion wird von der externen Ingest-Pipeline aufgerufen.
    Sie initialisiert den Service und liefert das standardisierte Dictionary zurück.
    """
    service = TranslationService()
    return service.übersetze_text(text)

def main():
    # Einrichten des ArgumentParsers für die Konsole
    parser = argparse.ArgumentParser(
        description="Modul D – Lokales Übersetzungsmodul (DE -> EN) für die Nordharz Anlagenbau GmbH"
    )
    
    # Erlaubt entweder direkten Text per Flag oder eine Standard-Eingabe
    parser.add_argument(
        "--text", 
        type=str, 
        help="Der deutsche Text, der ins Englische übersetzt werden soll."
    )
    
    args = parser.parse_args()

    # Falls Text übergeben wurde, diesen übersetzen
    if args.text:
        print("[*] Starte Übersetzungsvorgang...", file=sys.stderr)
        ergebnis = uebersetze(args.text)
        
        # Ausgabe des Ergebnisses als schön formatiertes JSON in der Konsole
        print(json.dumps(ergebnis, indent=4, ensure_ascii=False))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()