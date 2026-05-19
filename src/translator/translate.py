# translate.py
import os
# DAS HIER MUSS DIE ABSOLUT ERSTE ZEILE SEIN, BEVOR IRGENDWAS ANDERES IMPORTIERT WIRD!
os.environ["HF_SKIP_CHECK_TORCH_LOAD_IN_SAFE"] = "True"

import argparse
import json
import sys
from service import TranslationService

def main():
    parser = argparse.ArgumentParser(
        description="Modul D – Konsolen-Schnittstelle"
    )
    parser.add_argument(
        "--text", 
        type=str, 
        required=True,
        help="Der deutsche Text, der übersetzt werden soll."
    )
    
    args = parser.parse_args()

    service = TranslationService()
    ergebnis = service.übersetze_text(args.text)
    
    # Gibt das saubere JSON in der Konsole aus
    print(json.dumps(ergebnis, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    main()
