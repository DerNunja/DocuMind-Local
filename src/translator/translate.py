import time
from service import TranslationService

def run_integration_test():
    print("=" * 60)
    print("[*] Starte automatisierten Test")
    print("=" * 60)

    test_saetze = [
        "Das Rohrleitungs- und Instrumentenfließschema muss überprüft werden.",
        "Der Wartungsplan ist im Anhang zu finden.",
        "Projektmanagement ermöglicht eine frühzeitige Prüfung.",
        "Sicherheitsventile müssen kalibriert werden."
    ]

    service = TranslationService()
    erfolgreich = 0

    for i, satz in enumerate(test_saetze, 1):
        print(f"\n[Test {i}] {satz}")

        start = time.time()
        ergebnis = service.übersetze_text(satz, modus="de-en")
        end = time.time()

        print(f"Übersetzung: {ergebnis['translated']}")
        print(f"Zeit: {end - start:.2f}s")

        if ergebnis["translated"]:
            erfolgreich += 1
            print("OK")
        else:
            print("FAIL")

    print(f"\nErgebnis: {erfolgreich}/{len(test_saetze)} erfolgreich")

if __name__ == "__main__":
    run_integration_test()
