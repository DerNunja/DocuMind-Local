# test_translation.py
import time
from translate import uebersetze

def run_integration_test():
    print("=" * 60)
    print("[*] Starte automatisierten Validierungstest für Modul D")
    print("=" * 60)

    # Testdaten: Typische Sätze aus dem industriellen Anlagenbau der Nordharz GmbH
    test_saetze = [
        "Das Rohrleitungs- und Instrumentenfließschema (R&I) muss überprüft werden.",
        "Der Wartungsplan für die Kreiselpumpe ist im Anhang zu finden.",
        "Projektmanagement ermöglicht eine frühzeitige Machbarkeitsprüfung.",
        "Sicherheitsventile müssen vor der Inbetriebnahme kalibriert werden."
    ]

    erfolgreich = 0

    for i, satz in enumerate(test_saetze, 1):
        print(f"\n[Test {i}] Original (DE): '{satz}'")
        
        start_time = time.time()
        # Aufruf deiner Übersetzungsfunktion
        ergebnis = uebersetze(satz)
        end_time = time.time()
        
        laufzeit = end_time - start_time
        
        print(f"[Test {i}] Übersetzung (EN): '{ergebnis['translated']}'")
        print(f"[Test {i}] Sprache: {ergebnis['language']} | Laufzeit: {laufzeit:.2f}s")
        
        if ergebnis['translated'] != "" and ergebnis['language'] == "en":
            erfolgreich += 1
            print(f"[+] Test {i} ERFOLGREICH!")
        else:
            print(f"[-] Test {i} FEHLGESCHLAGEN!")

    print("\n" + "=" * 60)
    print(f"[ZUSAMMENFASSUNG] {erfolgreich} von {len(test_saetze)} Tests erfolgreich absolviert.")
    print("=" * 60)

if __name__ == "__main__":
    run_integration_test()
