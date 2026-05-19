"""
modul_d_stub.py – Ersatz für Modul D (Übersetzer)

Wenn Modul D fertiggestellt wird, einfach den Import in ingest_pipeline.py
von `modul_d_stub` auf `modul_d` ändern.
"""

from __future__ import annotations


def _versuche_marian_import():
    """Prüft ob transformers/Marian verfügbar ist – ohne Absturz."""
    try:
        from transformers import MarianMTModel, MarianTokenizer
        return MarianMTModel, MarianTokenizer
    except ImportError:
        return None, None


def erkenne_sprache(text: str) -> str:
    """
    Einfache heuristische Spracherkennung (kein API-Key nötig).
    Für produktiven Einsatz: langdetect installieren.
    """
    try:
        from langdetect import detect
        return detect(text[:500])   # Erste 500 Zeichen reichen
    except Exception:
        pass

    # Heuristik als Fallback: deutsche Wörter zählen
    deutsche_woerter = {"und", "der", "die", "das", "ist", "mit", "von", "für",
                        "ein", "eine", "nicht", "auch", "auf", "an", "bei"}
    woerter = set(text.lower().split()[:100])
    treffer = len(woerter & deutsche_woerter)
    return "de" if treffer >= 3 else "en"


def uebersetze(text: str, zielsprache: str = "de") -> dict:
    """
    Simuliert die Ausgabe von Modul D.

    Rückgabe (identisch mit Modul D Interface):
        {
            "originaltext":    str,   # Eingabetext unverändert
            "uebersetzung":    str,   # übersetzter Text (oder Original als Fallback)
            "ausgangssprache": str,   # erkannte Ausgangssprache ("de", "en", ...)
            "zielsprache":     str,   # gewünschte Zielsprache
            "uebersetzt":      bool,  # True = echte Übersetzung, False = Fallback
        }
    """
    ausgangssprache = erkenne_sprache(text)

    # Keine Übersetzung nötig wenn bereits in Zielsprache
    if ausgangssprache == zielsprache:
        return {
            "originaltext":    text,
            "uebersetzung":    text,
            "ausgangssprache": ausgangssprache,
            "zielsprache":     zielsprache,
            "uebersetzt":      False,
        }

    # Versuch: Marian NMT lokal (optional)
    MarianMTModel, MarianTokenizer = _versuche_marian_import()
    if MarianMTModel is not None:
        try:
            modell_name = f"Helsinki-NLP/opus-mt-{ausgangssprache}-{zielsprache}"
            tokenizer = MarianTokenizer.from_pretrained(modell_name)
            modell    = MarianMTModel.from_pretrained(modell_name)

            # Text in Abschnitte aufteilen (Marian hat Token-Limit)
            abschnitte = [text[i:i+400] for i in range(0, len(text), 400)]
            uebersetzt = []
            for abschnitt in abschnitte:
                tokens = tokenizer([abschnitt], return_tensors="pt", padding=True, truncation=True)
                ids    = modell.generate(**tokens)
                uebersetzt.append(tokenizer.decode(ids[0], skip_special_tokens=True))

            return {
                "originaltext":    text,
                "uebersetzung":    " ".join(uebersetzt),
                "ausgangssprache": ausgangssprache,
                "zielsprache":     zielsprache,
                "uebersetzt":      True,
            }
        except Exception as e:
            print(f"[modul_d_stub] Marian fehlgeschlagen ({e}) → Fallback auf Original.")

    # Fallback: Originaltext zurückgeben (kein Absturz)
    print(f"[modul_d_stub] Kein Übersetzer verfügbar – Original wird verwendet.")
    return {
        "originaltext":    text,
        "uebersetzung":    text,       # ← Originaltext als Platzhalter
        "ausgangssprache": ausgangssprache,
        "zielsprache":     zielsprache,
        "uebersetzt":      False,
    }


# ── Direktaufruf zum Testen ──────────────────────────────────────────────
if __name__ == "__main__":
    testtext_en = "The project budget was approved. The team will begin implementation next week."
    testtext_de = "Das Budget wurde genehmigt. Das Team beginnt nächste Woche mit der Umsetzung."

    print("=== Test 1: Englisch → Deutsch ===")
    result = uebersetze(testtext_en, "de")
    print(f"  Übersetzt: {result['uebersetzt']}")
    print(f"  Sprache:   {result['ausgangssprache']} → {result['zielsprache']}")
    print(f"  Ergebnis:  {result['uebersetzung'][:100]}")

    print("\n=== Test 2: Deutsch → gleiche Sprache (kein Aufwand) ===")
    result2 = uebersetze(testtext_de, "de")
    print(f"  Übersetzt: {result2['uebersetzt']}  (erwartet: False)")
