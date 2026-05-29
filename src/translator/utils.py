# utils.py
import re

# =====================================================
# OCR CLEANING
# =====================================================

def bereinige_ocr_text(text: str) -> str:
    """
    Cleans noisy OCR text for better translation quality.
    """

    if not text:
        return ""

    # ---------------------------------------------
    # Fix hyphen line breaks
    # e.g. "Projekt-\nmanagement" -> "Projektmanagement"
    # ---------------------------------------------

    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)

    # ---------------------------------------------
    # Replace newlines with spaces
    # ---------------------------------------------

    text = text.replace("\n", " ")

    # ---------------------------------------------
    # Normalize multiple spaces
    # ---------------------------------------------

    text = re.sub(r"\s+", " ", text)

    # ---------------------------------------------
    # Trim
    # ---------------------------------------------

    return text.strip()


# =====================================================
# SENTENCE SEGMENTATION
# =====================================================

def segmentiere_text(text: str, max_zeichen: int = 400) -> list:
    """
    Splits text into translation-friendly segments.
    """

    if not text:
        return []

    # ---------------------------------------------
    # Split into sentences
    # ---------------------------------------------

    saetze = re.split(r"(?<=[.!?])\s+", text)

    segmente = []
    aktuelles_segment = ""

    # ---------------------------------------------
    # Build segments under character limit
    # ---------------------------------------------

    for satz in saetze:

        satz = satz.strip()

        if not satz:
            continue

        # If adding would exceed limit -> flush
        if len(aktuelles_segment) + len(satz) + 1 > max_zeichen:

            if aktuelles_segment:
                segmente.append(aktuelles_segment.strip())

            aktuelles_segment = satz

        else:

            if aktuelles_segment:
                aktuelles_segment += " " + satz
            else:
                aktuelles_segment = satz

    # ---------------------------------------------
    # Append last segment
    # ---------------------------------------------

    if aktuelles_segment:
        segmente.append(aktuelles_segment.strip())

    return segmente


# =====================================================
# OPTIONAL: TEXT QUALITY CHECK (future-proof)
# =====================================================

def ist_technischer_text(text: str) -> bool:
    """
    Simple heuristic to detect technical German text.
    Useful later for routing models or preprocessing.
    """

    keywords = [
        "Ventil",
        "Pumpe",
        "Druck",
        "Rohr",
        "Flansch",
        "Wartung",
        "Sicherheits"
    ]

    return any(k.lower() in text.lower() for k in keywords)

