import re

def bereinige_ocr_text(text: str) -> str:
    """
    Bereinigt unsauberen Text aus der Digitalisierung (OCR), 
    um die Übersetzungsqualität des Modells zu maximieren.
    """
    if not text:
        return ""
    
    # 1. Trennstriche am Zeilenende entfernen (z.B. "Projekt-\nmanagement" -> "Projektmanagement")
    # Das \\s* fängt eventuelle Leerzeichen vor oder nach dem Zeilenumbruch ab
    text = re.sub(f"(\\w+)-\\s*\\n\\s*(\\w+)", r"\1\2", text)
    
    # 2. Einfache Zeilenumbrüche durch normale Leerzeichen ersetzen,
    # damit zusammenhängende Sätze nicht zerschnitten werden
    text = text.replace("\n", " ")
    
    # 3. Mehrfache aufeinanderfolgende Leerzeichen zu einem einzigen Leerzeichen reduzieren
    text = re.sub(r"\s+", " ", text)
    
    # 4. Führende und anhängende Leerzeichen abschneiden
    return text.strip()

def segmentiere_text(text: str, max_zeichen: int = 400) -> list:
    """
    Teilt einen langen Text in kleinere, logische Abschnitte (Sätze) auf.
    Das Übersetzungsmodell arbeitet am besten mit einzelnen Sätzen statt riesigen Textblöcken.
    """
    # Einfache Aufteilung anhand von Satzzeichen (. ! ?) gefolgt von einem Leerzeichen
    saetze = re.split(r"(?<=[.!?])\s+", text)
    
    segmente = []
    aktuelles_segment = ""
    
    for satz in saetze:
        # Falls ein einzelner Satz schon zu lang ist oder das Limit mit dem neuen Satz überschritten wird
        if len(aktuelles_segment) + len(satz) + 1 > max_zeichen:
            if aktuelles_segment:
                segmente.append(aktuelles_segment.strip())
            aktuelles_segment = satz
        else:
            if aktuelles_segment:
                aktuelles_segment += " " + satz
            else:
                aktuelles_segment = satz
                
    if aktuelles_segment:
        segmente.append(aktuelles_segment.strip())
        
    return segmente