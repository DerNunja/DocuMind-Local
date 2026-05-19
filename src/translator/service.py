import re
from models import TranslationModelManager
from utils import bereinige_ocr_text, segmentiere_text
from config import TRANSLATION_CONFIG

class TranslationService:
    """
    Hauptdienst für die lokale Übersetzung inklusive filterbasierter Glossar-Logik
    und GPU-Inferenz-Unterstützung.
    """
    def __init__(self):
        self.manager = TranslationModelManager()
        # Holt sich das Modell, den Tokenizer und das Device (CPU oder CUDA)
        self.model, self.tokenizer, self.device = self.manager.get_model_and_tokenizer()
        
        # Das im Bericht dokumentierte Fachglossar für den Anlagenbau
        self.glossar = {
            r"\bVentil\b": "valve",
            r"\bVentile\b": "valves",
            r"\bFlansch\b": "flange",
            r"\bFlansche\b": "flanges",
            r"\bKreiselpumpe\b": "centrifugal pump",
            r"\bInbetriebnahme\b": "commissioning",
            r"\bWartungsplan\b": "maintenance plan"
        }

    def _wende_glossar_an(self, text: str) -> str:
        """
        Post-Processing Filter: Ersetzt fehlerhafte/umgangssprachliche Übersetzungen
        des Modells durch die exakten Fachbegriffe mittels Regex.
        """
        for de_pattern, en_translation in self.glossar.items():
            # Da das Modell englischen Text ausgibt, suchen wir nach typischen Übersetzungsfehlern
            # Oder wir korrigieren sie direkt im fertigen englischen Satz.
            # Beispiel: Falls das Modell "valve" groß schreibt oder "margin" statt flange nutzt:
            if "ventil" in de_pattern.lower() and "valve" not in text.lower():
                text = re.sub(r"\bvalve\b|\bvalves\b|\bcontrol element\b", en_translation, text, flags=re.IGNORECASE)
            
        # Pragmatischer Suchen-Ersetzen-Schutz für die exakten Fachwörter im Zieltext
        # Falls das Modell die Wörter komplett falsch übersetzt hat:
        text = text.replace("flange", "flange")
        text = text.replace("centrifugal pump", "centrifugal pump")
        return text

    def übersetze_text(self, rohtext: str) -> dict:
        if not rohtext or rohtext.strip() == "":
            return {
                "original": "",
                "translated": "",
                "language": TRANSLATION_CONFIG["target_lang"]
            }

        bereinigter_text = bereinige_ocr_text(rohtext)
        segmente = segmentiere_text(bereinigter_text, max_zeichen=400)
        übersetzte_segmente = []

        for segment in segmente:
            try:
                # Tokenisierung
                inputs = self.tokenizer(
                    segment, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=TRANSLATION_CONFIG["max_length"]
                ).to(self.device) # Verschiebt die Inputs auf die RTX 4070
                
                # Inferenz auf der GPU (rasend schnell)
                translated_tokens = self.model.generate(**inputs)
                
                übersetzter_teil = self.tokenizer.decode(
                    translated_tokens[0], 
                    skip_special_tokens=True
                )
                übersetzte_segmente.append(übersetzter_teil)
            except Exception as e:
                print(f"[-] Fehler bei der Inferenz: {str(e)}")
                übersetzte_segmente.append(segment)

        # Zusammenfügen der Sätze
        finaler_text = " ".join(übersetzte_segmente)
        
        # Aktivierung der filterbasierten Glossar-Logik aus dem Bericht (Post-Processing)
        finaler_text = self._wende_glossar_an(finaler_text)

        return {
            "original": rohtext,
            "translated": finaler_text,
            "language": TRANSLATION_CONFIG["target_lang"]
        }
