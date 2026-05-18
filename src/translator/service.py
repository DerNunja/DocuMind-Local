# service.py
from models import TranslationModelManager
from utils import bereinige_ocr_text, segmentiere_text
from config import TRANSLATION_CONFIG

class TranslationService:
    """
    Hauptdienst für die Durchführung der lokalen Übersetzung von Deutsch nach Englisch.
    Nutzt den Model-Manager und die Textbereinigung.
    """
    def __init__(self):
        # Holt sich die Singleton-Instanz des Modells
        self.manager = TranslationModelManager()
        self.model, self.tokenizer = self.manager.get_model_and_tokenizer()

    def übersetze_text(self, rohtext: str) -> dict:
        """
        Nimmt unsauberen deutschen Text entgegen, bereinigt ihn,
        übersetzt ihn ins Englische und gibt ein strukturiertes Dictionary zurück.
        """
        if not rohtext or rohtext.strip() == "":
            return {
                "original": "",
                "translated": "",
                "language": TRANSLATION_CONFIG["target_lang"]
            }

        # 1. Schritt: Textbereinigung (OCR-Fehler entfernen)
        bereinigter_text = bereinige_ocr_text(rohtext)
        
        # 2. Schritt: Text in verarbeitbare Segmente aufteilen
        segmente = segmentiere_text(bereinigter_text, max_zeichen=400)
        übersetzte_segmente = []

        # 3. Schritt: Segmentweise Übersetzung durchführen
        for segment in segmente:
            try:
                # Tokenisierung des Textsegments für PyTorch
                inputs = self.tokenizer(
                    segment, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=TRANSLATION_CONFIG["max_length"]
                )
                
                # Generierung der Übersetzung (Inferenz auf der CPU)
                translated_tokens = self.model.generate(**inputs)
                
                # Dekodierung der Tokens zurück in lesbaren Text
                übersetzter_teil = self.tokenizer.decode(
                    translated_tokens[0], 
                    skip_special_tokens=True
                )
                übersetzte_segmente.append(übersetzter_teil)
            except Exception as e:
                print(f"[-] Fehler bei der Übersetzung des Segments: {str(e)}")
                # Falls ein Segment fehlschlägt, das Original als Fallback behalten
                übersetzte_segmente.append(segment)

        # 4. Schritt: Zusammenfügen der übersetzten Segmente
        finaler_übersetzter_text = " ".join(übersetzte_segmente)

        # 5. Schritt: Standardisiertes Ausgabeformat zurückgeben
        return {
            "original": rohtext,
            "translated": finaler_übersetzter_text,
            "language": TRANSLATION_CONFIG["target_lang"]  # Gibt fest "en" zurück
        }