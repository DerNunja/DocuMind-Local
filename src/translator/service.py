# service.py
import re
import torch
from models import ModelManager

class TranslationService:
    def __init__(self):
        self.manager = ModelManager()
        
        # Fachglossar für Französisch (Korrekturen für den französischen Output falls nötig)
        self.fr_glossar = {
            r"\bmargin\b": "bride",  # Au cas où
            r"\bvalve\b": "vanne"
        }

    def _wende_glossar_an(self, text: str) -> str:
        for fehler, korrektur in self.fr_glossar.items():
            text = re.sub(fehler, korrektur, text, flags=re.IGNORECASE)
        return text

    def übersetze_text(self, rohtext: str, modus: str = "de-en") -> dict:
        if not rohtext.strip():
            return {"original": "", "translated": "", "language": "de"}

        model, tokenizer, device = self.manager.get_model_and_tokenizer(modus)

        segmente = [s.strip() for s in re.split(r'(?<=[.!?])\s+', rohtext) if s.strip()]
        übersetzte_segmente = []

        for segment in segmente:
            inputs = tokenizer(segment, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                translated_tokens = model.generate(**inputs)
            
            satz_übersetzt = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
            übersetzte_segmente.append(satz_übersetzt)

        finaler_text = " ".join(übersetzte_segmente)
        
        # Glossar anwenden, wenn wir nach Französisch übersetzen
        if modus == "de-fr":
            finaler_text = self._wende_glossar_an(finaler_text)

        return {
            "original": rohtext,
            "translated": finaler_text,
            "language": "de"
        }
