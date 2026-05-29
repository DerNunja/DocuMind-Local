#service.py
import re
import torch
from models import ModelManager
from utils import bereinige_ocr_text, segmentiere_text


class TranslationService:

    def __init__(self):

        # ---------------------------------------------------
        # Model Manager
        # ---------------------------------------------------

        self.manager = ModelManager()

        # ---------------------------------------------------
        # Technical Glossaries
        # ---------------------------------------------------

        self.glossare = {

            # -----------------------------------------------
            # German -> French
            # -----------------------------------------------

            "de-fr": {

                r"\bvalve\b": "vanne",
                r"\bpump\b": "pompe",

                r"\bSicherheitsventil\b": "soupape de sécurité",
                r"\bKreiselpumpe\b": "pompe centrifuge",
                r"\bFlansch\b": "bride",
            },

            # -----------------------------------------------
            # German -> Arabic
            # -----------------------------------------------

            "de-ar": {

                r"\bpump\b": "مضخة",
                r"\bvalve\b": "صمام",

                r"\bSicherheitsventil\b": "صمام أمان",
                r"\bKreiselpumpe\b": "مضخة طرد مركزي",
                r"\bFlansch\b": "شفة",
            },

            # -----------------------------------------------
            # German -> English
            # -----------------------------------------------

            "de-en": {

                r"\bKreiselpumpe\b": "centrifugal pump",
                r"\bSicherheitsventil\b": "safety valve",
                r"\bFlansch\b": "flange",
            }
        }

    # =====================================================
    # Apply Glossary
    # =====================================================

    def _wende_glossar_an(self, text: str, modus: str) -> str:

        if modus not in self.glossare:
            return text

        for fehler, korrektur in self.glossare[modus].items():

            text = re.sub(
                fehler,
                korrektur,
                text,
                flags=re.IGNORECASE
            )

        return text

    # =====================================================
    # Main Translation Function
    # =====================================================

    def übersetze_text(
        self,
        rohtext: str,
        modus: str = "de-en"
    ) -> dict:

        # -------------------------------------------------
        # Empty Input Protection
        # -------------------------------------------------

        if not rohtext.strip():

            return {
                "original": "",
                "translated": "",
                "mode": modus
            }

        # -------------------------------------------------
        # OCR Cleanup
        # -------------------------------------------------

        bereinigter_text = bereinige_ocr_text(rohtext)

        # -------------------------------------------------
        # Segment Text
        # -------------------------------------------------

        segmente = segmentiere_text(
            bereinigter_text,
            max_zeichen=400
        )

        # -------------------------------------------------
        # Load Model
        # -------------------------------------------------

        model, tokenizer, device = (
            self.manager.get_model_and_tokenizer(modus)
        )

        übersetzte_segmente = []

        # -------------------------------------------------
        # Translate Segment by Segment
        # -------------------------------------------------

        for segment in segmente:

            try:

                # Tokenize
                inputs = tokenizer(
                    segment,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(device)

                # Inference
                with torch.no_grad():

                    translated_tokens = model.generate(
                        **inputs,
                        max_length=512,
                        num_beams=4,
                        early_stopping=True
                    )

                # Decode
                satz_übersetzt = tokenizer.decode(
                    translated_tokens[0],
                    skip_special_tokens=True
                )

                übersetzte_segmente.append(
                    satz_übersetzt
                )

            except Exception as e:

                print(f"[!] Fehler bei Segment: {segment}")
                print(f"[!] Details: {e}")

                übersetzte_segmente.append(
                    "[ÜBERSETZUNGSFEHLER]"
                )

        # -------------------------------------------------
        # Merge Final Text
        # -------------------------------------------------

        finaler_text = " ".join(
            übersetzte_segmente
        )

        # -------------------------------------------------
        # Apply Technical Glossary
        # -------------------------------------------------

        finaler_text = self._wende_glossar_an(
            finaler_text,
            modus
        )

        # -------------------------------------------------
        # Final JSON Response
        # -------------------------------------------------

        return {

            "original": rohtext,

            "cleaned_text": bereinigter_text,

            "translated": finaler_text,

            "mode": modus,

            "segments": len(segmente),

            "device": device
        }
