# models.py
import os
os.environ["HF_SKIP_CHECK_TORCH_LOAD_IN_SAFE"] = "True"

import torch
from transformers import MarianMTModel, MarianTokenizer

LOCAL_MODEL_DIR = "./marian_model"

class ModelManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] Initialisiere Model-Manager auf Hardware-Beschleuniger: {self.device.upper()}")

    def get_model_and_tokenizer(self, mode="de-en"):
        """Lädt das passende Modell basierend auf dem Modus (de-en oder de-fr)."""
        if mode == "de-fr":
            model_name = "Helsinki-NLP/opus-mt-de-fr"
        else:
            model_name = "Helsinki-NLP/opus-mt-de-en"
        
        try:
            tokenizer = MarianTokenizer.from_pretrained(model_name, cache_dir=LOCAL_MODEL_DIR)
            model = MarianMTModel.from_pretrained(
                model_name, 
                cache_dir=LOCAL_MODEL_DIR,
                use_safetensors=True
            ).to(self.device)
            return model, tokenizer, self.device
        except Exception as e:
            print(f"[!] Fehler beim Laden des Modells ({model_name}): {e}")
            raise e
