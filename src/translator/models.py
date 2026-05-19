import os
import torch
from transformers import MarianMTModel, MarianTokenizer
from config import MODEL_NAME, LOCAL_MODEL_DIR

class TranslationModelManager:
    """
    Singleton-Manager zum Laden und Verwalten des Übersetzungsmodells auf der GPU/CPU.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TranslationModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        print(f"[*] Initialisiere Übersetzungsmodell: {MODEL_NAME}...")
        
        os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
        
        # Automatische Erkennung der RTX 4070 (CUDA)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] Nutze Hardware-Beschleunigung: {self.device.upper()}")
        
        try:
            self.tokenizer = MarianTokenizer.from_pretrained(
                MODEL_NAME, 
                cache_dir=LOCAL_MODEL_DIR
            )
            # Modell laden und direkt auf die Grafikkarte (VRAM) schieben
            self.model = MarianMTModel.from_pretrained(
                MODEL_NAME, 
                cache_dir=LOCAL_MODEL_DIR
            ).to(self.device)
            
            self._initialized = True
            print("[+] Modell erfolgreich in den Speicher geladen!")
        except Exception as e:
            print(f"[-] Fehler beim Laden des Modells: {str(e)}")
            raise e

    def get_model_and_tokenizer(self):
        """
        Gibt Modell, Tokenizer und das aktive Device zurück.
        """
        return self.model, self.tokenizer, self.device
