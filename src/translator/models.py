import os
from transformers import MarianMTModel, MarianTokenizer
from config import MODEL_NAME, LOCAL_MODEL_DIR

class TranslationModelManager:
    """
    Singleton-Manager zum Laden und Verwalten des Übersetzungsmodells.
    Verhindert, dass das Modell bei jedem Aufruf neu in den RAM geladen werden muss.
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
        print(f"[*] Lokaler Speicherpfad: {LOCAL_MODEL_DIR}")
        
        # Sicherstellen, dass das lokale Verzeichnis existiert
        os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
        
        try:
            # Laden des Tokenizers und des Modells (lädt beim ersten Mal aus dem Internet, danach aus dem Cache)
            self.tokenizer = MarianTokenizer.from_pretrained(
                MODEL_NAME, 
                cache_dir=LOCAL_MODEL_DIR
            )
            self.model = MarianMTModel.from_pretrained(
                MODEL_NAME, 
                cache_dir=LOCAL_MODEL_DIR
            )
            self._initialized = True
            print("[+] Modell und Tokenizer erfolgreich geladen!")
        except Exception as e:
            print(f"[-] Fehler beim Laden des Modells: {str(e)}")
            raise e

    def get_model_and_tokenizer(self):
        """
        Gibt die geladenen Instanzen von Modell und Tokenizer zurück.
        """
        return self.model, self.tokenizer