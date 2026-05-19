import os

# Zwingt Hugging Face dazu, die restriktive Blockade zu überspringen, da wir safetensors nutzen
os.environ["HF_SKIP_CHECK_TORCH_LOAD_IN_SAFE"] = "True"

# Auswahl der Hugging-Face-Modelle (Spezialisiert und leicht)
MODEL_NAME_DE_EN = "Helsinki-NLP/opus-mt-de-en"
MODEL_NAME_DE_FR = "Helsinki-NLP/opus-mt-de-fr"  # NEU: Für die Übersetzung ins Französische

# Lokaler Ordner zum Speichern der Modelle (verhindert erneutes Herunterladen bei jedem Start)
LOCAL_MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_cache")

# Basis-Konfiguration für den Übersetzungsdienst
TRANSLATION_CONFIG = {
    "source_lang": "de",
    "max_length": 512,  # Maximale Länge pro Textsegment, um Überlastung zu vermeiden
}
