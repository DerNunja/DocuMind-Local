import os

# Auswahl des Hugging-Face-Modells (Spezialisiert und extrem leicht für CPU)
# Dieses Modell übersetzt spezifisch von Deutsch (DE) nach Englisch (EN)
MODEL_NAME = "Helsinki-NLP/opus-mt-de-en"

# Lokaler Ordner zum Speichern des Modells (verhindert erneutes Herunterladen bei jedem Start)
LOCAL_MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_cache")

# Basis-Konfiguration für den Übersetzungsdienst
TRANSLATION_CONFIG = {
    "source_lang": "de",
    "target_lang": "en",
    "max_length": 512,  # Maximale Länge pro Textsegment, um CPU-Überlastung zu vermeiden
}