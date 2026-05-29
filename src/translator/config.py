import os

os.environ["HF_SKIP_CHECK_TORCH_LOAD_IN_SAFE"] = "True"

LOCAL_MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_cache")

SUPPORTED_MODELS = {
    "de-en": "Helsinki-NLP/opus-mt-de-en",
    "de-fr": "Helsinki-NLP/opus-mt-de-fr",
    "de-ar": "Helsinki-NLP/opus-mt-de-ar",

    "fr-de": "Helsinki-NLP/opus-mt-fr-de",
    "ar-de": "Helsinki-NLP/opus-mt-ar-de",
    "en-de": "Helsinki-NLP/opus-mt-en-de",
}

TRANSLATION_CONFIG = {
    "max_length": 512,
}
