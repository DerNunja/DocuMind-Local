import os
os.environ["HF_SKIP_CHECK_TORCH_LOAD_IN_SAFE"] = "True"

import torch
from transformers import MarianMTModel, MarianTokenizer
from config import SUPPORTED_MODELS, LOCAL_MODEL_DIR

class ModelManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loaded_models = {}

        print(f"[*] Device: {self.device.upper()}")

    def get_model_and_tokenizer(self, mode="de-en"):

        if mode not in SUPPORTED_MODELS:
            raise ValueError(f"Nicht unterstützter Modus: {mode}")

        # Cache already loaded models in RAM
        if mode in self.loaded_models:
            return self.loaded_models[mode]

        model_name = SUPPORTED_MODELS[mode]

        tokenizer = MarianTokenizer.from_pretrained(
            model_name,
            cache_dir=LOCAL_MODEL_DIR
        )

        model = MarianMTModel.from_pretrained(
            model_name,
            cache_dir=LOCAL_MODEL_DIR,
            use_safetensors=True
        ).to(self.device)

        self.loaded_models[mode] = (model, tokenizer, self.device)

        return self.loaded_models[mode]
