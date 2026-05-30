from __future__ import annotations

import json
import os
import re
from typing import Any

import requests

from .prompts import REPAIR_SYSTEM_PROMPT, REPAIR_USER_TEMPLATE


#DEFAULT_CHAT_MODEL = os.getenv("LM_STUDIO_CHAT_MODEL", "qwen3.6-35b-a3b")
DEFAULT_CHAT_MODEL = os.getenv("LM_STUDIO_CHAT_MODEL", "google/gemma-4-e4b")
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "LM_STUDIO_EMBEDDING_MODEL", "text-embedding-qwen3-embedding-4b"
)
DEFAULT_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")


class LMStudioError(RuntimeError):
    pass


class LMStudioClient:
    def __init__(
        self,
        chat_model: str = DEFAULT_CHAT_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = 120,
    ) -> None:
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        content = self.chat(system_prompt, user_prompt)
        try:
            return parse_json_object(content)
        except ValueError:
            repaired = self.chat(
                REPAIR_SYSTEM_PROMPT.strip(),
                REPAIR_USER_TEMPLATE.replace("{content}", content).strip(),
            )
            return parse_json_object(repaired)

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.chat_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def embed(self, text: str) -> list[float]:
        payload = {"model": self.embedding_model, "input": text}
        response = requests.post(
            f"{self.base_url}/embeddings",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]

    def list_models(self) -> list[str]:
        return list_models(self.base_url, timeout=self.timeout)


def list_models(base_url: str = DEFAULT_BASE_URL, timeout: int = 5) -> list[str]:
    response = requests.get(f"{base_url.rstrip('/')}/models", timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return sorted(
        model["id"]
        for model in data.get("data", [])
        if isinstance(model, dict) and model.get("id")
    )


def list_chat_models(base_url: str = DEFAULT_BASE_URL, timeout: int = 5) -> list[str]:
    return [model for model in list_models(base_url, timeout) if not is_embedding_model(model)]


def is_embedding_model(model_id: str) -> bool:
    normalized = model_id.lower()
    return "embedding" in normalized or "embed" in normalized


def parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    elif not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in LLM response")
        text = text[start : end + 1]
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object")
    return parsed
