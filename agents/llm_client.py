"""Thin OpenAI-compatible client for talking to a local LLM server (vLLM or
Ollama) or a real hosted API - only base_url (and optionally api_key) needs
to change. Every agent calls this same chat() method, so swapping models or
backends never touches agent logic.
"""

import os
from typing import Dict, List

from openai import OpenAI


class LLMClient:
    def __init__(self, model_config: dict):
        self.model_name = model_config["name"]
        self.temperature = model_config.get("temperature", 0.0)
        self.max_tokens = model_config.get("max_tokens", 512)
        self.seed = model_config.get("seed", 42)

        base_url = model_config.get("base_url", "http://localhost:8000/v1")
        # Local servers (vLLM/Ollama) accept any non-empty string as the API key.
        api_key = os.environ.get("OPENAI_API_KEY", "not-needed-for-local-models")
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def chat(self, messages: List[Dict[str, str]], **overrides) -> str:
        """Send a chat completion request and return the assistant's text."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=overrides.get("temperature", self.temperature),
            max_tokens=overrides.get("max_tokens", self.max_tokens),
            seed=overrides.get("seed", self.seed),
        )
        return response.choices[0].message.content or ""

