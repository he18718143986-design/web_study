import json
import os
from typing import Any, Dict

from backend.llm.client import LLMClient
from backend.services.http_retry import post_json


def _error_response(error_msg: str) -> str:
    """Generate a structured error response that conforms to the schema."""
    return json.dumps({
        "summary_points": [
            {
                "id": "error",
                "text": f"Error: {error_msg}",
                "confidence": "low"
            }
        ],
        "detailed_explanation": f"This response indicates an error occurred: {error_msg}",
        "evidence": [],
        "reproducible_example": ""
    })


class OllamaAdapter(LLMClient):
    """Calls a local Ollama server (open-source model runner)."""

    def __init__(self, model_id: str = "llama3.2") -> None:
        self.model_id = model_id
        self.endpoint = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        payload: Dict[str, Any] = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
        }
        params = kwargs.get("parameters")
        if params:
            payload.update(params)

        try:
            resp = await post_json(self.endpoint, payload)
            data = resp.json()
            if isinstance(data, dict) and data.get("response"):
                return str(data.get("response"))
            return _error_response(f"Ollama adapter unexpected response: {data}")
        except Exception as exc:  # pragma: no cover - network dependent
            error_msg = str(exc)[:200]  # Limit error message length
            return _error_response(f"Ollama adapter error after retries: {error_msg}")
