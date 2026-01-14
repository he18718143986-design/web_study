import json
from typing import Any, Dict, List

from backend.llm.client import LLMClient


class MockAdapter(LLMClient):
    async def generate(self, prompt: str, **kwargs: Any) -> str:
        # Deterministic structured payload to satisfy schema.
        summary_points: List[Dict[str, str]] = [
            {"id": "p1", "text": f"Echo of '{prompt}'", "confidence": "high"},
            {"id": "p2", "text": "Mock systems return static data", "confidence": "medium"},
            {"id": "p3", "text": "Use real adapters in production", "confidence": "medium"},
        ]
        structured: Dict[str, Any] = {
            "summary_points": summary_points,
            "detailed_explanation": "This is a mock structured response for testing pipelines.",
            "evidence": ["https://example.com/mock"],
            "reproducible_example": "print('mock')",
        }
        return json.dumps(structured)
