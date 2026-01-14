import json
import os
from typing import Any, Dict, Optional

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


class HuggingFaceAdapter(LLMClient):
    def __init__(self, model_id: str = "gpt2") -> None:
        self.model_id = model_id

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not api_key:
            return _error_response("HUGGINGFACE_API_KEY not set; cannot call Hugging Face Inference API.")

        endpoint = f"https://api-inference.huggingface.co/models/{self.model_id}"
        headers = {"Authorization": f"Bearer {api_key}"}

        payload: Dict[str, Any] = {"inputs": prompt}
        parameters: Optional[Dict[str, Any]] = kwargs.get("parameters")
        if parameters:
            payload["parameters"] = parameters

        try:
            response = await post_json(endpoint, payload, headers=headers)
        except Exception as exc:  # pragma: no cover - network dependent
            error_msg = str(exc)[:200]  # Limit error message length
            return _error_response(f"HF adapter error after retries: {error_msg}")

        # Check for empty response
        if not response.text or not response.text.strip():
            return _error_response("Empty response from Hugging Face API")
        
        try:
            data = response.json()
        except Exception as json_exc:
            # If response is not JSON, return error message as structured response
            error_msg = response.text[:200] if response.text else "Invalid JSON response"
            return _error_response(f"HF API returned non-JSON: {error_msg}")
        
        # Handle HF API error responses
        if isinstance(data, dict):
            if "error" in data:
                return _error_response(f"HF API error: {data.get('error', 'Unknown error')}")
            if "estimated_time" in data:
                # Model is loading
                return _error_response(f"HF model is loading, estimated time: {data.get('estimated_time')}s")
        
        # Extract generated text from various HF response formats
        if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
            return str(data[0]["generated_text"])
        if isinstance(data, dict) and "generated_text" in data:
            return str(data["generated_text"])
        
        # If we can't parse the response, return it as an error structured response
        return _error_response(f"HF adapter received unexpected response format: {str(data)[:200]}")
