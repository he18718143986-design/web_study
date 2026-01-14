
import json
import os
from typing import Any
from huggingface_hub import InferenceClient
from backend.llm.client import LLMClient


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
    def __init__(self, model_id: str = "bigscience/bloom-560m") -> None:
        self.model_id = model_id
        api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not api_key:
            raise RuntimeError("HUGGINGFACE_API_KEY not set; cannot call Hugging Face Inference API.")
        self.inf = InferenceClient(model=self.model_id, token=api_key)

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        import asyncio
        loop = asyncio.get_event_loop()
        try:
            try:
                res = await loop.run_in_executor(None, lambda: self.inf.text_generation(prompt))
            except BaseException as stop_exc:
                return _error_response(f"HF adapter internal error: {str(stop_exc)[:200]}")
            # 兼容返回格式
            if res is None:
                return _error_response("HF adapter returned None")
            if isinstance(res, dict) and "error" in res:
                return _error_response(f"HF API error: {res.get('error', 'Unknown error')}")
            if isinstance(res, list) and res and isinstance(res[0], dict) and "generated_text" in res[0]:
                return str(res[0]["generated_text"])
            if isinstance(res, dict) and "generated_text" in res:
                return str(res["generated_text"])
            return str(res)
        except BaseException as exc:
            return _error_response(f"HF adapter error: {str(exc)[:200]}")
