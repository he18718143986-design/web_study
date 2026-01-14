import asyncio
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import jsonschema

from backend.llm.adapters.hf_adapter import HuggingFaceAdapter
from backend.llm.adapters.mock_adapter import MockAdapter
from backend.llm.adapters.ollama_adapter import OllamaAdapter
from backend.prompt.registry import get_prompt

ResponseItem = Dict[str, Any]


_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schema" / "structured_response.json"
try:
    _STRUCTURED_SCHEMA = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
except Exception:
    _STRUCTURED_SCHEMA = None


def _build_client(model_id: str):
    mid = model_id.lower()
    if mid.startswith("mock"):
        return MockAdapter()
    if mid == "hf":
        # Allow overriding HF model id via env; default bigscience/bloom-560m for cloud inference.
        model_name = os.getenv("HF_MODEL_ID", "bigscience/bloom-560m")
        return HuggingFaceAdapter(model_id=model_name)
    if mid == "ollama":
        model_name = os.getenv("OLLAMA_MODEL", "llama3.2")
        return OllamaAdapter(model_id=model_name)
    raise ValueError(f"Unsupported model id: {model_id}")


def _render_prompt(prompt_id: str, prompt_version: str, question: str) -> tuple[str, Optional[str]]:
    template = get_prompt(prompt_id, prompt_version)
    if not template:
        return question, None
    rendered = template.replace("<USER_QUESTION>", question)
    rendered = rendered.replace("<CONTEXT_HISTORY>", "")
    prompt_hash = hashlib.sha256(template.encode("utf-8")).hexdigest()
    return rendered, prompt_hash


async def _call_model(model_id: str, question: str, structured: bool, prompt_id: str, prompt_version: str) -> ResponseItem:
    try:
        client = _build_client(model_id)
        prompt_used, prompt_hash = _render_prompt(prompt_id, prompt_version, question)
        request_id = hashlib.sha256(f"{model_id}:{prompt_id}:{prompt_version}:{time.time_ns()}".encode()).hexdigest()[:16]
        t0 = time.perf_counter()
        raw = await client.generate(prompt_used)
        latency = time.perf_counter() - t0
        model_name = getattr(client, "model_id", model_id)
        meta = {
            "model_id": model_id,
            "backend": model_id.lower(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_name": model_name,
            "prompt_id": prompt_id,
            "prompt_version": prompt_version,
        }
        if prompt_hash:
            meta["prompt_hash"] = prompt_hash
        meta["prompt_used"] = prompt_used
        meta["request_id"] = request_id
        meta["latency_s"] = round(latency, 4)
        if not structured:
            meta["usage_tokens_estimate"] = len(str(raw).split())
            return {"model_id": model_id, "raw": raw, "meta": meta}

        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("Parsed JSON is not an object")
            if _STRUCTURED_SCHEMA:
                jsonschema.validate(instance=parsed, schema=_STRUCTURED_SCHEMA)
            meta["usage_tokens_estimate"] = len(json.dumps(parsed).split())
            return {"model_id": model_id, "parsed": parsed, "raw": raw, "meta": meta}
        except Exception as exc:
            meta["usage_tokens_estimate"] = len(str(raw).split())
            return {"model_id": model_id, "raw": raw, "parse_error": str(exc), "meta": meta}
    except Exception as exc:  # pragma: no cover - protective path
        return {"model_id": model_id, "error": str(exc)}


async def multi_model_query(
    question: str,
    model_ids: List[str],
    structured: bool = False,
    prompt_id: str = "answerer_v1",
    prompt_version: str = "v1",
) -> Dict[str, Any]:
    # 为每个模型调用添加超时保护（90秒），避免慢速模型阻塞整个请求
    async def _call_with_timeout(model_id: str) -> ResponseItem:
        try:
            return await asyncio.wait_for(
                _call_model(model_id.strip(), question, structured, prompt_id, prompt_version),
                timeout=180.0
            )
        except asyncio.TimeoutError:
            return {
                "model_id": model_id,
                "error": f"Model call timed out after 90 seconds",
                "meta": {
                    "model_id": model_id,
                    "timeout": True,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            }
        except Exception as exc:
            return {
                "model_id": model_id,
                "error": f"Model call failed: {str(exc)}",
                "meta": {
                    "model_id": model_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            }
    
    tasks = [
        _call_with_timeout(mid)
        for mid in model_ids
        if mid.strip()
    ]
    # 使用 return_exceptions=True，这样即使某个模型失败，其他模型的结果也能返回
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 将异常转换为错误响应
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            model_id = model_ids[i] if i < len(model_ids) else "unknown"
            processed_results.append({
                "model_id": model_id,
                "error": f"Unexpected error: {str(result)}",
                "meta": {
                    "model_id": model_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            })
        else:
            processed_results.append(result)
    
    return {"question": question, "responses": processed_results, "prompt_id": prompt_id, "prompt_version": prompt_version}
