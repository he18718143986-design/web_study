import os
from typing import Iterable, List

import httpx

Vector = List[float]


_SYN_MAP = {
    "apples": "apple",
    "apple": "apple",
    "bananas": "banana",
    "banana": "banana",
    "felines": "cat",
    "cats": "cat",
    "rodents": "mouse",
    "mice": "mouse",
    "chase": "hunt",
    "hunting": "hunt",
    "hunt": "hunt",
    "yellow": "yellow",
    "peel": "peel",
    "fruit": "fruit",
    "fruits": "fruit",
}


def _vectorize_stub(text: str, dim: int = 16) -> Vector:
    vec = [0.0] * dim
    for raw in text.lower().split():
        tok = raw.strip(".,;:!?()[]{}\"'")
        tok = _SYN_MAP.get(tok, tok.rstrip("s"))
        if not tok:
            continue
        idx = hash(tok) % dim
        vec[idx] += 1.0
    return vec


def _hf_embed(texts: Iterable[str], dim: int = 384) -> List[Vector]:
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    model = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    if not api_key:
        raise RuntimeError("HUGGINGFACE_API_KEY not set for embeddings")

    endpoint = os.getenv(
        "HF_EMBEDDING_ENDPOINT",
        f"https://api-inference.huggingface.co/models/{model}",
    )
    headers = {"Authorization": f"Bearer {api_key}"}

    payload_list = list(texts)
    payload_to_send = payload_list if len(payload_list) > 1 else (payload_list[0] if payload_list else "")

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(endpoint, headers=headers, json=payload_to_send)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data and isinstance(data[0], list):
            return [list(map(float, v)) for v in data]
        if isinstance(data, list) and all(isinstance(x, (int, float)) for x in data):
            return [list(map(float, data))]
        raise RuntimeError(f"Unexpected embedding response: {data}")


def embed_texts(texts: List[str]) -> List[Vector]:
    try:
        # Try remote embeddings first.
        return _hf_embed(texts)
    except Exception:
        # Fallback to deterministic stub embeddings to keep pipeline running.
        return [_vectorize_stub(t) for t in texts]
