import os
from typing import Literal

import httpx

Label = Literal["entailment", "neutral", "contradiction"]

_NEG_WORDS = {"not", "no", "never", "none", "nobody", "nothing", "n't"}
_ANTONYM_PAIRS = {
    ("true", "false"),
    ("yes", "no"),
    ("up", "down"),
    ("increase", "decrease"),
    ("good", "bad"),
}


def _normalize(text: str) -> set[str]:
    tokens = set()
    for raw in text.lower().replace(".", " ").split():
        tok = raw.strip(".,;:!?()[]{}\"'")
        if tok:
            tokens.add(tok)
    return tokens


def _hf_nli(a: str, b: str) -> Label:
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    model = os.getenv("HF_NLI_MODEL", "facebook/bart-large-mnli")
    if not api_key:
        raise RuntimeError("HUGGINGFACE_API_KEY not set for NLI")

    endpoint = os.getenv("HF_NLI_ENDPOINT", f"https://api-inference.huggingface.co/models/{model}")
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"inputs": {"premise": a, "hypothesis": b}}
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(endpoint, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
    # Response often is list of list of dicts with labels and scores
    candidates = None
    if isinstance(data, list) and data and isinstance(data[0], list):
        candidates = data[0]
    elif isinstance(data, list):
        candidates = data
    else:
        raise RuntimeError(f"Unexpected NLI response: {data}")

    best = max(candidates, key=lambda x: x.get("score", 0.0))
    label = best.get("label", "neutral").lower()
    score = float(best.get("score", 0.0))
    threshold = float(os.getenv("NLI_THRESHOLD", "0.6"))
    if score < threshold:
        return "neutral"
    if "entailment" in label:
        return "entailment"
    if "contradiction" in label or "refute" in label:
        return "contradiction"
    return "neutral"


def simple_nli(a: str, b: str) -> Label:
    try:
        return _hf_nli(a, b)
    except Exception:
        ta = _normalize(a)
        tb = _normalize(b)
        if _NEG_WORDS & ta and not _NEG_WORDS & tb:
            return "contradiction"
        if _NEG_WORDS & tb and not _NEG_WORDS & ta:
            return "contradiction"
        for x, y in _ANTONYM_PAIRS:
            if (x in ta and y in tb) or (y in ta and x in tb):
                return "contradiction"
        overlap = ta & tb
        if overlap:
            return "entailment"
        return "neutral"
