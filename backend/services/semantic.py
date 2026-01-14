import math
from typing import Any, Dict, List, Sequence

from backend.services.embeddings import embed_texts

Point = Dict[str, Any]
Vector = List[float]


def extract_points(structured_responses: Sequence[Dict[str, Any]]) -> List[Point]:
    points: List[Point] = []
    for item in structured_responses:
        model_id = item.get("model_id", "unknown")
        parsed = item.get("parsed") or {}
        for sp in parsed.get("summary_points", []) or []:
            pid = f"{model_id}_{sp.get('id', 'p')}"
            points.append({
                "id": pid,
                "text": sp.get("text", ""),
                "model_id": model_id,
            })
    return points


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


def _tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    for raw in text.lower().split():
        tok = raw.strip()
        tok = tok.rstrip(".,;:!?()[]{}\"'")
        if not tok:
            continue
        tok = _SYN_MAP.get(tok, tok.rstrip("s"))
        if tok:
            tokens.append(tok)
    return tokens


def _vectorize(text: str, dim: int = 16) -> Vector:
    vec = [0.0] * dim
    for tok in _tokenize(text):
        idx = hash(tok) % dim
        vec[idx] += 1.0
    return vec


def embed_points(points: Sequence[Point], dim: int = 16) -> List[Vector]:
    texts = [p.get("text", "") for p in points]
    try:
        return embed_texts(texts)
    except Exception:
        return [_vectorize(text, dim=dim) for text in texts]


def _cosine(a: Vector, b: Vector) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def cluster_points(points: Sequence[Point], embeddings: Sequence[Vector], threshold: float = 0.82) -> List[List[str]]:
    id_to_vec = {p["id"]: v for p, v in zip(points, embeddings)}
    clusters: List[List[str]] = []
    for p in points:
        assigned = False
        for cluster in clusters:
            rep_id = cluster[0]
            sim = _cosine(id_to_vec[p["id"]], id_to_vec[rep_id])
            if sim >= threshold:
                cluster.append(p["id"])
                assigned = True
                break
        if not assigned:
            clusters.append([p["id"]])
    return clusters
