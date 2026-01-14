import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from backend.services.embeddings import embed_texts, Vector

STORE_DIR = Path(__file__).resolve().parent / "data"
STORE_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_PATH = STORE_DIR / "vector_index.json"


def _load_index() -> List[Dict[str, Any]]:
    if not VECTOR_PATH.exists():
        return []
    try:
        return json.loads(VECTOR_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _save_index(items: List[Dict[str, Any]]) -> None:
    VECTOR_PATH.write_text(json.dumps(items), encoding="utf-8")


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def add_documents(session_id: str, docs: List[Dict[str, Any]]) -> None:
    """docs: list of {text: str, meta: {...}}"""
    texts = [d.get("text", "") for d in docs]
    embeddings = embed_texts(texts)
    items = _load_index()
    for doc, emb in zip(docs, embeddings):
        items.append({
            "session_id": session_id,
            "text": doc.get("text", ""),
            "embedding": emb,
            "meta": doc.get("meta", {}),
        })
    _save_index(items)


def search_similar(query: str, top_k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
    index = _load_index()
    if not index:
        return []
    query_vecs = embed_texts([query])
    qvec = query_vecs[0] if query_vecs else []
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for item in index:
        emb = item.get("embedding") or []
        score = _cosine(qvec, emb)
        scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]
