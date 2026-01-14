from typing import Any, Dict, List

from backend.prompt.registry import get_prompt
from backend.services.nli import simple_nli


def _judge_pair(a_text: str, b_text: str) -> Dict[str, Any]:
    label = simple_nli(a_text, b_text)
    if label == "contradiction":
        judgement = "disagree"
    elif label == "entailment":
        judgement = "agree"
    else:
        judgement = "uncertain"
    return {
        "judgement": judgement,
        "reason": f"Heuristic NLI -> {label}",
        "confidence": "medium",
    }


def cross_evaluate(clusters: List[List[str]], point_lookup: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for cluster in clusters:
        # Only consider pairs from different models
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                a_id = cluster[i]
                b_id = cluster[j]
                a = point_lookup.get(a_id)
                b = point_lookup.get(b_id)
                if not a or not b:
                    continue
                if a.get("model_id") == b.get("model_id"):
                    continue
                eval_result = _judge_pair(a.get("text", ""), b.get("text", ""))
                results.append(
                    {
                        "cluster_id": str(cluster),
                        "a": {"id": a_id, "model_id": a.get("model_id")},
                        "b": {"id": b_id, "model_id": b.get("model_id")},
                        **eval_result,
                        "prompt_used": bool(get_prompt("peerreviewer_v1")),
                    }
                )
    return results
