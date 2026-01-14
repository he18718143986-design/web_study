from typing import Any, Dict, List

from backend.services.cross_eval import cross_evaluate
from backend.services.nli import simple_nli
from backend.services.semantic import cluster_points, embed_points, extract_points


def _build_point_lookup(points: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {p["id"]: p for p in points}


def _nli_matrix(clusters: List[List[str]], point_lookup: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    nli_results: List[Dict[str, Any]] = []
    for cluster in clusters:
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                a_id, b_id = cluster[i], cluster[j]
                a = point_lookup.get(a_id)
                b = point_lookup.get(b_id)
                if not a or not b:
                    continue
                label = simple_nli(a.get("text", ""), b.get("text", ""))
                nli_results.append(
                    {
                        "cluster_id": str(cluster),
                        "a": {"id": a_id, "model_id": a.get("model_id")},
                        "b": {"id": b_id, "model_id": b.get("model_id")},
                        "label": label,
                    }
                )
    return nli_results


def _summarize(clusters: List[List[str]], nli_results: List[Dict[str, Any]], cross_results: List[Dict[str, Any]], point_lookup: Dict[str, Dict[str, Any]]):
    contradictions: List[Dict[str, Any]] = []
    confirmed: List[Dict[str, Any]] = []
    followups: List[str] = []

    cluster_has_contradiction = set()
    for item in nli_results:
        if item["label"] == "contradiction":
            cluster_has_contradiction.add(item["cluster_id"])

    for cluster in clusters:
        cid = str(cluster)
        texts = [point_lookup[p]["text"] for p in cluster if p in point_lookup]
        models = {point_lookup[p]["model_id"] for p in cluster if p in point_lookup}
        if cid in cluster_has_contradiction:
            contradictions.append(
                {
                    "cluster_id": cid,
                    "points": [{"id": p, "model_id": point_lookup[p]["model_id"], "text": point_lookup[p]["text"]} for p in cluster if p in point_lookup],
                    "reason": "Heuristic NLI detected contradiction",
                }
            )
            followups.append(f"Clarify conflict for cluster {cid}: {texts}")
        else:
            confirmed.append(
                {
                    "cluster_id": cid,
                    "points": [{"id": p, "model_id": point_lookup[p]["model_id"], "text": point_lookup[p]["text"]} for p in cluster if p in point_lookup],
                    "models": sorted(models),
                }
            )

    recommendation = "run_rag" if contradictions else "continue_rounds"
    return {
        "confirmed": confirmed,
        "contradictions": contradictions,
        "followups": followups,
        "recommendation": recommendation,
        "cross_eval": cross_results,
        "nli": nli_results,
    }


def aggregate_structured_responses(structured: List[Dict[str, Any]]):
    points = extract_points(structured)
    embeddings = embed_points(points)
    clusters = cluster_points(points, embeddings, threshold=0.5)
    lookup = _build_point_lookup(points)
    nli_results = _nli_matrix(clusters, lookup)
    cross_results = cross_evaluate(clusters, lookup)
    return _summarize(clusters, nli_results, cross_results, lookup)
