from backend.services.aggregator import aggregate_structured_responses

def test_aggregator_detects_contradiction():
    structured = [
        {
            "model_id": "m1",
            "parsed": {
                "summary_points": [
                    {"id": "p1", "text": "The sky is blue", "confidence": "high"},
                    {"id": "p2", "text": "Cats are mammals", "confidence": "high"},
                ]
            },
        },
        {
            "model_id": "m2",
            "parsed": {
                "summary_points": [
                    {"id": "p3", "text": "The sky is not blue", "confidence": "high"},
                    {"id": "p4", "text": "Cats are mammals", "confidence": "high"},
                ]
            },
        },
    ]

    report = aggregate_structured_responses(structured)
    assert report["contradictions"], "Should contain at least one contradiction"
    contradiction_clusters = {c["cluster_id"] for c in report["contradictions"]}
    assert contradiction_clusters, "Contradiction clusters should be present"
    assert any("followups" for _ in [report]), "Followups should be suggested"
