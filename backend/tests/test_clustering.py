from backend.services.semantic import cluster_points, embed_points, extract_points


def test_clustering_pairs_grouped():
    structured = [
        {
            "model_id": "m1",
            "parsed": {
                "summary_points": [
                    {"id": "a1", "text": "Apple is a fruit", "confidence": "high"},
                    {"id": "b1", "text": "Banana is yellow", "confidence": "high"},
                    {"id": "c1", "text": "Cats chase mice", "confidence": "high"},
                ]
            },
        },
        {
            "model_id": "m2",
            "parsed": {
                "summary_points": [
                    {"id": "a2", "text": "Apples are fruits", "confidence": "high"},
                    {"id": "b2", "text": "Bananas have yellow peel", "confidence": "high"},
                    {"id": "c2", "text": "Felines hunt rodents", "confidence": "high"},
                ]
            },
        },
    ]

    points = extract_points(structured)
    embeddings = embed_points(points)
    clusters = cluster_points(points, embeddings, threshold=0.5)

    # Expect <= 3 clusters for 3 semantic pairs
    assert len(clusters) <= 3

    # Ensure all points are assigned
    flat_ids = {pid for cluster in clusters for pid in cluster}
    assert flat_ids == {p["id"] for p in points}
