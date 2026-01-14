import pytest

from backend.services.orchestrator import multi_model_query


@pytest.mark.asyncio
async def test_mock_structured_parse_success_rate():
    runs = 5
    successes = 0
    for _ in range(runs):
        result = await multi_model_query("test question", ["mock"], structured=True)
        assert "responses" in result
        resp = result["responses"][0]
        if "parsed" in resp and not resp.get("parse_error"):
            successes += 1
    assert successes >= int(0.9 * runs), f"parse success {successes}/{runs} below threshold"