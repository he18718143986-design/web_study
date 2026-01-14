import pytest

from backend.services.iteration_controller import run_iterations


@pytest.mark.asyncio
async def test_end_to_end_iteration_converges_or_stops():
    result = await run_iterations("Is X true or false?", ["mock", "mock"], max_rounds=3)
    assert result["state"] in {"converged", "max_rounds_reached"}
    assert result["rounds"], "Rounds should not be empty"
    assert len(result["rounds"]) <= 3
