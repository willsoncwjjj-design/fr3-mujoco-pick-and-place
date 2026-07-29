import pytest

from main import run


@pytest.mark.integration
def test_baseline_pick_and_place():
    result = run(target_name="cube_red", use_viewer=False)
    assert result["success"], result
