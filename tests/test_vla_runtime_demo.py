import pytest

from vla_runtime_demo import run_demo


@pytest.mark.integration
def test_vla_runtime_demo_preempts_stale_actions():
    result = run_demo(use_viewer=False)

    assert result["success"]
    assert result["executed_actions"] == 3
    assert result["preempted_actions"] == 2
    event_types = [event["event_type"] for event in result["events"]]
    assert event_types == [
        "chunk_activated",
        "action_dispatched",
        "action_completed",
        "chunk_cancelled",
        "chunk_activated",
        "action_dispatched",
        "action_completed",
        "action_dispatched",
        "action_completed",
        "chunk_completed",
    ]
