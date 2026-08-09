import pytest

from vla_runtime.contracts import ActionChunk, DeltaEEAction
from vla_runtime.manager import ActionManager


def make_chunk(chunk_id, observation_id, action_count=2):
    action = DeltaEEAction((0.005, 0.0, 0.0), (0.0, 0.0, 0.0))
    return ActionChunk(
        chunk_id=chunk_id,
        source_observation_id=observation_id,
        actions=(action,) * action_count,
        action_dt_s=0.02,
        created_at_s=float(observation_id),
        policy_version="scripted-v1",
    )


def test_manager_consumes_actions_in_order():
    manager = ActionManager()
    manager.update_observation(1)
    assert manager.submit(make_chunk("chunk-1", 1)).accepted

    first = manager.dispatch_next()
    manager.complete_action(first)
    second = manager.dispatch_next()
    manager.complete_action(second)

    assert (first.action_index, second.action_index) == (0, 1)
    assert manager.dispatch_next() is None
    assert manager.events[-1].event_type == "chunk_completed"


def test_manager_rejects_stale_chunk():
    manager = ActionManager()
    manager.update_observation(2)

    submission = manager.submit(make_chunk("stale", 1))

    assert not submission.accepted
    assert submission.reason == "stale_observation"
    assert manager.dispatch_next() is None


def test_manager_preempts_only_after_inflight_action_finishes():
    manager = ActionManager()
    manager.update_observation(1)
    manager.submit(make_chunk("old", 1, action_count=3))
    inflight = manager.dispatch_next()

    manager.update_observation(2)
    pending = manager.submit(make_chunk("new", 2, action_count=2))

    assert pending.accepted
    assert pending.disposition == "pending"
    assert manager.inflight == inflight
    manager.complete_action(inflight)

    replacement = manager.dispatch_next()
    assert replacement.chunk_id == "new"
    assert replacement.action_index == 0
    cancellation = next(
        event
        for event in manager.events
        if event.event_type == "chunk_cancelled" and event.chunk_id == "old"
    )
    assert cancellation.remaining_actions == 2
    assert cancellation.reason == "new_observation:2"


def test_manager_rejects_second_chunk_from_same_observation():
    manager = ActionManager()
    manager.update_observation(1)
    manager.submit(make_chunk("first", 1))

    submission = manager.submit(make_chunk("duplicate", 1))

    assert not submission.accepted
    assert submission.reason == "not_newer_than_current_chunk"


def test_manager_requires_monotonic_observation_ids():
    manager = ActionManager()
    manager.update_observation(2)

    with pytest.raises(ValueError, match="monotonic"):
        manager.update_observation(1)
