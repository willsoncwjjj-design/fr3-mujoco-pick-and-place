import time

import numpy as np

from vla_runtime.contracts import DeltaEEAction, RGBFrame, VLAObservation
from vla_runtime.policies import AsyncPolicyWorker, ScriptedPolicy


def make_observation(observation_id):
    return VLAObservation(
        observation_id=observation_id,
        timestamp_s=float(observation_id),
        frames=(RGBFrame("top_cam", np.zeros((2, 2, 3), dtype=np.uint8)),),
        qpos=(0.0,) * 7,
        qvel=(0.0,) * 7,
        ee_position=(0.4, 0.0, 0.5),
        ee_quaternion=(1.0, 0.0, 0.0, 0.0),
        gripper_position=0.04,
        instruction="移动机械臂",
        subgoal="向上移动",
    )


def poll_until_complete(worker, timeout_s=0.5):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        results = worker.poll()
        if results:
            return results
        time.sleep(0.005)
    raise AssertionError("异步策略未在超时时间内返回")


def test_async_policy_exposes_latency_without_blocking_request():
    action = DeltaEEAction((0.0, 0.0, 0.005), (0.0, 0.0, 0.0))
    policy = ScriptedPolicy({1: (action,)}, latency_s=0.03)

    with AsyncPolicyWorker(policy) as worker:
        request_id = worker.request(make_observation(1))
        assert worker.pending_count == 1
        assert worker.poll() == ()
        result = poll_until_complete(worker)[0]

    assert result.request_id == request_id
    assert result.observation_id == 1
    assert result.chunk.source_observation_id == 1
    assert result.latency_s >= 0.02


def test_async_policy_returns_structured_error():
    with AsyncPolicyWorker(ScriptedPolicy({})) as worker:
        worker.request(make_observation(2))
        result = poll_until_complete(worker)[0]

    assert result.chunk is None
    assert "No scripted actions for observation 2" in result.error_message
