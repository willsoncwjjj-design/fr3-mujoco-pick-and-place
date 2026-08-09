import numpy as np
import pytest

from vla_runtime.contracts import (
    ActionChunk,
    DeltaEEAction,
    RGBFrame,
    VLAObservation,
)


def make_observation(observation_id=1):
    return VLAObservation(
        observation_id=observation_id,
        timestamp_s=1.25,
        frames=(RGBFrame("top_cam", np.zeros((4, 5, 3), dtype=np.uint8)),),
        qpos=(0.0,) * 7,
        qvel=(0.0,) * 7,
        ee_position=(0.4, 0.0, 0.5),
        ee_quaternion=(2.0, 0.0, 0.0, 0.0),
        gripper_position=0.04,
        instruction="把红色方块移动到目标区",
        subgoal="移动到方块上方",
    )


def test_rgb_frame_owns_read_only_copy():
    source = np.zeros((2, 3, 3), dtype=np.uint8)
    frame = RGBFrame("top_cam", source)

    source[0, 0, 0] = 255

    assert frame.image[0, 0, 0] == 0
    assert not frame.image.flags.writeable


def test_observation_normalizes_quaternion():
    observation = make_observation()

    assert observation.ee_quaternion == pytest.approx((1.0, 0.0, 0.0, 0.0))


@pytest.mark.parametrize(
    "action",
    [
        DeltaEEAction((0.01, 0.0, 0.0), (0.0, 0.0, 0.0)),
        DeltaEEAction((0.0, 0.0, 0.0), (0.0, 0.0, 0.1), "close"),
    ],
)
def test_action_chunk_preserves_action_order(action):
    chunk = ActionChunk(
        chunk_id="chunk-1",
        source_observation_id=1,
        actions=(action,),
        action_dt_s=0.02,
        created_at_s=1.5,
        policy_version="scripted-v1",
    )

    assert chunk.actions == (action,)


def test_invalid_action_and_chunk_are_rejected():
    with pytest.raises(ValueError, match="gripper_command"):
        DeltaEEAction((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), "hold")

    with pytest.raises(ValueError, match="at least one action"):
        ActionChunk("chunk-1", 1, (), 0.02, 1.0, "scripted-v1")
