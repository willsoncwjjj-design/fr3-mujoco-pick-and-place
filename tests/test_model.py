import mujoco
import numpy as np

from robot.gripper import PandaGripper
from robot.panda import PandaRobot


def test_model_contains_required_entities(model, data):
    robot = PandaRobot(model, data)
    gripper = PandaGripper(model, data)

    assert len(robot.arm_joint_ids) == 7
    assert len(robot.arm_act_ids) == 7
    assert len(gripper.act_ids) == 2
    assert mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, "top_cam"
    ) >= 0


def test_joint_position_validation(model, data):
    robot = PandaRobot(model, data)

    robot.set_ctrl(np.zeros(7))
    assert np.allclose(data.ctrl[robot.arm_act_ids], 0.0)

    try:
        robot.set_ctrl(np.zeros(6))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid joint vector to be rejected")
