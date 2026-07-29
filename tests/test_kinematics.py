import numpy as np

from robot.kinematics import DLSIKSolver
from robot.panda import PandaRobot


def test_ik_uses_private_state(model, data):
    robot = PandaRobot(model, data)
    solver = DLSIKSolver(model, data, "attachment_site", 0.1)
    target_position, target_quaternion = robot.get_end_effector_pose()
    state_before = data.qpos.copy()

    solution = solver.solve(target_position, target_quaternion)

    assert solution is not None
    assert np.allclose(data.qpos, state_before)


def test_known_grasp_pose_is_reachable(model, data):
    solver = DLSIKSolver(model, data, "attachment_site", 0.1)
    solution = solver.solve(
        np.array([0.3, 0.15, 0.90]),
        np.array([0.0, 1.0, 0.0, 0.0]),
    )

    assert solution is not None
    limits = model.jnt_range[solver.arm_joint_ids]
    assert np.all(solution >= limits[:, 0])
    assert np.all(solution <= limits[:, 1])
