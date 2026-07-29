import numpy as np

from planning.grasp_planner import GraspPlanner
from planning.trajectory_planner import TrajectoryPlanner
from robot.kinematics import DLSIKSolver


def test_trajectory_has_expected_commands(model, data):
    solver = DLSIKSolver(model, data, "attachment_site", 0.1)
    grasp_planner = GraspPlanner(model, data, ik_solver=solver)
    pose = grasp_planner.compute_grasp_pose(np.array([0.3, 0.15, 0.88]))

    trajectory, commands = TrajectoryPlanner(solver).plan(
        pose, steps_per_segment=10
    )

    assert trajectory is not None
    assert len(trajectory) == len(commands) == 50
    assert commands[19] == "close"
    assert commands[49] == "open"
    assert sum(command is not None for command in commands) == 2
