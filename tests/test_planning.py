import numpy as np
import pytest

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


def test_grasp_planner_accepts_external_place_target(model, data):
    solver = DLSIKSolver(model, data, "attachment_site", 0.1)
    grasp_planner = GraspPlanner(model, data, ik_solver=solver)
    object_position = np.array([0.3, 0.15, 0.88])
    external_place_xy = np.array([0.46, -0.18])

    pose = grasp_planner.compute_grasp_pose(
        object_position,
        place_xy=external_place_xy,
    )

    assert np.allclose(pose["place_pre"][:2], external_place_xy)
    assert np.allclose(pose["place"][:2], external_place_xy)
    assert np.allclose(grasp_planner.place_xy, [0.3, -0.15])


@pytest.mark.parametrize(
    "place_xy",
    ([0.3], [0.3, -0.15, 0.0], [np.nan, -0.15]),
)
def test_grasp_planner_rejects_invalid_place_target(model, data, place_xy):
    planner = GraspPlanner(model, data)

    with pytest.raises(ValueError, match="two finite values"):
        planner.compute_grasp_pose([0.3, 0.15, 0.88], place_xy=place_xy)
