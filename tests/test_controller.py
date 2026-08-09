from types import SimpleNamespace

import numpy as np
import pytest

from control.controller import Controller, RobotState


class CameraStub:
    def capture(self):
        return None, np.ones((2, 2)), np.zeros((2, 2, 2)), {}


class DetectorStub:
    def detect_sim(self, segmentation, catalog):
        return [{"body_id": 7}]


class LocalizerStub:
    def __init__(self, position):
        self.position = position

    def localize(self, detections, depth, camera_pose):
        if self.position is None:
            return []
        return [{"body_id": 7, "position": np.asarray(self.position)}]


class GraspPlannerSpy:
    def __init__(self):
        self.place_xy = np.array([0.3, -0.15])
        self.received_place_xy = None

    def compute_grasp_pose(self, object_position, place_xy=None):
        self.received_place_xy = np.asarray(place_xy).copy()
        return {"pose": "stub"}

    def check_ik_feasible(self, grasp_pose):
        return True


class TrajectoryPlannerStub:
    def plan(self, grasp_pose):
        return [np.zeros(7)], [None]


def make_controller(localized_position):
    controller = Controller(
        object_catalog=[{"body_id": 7}],
        env=None,
        robot=None,
        camera=CameraStub(),
        detector=DetectorStub(),
        localizer=LocalizerStub(localized_position),
        grasp_planner=SimpleNamespace(place_xy=np.array([0.3, -0.15])),
        trajectory_planner=None,
        gripper=None,
    )
    controller.target_object = {"body_id": 7}
    controller.object_position = np.array([0.3, 0.15, 0.88])
    return controller


def test_verification_rejects_missing_target():
    controller = make_controller(None)

    assert not controller._verify_grasp_success()
    assert controller.verification_metrics == {
        "target_visible": False,
        "moved_distance_m": None,
        "place_error_m": None,
        "minimum_movement_m": 0.02,
        "maximum_place_error_m": 0.05,
    }


def test_verification_requires_object_in_place_zone():
    assert make_controller([0.31, -0.145, 0.85])._verify_grasp_success()
    assert not make_controller([0.45, 0.10, 0.85])._verify_grasp_success()


def test_run_cycle_passes_external_place_target(monkeypatch):
    controller = make_controller([0.3, 0.15, 0.88])
    planner = GraspPlannerSpy()
    controller.grasp_planner = planner
    controller.trajectory_planner = TrajectoryPlannerStub()
    external_place_xy = np.array([0.46, -0.18])

    monkeypatch.setattr(controller, "_perceive", lambda: None)
    monkeypatch.setattr(controller, "_localize", lambda *args: None)
    monkeypatch.setattr(controller, "_check_reachability", lambda: None)
    monkeypatch.setattr(controller, "_execute", lambda: None)
    monkeypatch.setattr(controller, "_verify", lambda: None)

    result = controller.run_cycle(
        target_body_id=7,
        place_xy=external_place_xy,
    )

    assert result["success"]
    assert np.allclose(planner.received_place_xy, external_place_xy)
    assert result["failure_code"] is None
    assert result["failed_state"] is None


def test_verification_uses_current_cycle_place_target():
    controller = make_controller([0.465, -0.175, 0.85])
    controller.target_place_xy = np.array([0.46, -0.18])

    assert controller._verify_grasp_success()
    assert controller.verification_metrics["target_visible"]
    assert controller.verification_metrics["moved_distance_m"] == (
        np.linalg.norm(np.array([0.465, -0.175]) - np.array([0.3, 0.15]))
    )
    assert controller.verification_metrics["place_error_m"] == (
        np.linalg.norm(np.array([0.465, -0.175]) - np.array([0.46, -0.18]))
    )
    assert controller.verification_metrics["minimum_movement_m"] == 0.02
    assert controller.verification_metrics["maximum_place_error_m"] == 0.05


def test_run_cycle_rejects_invalid_place_target():
    controller = make_controller([0.3, 0.15, 0.88])
    controller.state = RobotState.DONE

    result = controller.run_cycle(place_xy=[0.3])

    assert not result["success"]
    assert result["error_message"] == "place_xy must contain two finite values"
    assert result["failure_code"] == "invalid_request"
    assert result["failed_state"] == "IDLE"


def test_run_cycle_classifies_localization_failure():
    controller = make_controller(None)

    result = controller.run_cycle()

    assert not result["success"]
    assert result["failure_code"] == "object_missing"
    assert result["failed_state"] == "LOCALIZE"
    assert result["error_message"] == "No objects localized"


def test_execute_joint_target_uses_servo_and_fixed_control_steps():
    controller = make_controller([0.3, 0.15, 0.88])
    robot = SimpleNamespace(
        joint_limits=[(-1.0, 1.0)] * 7,
        set_ctrl=lambda values: setattr(robot, "target", np.asarray(values)),
    )
    gripper = SimpleNamespace(
        execute=lambda command: setattr(gripper, "command", command)
    )
    env = SimpleNamespace(step_count=0)
    env.step = lambda: setattr(env, "step_count", env.step_count + 1)
    controller.robot = robot
    controller.gripper = gripper
    controller.env = env

    result = controller.execute_joint_target(
        [0.1] * 7,
        gripper_command="close",
        control_steps=4,
    )

    assert result["success"]
    assert robot.target == pytest.approx([0.1] * 7)
    assert gripper.command == "close"
    assert env.step_count == 4


def test_execute_joint_target_rejects_unsafe_command():
    controller = make_controller([0.3, 0.15, 0.88])
    controller.robot = SimpleNamespace(joint_limits=[(-1.0, 1.0)] * 7)

    with pytest.raises(ValueError, match="joint limits"):
        controller.execute_joint_target([2.0] * 7)


@pytest.mark.parametrize(
    ("failed_state", "expected_code"),
    [
        (RobotState.IDLE, "invalid_request"),
        (RobotState.PERCEIVE, "object_missing"),
        (RobotState.LOCALIZE, "object_missing"),
        (RobotState.PLAN, "ik_failed"),
        (RobotState.CHECK_REACHABILITY, "ik_failed"),
        (RobotState.EXECUTE, "pick_failed"),
        (RobotState.VERIFY, "verification_failed"),
        (RobotState.DONE, "internal_error"),
    ],
)
def test_failure_code_depends_on_state_not_error_text(
    failed_state,
    expected_code,
):
    controller = make_controller([0.3, 0.15, 0.88])
    controller.state = failed_state

    result = controller._fail("same diagnostic message")

    assert result["failure_code"] == expected_code
    assert result["failed_state"] == failed_state.name
