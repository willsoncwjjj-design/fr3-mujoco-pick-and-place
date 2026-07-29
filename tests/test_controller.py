from types import SimpleNamespace

import numpy as np

from control.controller import Controller


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
    assert not make_controller(None)._verify_grasp_success()


def test_verification_requires_object_in_place_zone():
    assert make_controller([0.31, -0.145, 0.85])._verify_grasp_success()
    assert not make_controller([0.45, 0.10, 0.85])._verify_grasp_success()
