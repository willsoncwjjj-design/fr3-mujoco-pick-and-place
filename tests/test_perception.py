import mujoco
import numpy as np

from perception.detector import Detector
from perception.localizer import ObjectLocalizer


def test_detector_maps_geom_to_catalog_body(model):
    body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "cube_red"
    )
    geom_id = next(
        index for index, owner in enumerate(model.geom_bodyid) if owner == body_id
    )
    segmentation = np.full((8, 8, 2), -1, dtype=int)
    segmentation[2:6, 3:7, 0] = geom_id
    catalog = [{"body_id": body_id, "class_name": "cube_red"}]

    detections = Detector(model).detect_sim(segmentation, catalog)

    assert len(detections) == 1
    assert detections[0]["body_id"] == body_id
    assert detections[0]["bbox"] == (3, 2, 6, 5)


def test_localizer_converts_camera_axes_to_world():
    localizer = ObjectLocalizer(
        np.array([[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]])
    )
    depth = np.ones((5, 5))
    detection = {"bbox": (2, 2, 2, 2), "class_name": "cube", "body_id": 1}
    camera_pose = {"position": np.array([0.0, 0.0, 2.0]), "rotation": np.eye(3)}

    result = localizer.localize([detection], depth, camera_pose)

    assert len(result) == 1
    assert np.allclose(result[0]["position"], [0.0, 0.0, 1.0])
