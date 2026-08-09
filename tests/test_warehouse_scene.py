from pathlib import Path

import mujoco
import numpy as np

from warehouse.scenario import WarehouseScenario

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = PROJECT_ROOT / "robot" / "franka_fr3" / "warehouse_scene.xml"
SCENARIO_PATH = (
    PROJECT_ROOT / "warehouse" / "configs" / "warehouse_sorting_minimal.json"
)


def test_minimal_warehouse_scene_matches_scenario():
    scenario = WarehouseScenario.load(SCENARIO_PATH)
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))

    assert scenario.object_ids == ("cube_red", "cube_blue")

    for object_id in scenario.object_ids:
        assert _body_id(model, object_id) >= 0

    marker_xy = []
    for destination in scenario.destinations.values():
        body_id = _body_id(model, destination["body_name"])
        marker_xy.append(model.body_pos[body_id, :2])

        assert np.allclose(model.body_pos[body_id, :2], destination["place_xy"])

        geom_ids = np.flatnonzero(model.geom_bodyid == body_id)
        assert len(geom_ids) == 1
        assert np.all(model.geom_contype[geom_ids] == 0)
        assert np.all(model.geom_conaffinity[geom_ids] == 0)

    assert np.linalg.norm(marker_xy[0] - marker_xy[1]) >= 0.14


def _body_id(model, body_name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
