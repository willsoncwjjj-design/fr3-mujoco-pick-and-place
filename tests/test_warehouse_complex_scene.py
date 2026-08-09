from pathlib import Path

import mujoco
import numpy as np

from warehouse.scenario import WarehouseScenario

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = (
    PROJECT_ROOT / "robot" / "franka_fr3" / "warehouse_complex_scene.xml"
)
SCENARIO_PATH = (
    PROJECT_ROOT / "warehouse" / "configs" / "warehouse_sorting_complex.json"
)


def test_complex_scene_starts_with_blue_blocking_red_destination():
    scenario = WarehouseScenario.load(SCENARIO_PATH)
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    assert scenario.buffer_destination_ids == ("buffer.slot_1",)
    blue_xy = data.body("cube_blue").xpos[:2]
    red_xy = data.body("cube_red").xpos[:2]
    priority_xy = scenario.destinations["priority_bin.slot_1"]["place_xy"]
    assert np.linalg.norm(blue_xy - priority_xy) < 0.01
    assert np.linalg.norm(red_xy - priority_xy) > 0.20

    for destination_id, destination in scenario.destinations.items():
        marker_xy = data.body(destination["body_name"]).xpos[:2]
        assert np.allclose(marker_xy, destination["place_xy"]), destination_id
        body_id = model.body(destination["body_name"]).id
        geom_ids = np.flatnonzero(model.geom_bodyid == body_id)
        assert np.all(model.geom_contype[geom_ids] == 0)
        assert np.all(model.geom_conaffinity[geom_ids] == 0)
