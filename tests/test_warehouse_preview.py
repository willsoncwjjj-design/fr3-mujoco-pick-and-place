from pathlib import Path

import mujoco

from warehouse_preview import build_preview

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = PROJECT_ROOT / "robot" / "franka_fr3" / "warehouse_scene.xml"


def test_preview_compiles_real_scene_into_control_cycles():
    preview = build_preview()
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))

    assert preview["mode"] == "dry_run"
    assert preview["task_id"] == "warehouse_sorting_minimal_v1"
    assert preview["scene"] == "robot/franka_fr3/warehouse_scene.xml"

    cycles = preview["cycles"]
    assert [cycle["object_id"] for cycle in cycles] == [
        "cube_red",
        "cube_blue",
    ]
    assert [cycle["target_body_id"] for cycle in cycles] == [
        _body_id(model, "cube_red"),
        _body_id(model, "cube_blue"),
    ]
    assert [cycle["place_xy"] for cycle in cycles] == [
        (0.30, -0.18),
        (0.46, -0.18),
    ]


def _body_id(model, body_name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
