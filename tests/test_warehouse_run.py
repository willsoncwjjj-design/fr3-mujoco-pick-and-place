import sys
from types import SimpleNamespace

import warehouse_run
from warehouse_run import build_single_request


def test_build_single_request_uses_minimal_scenario():
    request = build_single_request("cube_red")

    assert request.task_id == "warehouse_sorting_minimal_v1"
    assert request.object_id == "cube_red"
    assert request.destination_id == "priority_bin.slot_1"
    assert request.place_xy == (0.30, -0.18)
    assert request.scene_path == "robot/franka_fr3/warehouse_scene.xml"


def test_build_single_request_supports_blue_object():
    request = build_single_request("cube_blue")

    assert request.object_id == "cube_blue"
    assert request.destination_id == "standard_bin.slot_1"
    assert request.place_xy == (0.46, -0.18)


def test_cli_accepts_blue_object(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["warehouse_run.py", "--object", "cube_blue"],
    )

    args = warehouse_run.parse_args()

    assert args.object == "cube_blue"


def test_run_single_exposes_controller_verification(monkeypatch):
    verification = {
        "target_visible": True,
        "moved_distance_m": 0.33,
        "place_error_m": 0.01,
        "minimum_movement_m": 0.02,
        "maximum_place_error_m": 0.05,
    }

    def fake_run(**kwargs):
        assert kwargs == {
            "target_name": "cube_red",
            "use_viewer": False,
            "scene_path": "robot/franka_fr3/warehouse_scene.xml",
            "place_xy": (0.30, -0.18),
        }
        return {
            "success": True,
            "state": SimpleNamespace(name="DONE"),
            "failure_code": None,
            "failed_state": None,
            "verification": verification,
        }

    monkeypatch.setattr(warehouse_run, "run", fake_run)

    result = warehouse_run.run_single("cube_red")

    assert result["verification"] == verification
    assert result["failure_code"] is None
    assert result["failed_state"] is None
