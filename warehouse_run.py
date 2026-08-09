import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from main import run
from warehouse.scenario import WarehouseScenario

SCENE_PATH = "robot/franka_fr3/warehouse_scene.xml"
SCENARIO_PATH = (
    Path(__file__).resolve().parent
    / "warehouse"
    / "configs"
    / "warehouse_sorting_minimal.json"
)


@dataclass(frozen=True)
class SingleWarehouseRequest:
    task_id: str
    object_id: str
    destination_id: str
    place_xy: tuple[float, float]
    scene_path: str


def build_single_request(object_id="cube_red"):
    scenario = WarehouseScenario.load(SCENARIO_PATH)
    item = scenario.item_for(object_id)
    destination = scenario.destinations[item.destination_id]
    return SingleWarehouseRequest(
        task_id=scenario.task_id,
        object_id=item.object_id,
        destination_id=item.destination_id,
        place_xy=tuple(destination["place_xy"]),
        scene_path=SCENE_PATH,
    )


def run_single(object_id="cube_red", use_viewer=False):
    request = build_single_request(object_id)
    controller_result = run(
        target_name=request.object_id,
        use_viewer=use_viewer,
        scene_path=request.scene_path,
        place_xy=request.place_xy,
    )
    state = controller_result.get("state")
    return {
        "task_id": request.task_id,
        "object_id": request.object_id,
        "destination_id": request.destination_id,
        "place_xy": request.place_xy,
        "success": bool(controller_result.get("success")),
        "state": getattr(state, "name", str(state)),
        "error_message": controller_result.get("error_message"),
        "failure_code": controller_result.get("failure_code"),
        "failed_state": controller_result.get("failed_state"),
        "verification": controller_result.get("verification"),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one warehouse pick-and-place cycle in MuJoCo."
    )
    parser.add_argument(
        "--object",
        choices=("cube_red", "cube_blue"),
        default="cube_red",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="open the passive MuJoCo viewer (use mjpython on macOS)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_single(args.object, args.viewer)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
