import argparse
import json
from dataclasses import asdict
from pathlib import Path

from simulation.runtime import build_object_catalog
from simulation.scene import SimScene
from warehouse.execution import WarehouseTaskExecutor
from warehouse.planners import RuleBasedPlanner
from warehouse.scenario import WarehouseScenario

SCENE_PATH = "robot/franka_fr3/warehouse_scene.xml"
SCENARIO_PATH = (
    Path(__file__).resolve().parent
    / "warehouse"
    / "configs"
    / "warehouse_sorting_minimal.json"
)


def build_preview(task_text=None):
    scenario = WarehouseScenario.load(SCENARIO_PATH)
    scene = SimScene()
    model, _ = scene.setup(SCENE_PATH)
    object_catalog = build_object_catalog(model)
    plan = RuleBasedPlanner(scenario).plan(
        task_text or scenario.goal,
        scenario.scene_state(),
    )
    cycles = WarehouseTaskExecutor(
        scenario,
        controller=None,
        object_catalog=object_catalog,
    ).prepare(plan)
    return {
        "mode": "dry_run",
        "task_id": plan.task_id,
        "goal": plan.goal,
        "scene": SCENE_PATH,
        "cycles": [asdict(cycle) for cycle in cycles],
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preview warehouse control cycles without moving the robot."
    )
    parser.add_argument("--task", help="optional natural-language task goal")
    return parser.parse_args()


def main():
    args = parse_args()
    preview = build_preview(args.task)
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
