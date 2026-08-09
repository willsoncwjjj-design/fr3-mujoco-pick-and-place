import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from warehouse.planners import ConstraintAwarePlanner
from warehouse.scenario import WarehouseScenario
from warehouse_batch_run import ObservedBatchResult, run_observed_batch

SCENE_PATH = "robot/franka_fr3/warehouse_complex_scene.xml"
SCENARIO_PATH = (
    Path(__file__).resolve().parent
    / "warehouse"
    / "configs"
    / "warehouse_sorting_complex.json"
)


@dataclass(frozen=True)
class ComplexRunResult:
    report: ObservedBatchResult
    relocation_performed: bool
    buffer_cleared: bool
    goal_satisfied: bool


def run_complex(use_viewer=False):
    scenario = WarehouseScenario.load(SCENARIO_PATH)
    report = run_observed_batch(
        use_viewer=use_viewer,
        scene_path=SCENE_PATH,
        scenario_path=SCENARIO_PATH,
        planner_type=ConstraintAwarePlanner,
    )
    relocation_performed = any(
        item.operation == "relocate" for item in report.task_result.items
    )
    buffer_cleared = all(
        report.final_state.destination_for(destination_id).available
        for destination_id in scenario.buffer_destination_ids
    )
    return ComplexRunResult(
        report=report,
        relocation_performed=relocation_performed,
        buffer_cleared=buffer_cleared,
        goal_satisfied=(
            report.goal_satisfied
            and relocation_performed
            and buffer_cleared
        ),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run destination-unblocking warehouse sorting in MuJoCo."
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="open the passive MuJoCo viewer (use mjpython on macOS)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_complex(args.viewer)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if result.goal_satisfied else 1


if __name__ == "__main__":
    raise SystemExit(main())
