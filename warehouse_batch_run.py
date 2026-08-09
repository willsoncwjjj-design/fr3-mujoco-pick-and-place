import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from simulation.runtime import RobotRuntime
from warehouse.execution import TaskExecutionResult, WarehouseTaskExecutor
from warehouse.planners import RuleBasedPlanner
from warehouse.scenario import WarehouseScenario
from warehouse.schemas import TaskPlan
from warehouse.state import WarehouseStateObserver, WarehouseStateSnapshot

SCENE_PATH = "robot/franka_fr3/warehouse_scene.xml"
SCENARIO_PATH = (
    Path(__file__).resolve().parent
    / "warehouse"
    / "configs"
    / "warehouse_sorting_minimal.json"
)


@dataclass(frozen=True)
class ObservedBatchResult:
    initial_state: WarehouseStateSnapshot
    plan: TaskPlan
    task_result: TaskExecutionResult
    final_state: WarehouseStateSnapshot
    goal_satisfied: bool


def build_batch_plan(scene_state=None):
    scenario = WarehouseScenario.load(SCENARIO_PATH)
    plan = RuleBasedPlanner(scenario).plan(
        scenario.goal,
        scenario.scene_state() if scene_state is None else scene_state,
    )
    return scenario, plan


def run_observed_batch(
    use_viewer=False,
    observer_factory=None,
    scene_path=SCENE_PATH,
    scenario_path=SCENARIO_PATH,
    planner_type=RuleBasedPlanner,
):
    scenario = WarehouseScenario.load(scenario_path)
    with RobotRuntime(
        scene_path,
        use_viewer=use_viewer,
        object_names=scenario.object_ids,
    ) as runtime:
        factory = observer_factory or WarehouseStateObserver.from_runtime
        observer = factory(scenario, runtime)
        initial_state = observer.observe()
        plan = planner_type(scenario).plan(
            scenario.goal,
            initial_state.to_planner_state(scenario),
        )
        task_result = WarehouseTaskExecutor(
            scenario,
            runtime.controller,
            runtime.object_catalog,
        ).execute(plan)
        final_state = observer.observe()
        if runtime.viewer and runtime.viewer.is_running():
            runtime.viewer.wait_until_closed()
    goal_satisfied = task_result.success and set(
        final_state.completed_objects
    ) == set(scenario.object_ids)
    return ObservedBatchResult(
        initial_state=initial_state,
        plan=plan,
        task_result=task_result,
        final_state=final_state,
        goal_satisfied=goal_satisfied,
    )


def run_batch(use_viewer=False, observer_factory=None):
    return run_observed_batch(use_viewer, observer_factory).task_result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the red-blue warehouse batch in one MuJoCo scene."
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="open the passive MuJoCo viewer (use mjpython on macOS)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    report = run_observed_batch(args.viewer)
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    return 0 if report.goal_satisfied else 1


if __name__ == "__main__":
    raise SystemExit(main())
