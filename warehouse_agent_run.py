import argparse
import json
from dataclasses import asdict
from pathlib import Path

from simulation.runtime import RobotRuntime
from warehouse.agent import ClosedLoopWarehouseAgent
from warehouse.execution import WarehouseTaskExecutor
from warehouse.planners import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    OllamaPlanner,
)
from warehouse.scenario import WarehouseScenario
from warehouse.state import WarehouseStateObserver

SCENE_PATH = "robot/franka_fr3/warehouse_complex_scene.xml"
SCENARIO_PATH = (
    Path(__file__).resolve().parent
    / "warehouse"
    / "configs"
    / "warehouse_sorting_complex.json"
)


def run_ollama_agent(
    use_viewer=False,
    model=DEFAULT_OLLAMA_MODEL,
    base_url=DEFAULT_OLLAMA_URL,
    timeout_seconds=120,
    max_repair_attempts=2,
    max_iterations=8,
    initial_xy_by_object=None,
    planner_factory=None,
    observer_factory=None,
):
    scenario = WarehouseScenario.load(SCENARIO_PATH)
    with RobotRuntime(
        SCENE_PATH,
        use_viewer=use_viewer,
        object_names=scenario.object_ids,
    ) as runtime:
        for object_id, xy in (initial_xy_by_object or {}).items():
            runtime.set_free_body_xy(object_id, xy)
        observer_builder = observer_factory or WarehouseStateObserver.from_runtime
        observer = observer_builder(scenario, runtime)
        planner = (
            planner_factory(scenario)
            if planner_factory
            else OllamaPlanner(
                scenario,
                model=model,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
                max_repair_attempts=max_repair_attempts,
            )
        )
        executor = WarehouseTaskExecutor(
            scenario,
            runtime.controller,
            runtime.object_catalog,
        )
        result = ClosedLoopWarehouseAgent(
            scenario,
            observer,
            planner,
            executor,
            max_iterations=max_iterations,
        ).run()
        if runtime.viewer and runtime.viewer.is_running():
            runtime.viewer.wait_until_closed()
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the observe-plan-act Ollama warehouse agent loop."
    )
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--repair-attempts", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="open the passive MuJoCo viewer (use mjpython on macOS)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_ollama_agent(
        use_viewer=args.viewer,
        model=args.model,
        base_url=args.base_url,
        timeout_seconds=args.timeout,
        max_repair_attempts=args.repair_attempts,
        max_iterations=args.max_iterations,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
