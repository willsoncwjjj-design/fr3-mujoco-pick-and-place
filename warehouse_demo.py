import argparse
import json
import logging

from warehouse.planners import (
    DEFAULT_OLLAMA_MODEL,
    OllamaPlanner,
    RuleBasedPlanner,
)
from warehouse.scenario import WarehouseScenario


def build_plan(planner_name, task_text, model, available_objects=None):
    scenario = WarehouseScenario.load()
    scene_state = scenario.scene_state(available_objects)
    if planner_name == "ollama":
        planner = OllamaPlanner(scenario, model=model)
    else:
        planner = RuleBasedPlanner(scenario)
    return planner.plan(task_text, scene_state)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and validate a warehouse sorting task plan."
    )
    parser.add_argument(
        "--planner", choices=("rule", "ollama"), default="rule"
    )
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument(
        "--task",
        default="将当前货物按货类分拣到指定库位，并逐件校验结果",
    )
    parser.add_argument(
        "--objects",
        nargs="*",
        help="optional subset of visible MuJoCo object ids",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    plan = build_plan(args.planner, args.task, args.model, args.objects)
    print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
