import argparse
import json
from dataclasses import asdict, dataclass

from simulation.runtime import RobotRuntime
from warehouse.execution import TaskExecutionResult, WarehouseTaskExecutor
from warehouse.fault_injection import (
    INJECTABLE_FAILURE_STATES,
    FaultInjectingController,
    FaultInjection,
)
from warehouse_batch_run import SCENE_PATH, build_batch_plan


@dataclass(frozen=True)
class FaultDemoResult:
    injection: FaultInjection
    expected_action: str
    policy_verified: bool
    injected_events: tuple[FaultInjection, ...]
    task_result: TaskExecutionResult


def run_fault_demo(
    failure_code="object_missing",
    object_id="cube_red",
    use_viewer=False,
):
    scenario, plan = build_batch_plan()
    injection = FaultInjection(object_id, failure_code)
    with RobotRuntime(
        SCENE_PATH,
        use_viewer=use_viewer,
        object_names=scenario.object_ids,
    ) as runtime:
        controller = FaultInjectingController(
            runtime.controller,
            runtime.object_catalog,
            [injection],
        )
        task_result = WarehouseTaskExecutor(
            scenario,
            controller,
            runtime.object_catalog,
        ).execute(plan)
        if runtime.viewer and runtime.viewer.is_running():
            runtime.viewer.wait_until_closed()

    expected_action = scenario.failure_policy[failure_code]
    events = controller.events
    return FaultDemoResult(
        injection=injection,
        expected_action=expected_action,
        policy_verified=_policy_was_observed(
            task_result,
            injection,
            expected_action,
            events,
        ),
        injected_events=events,
        task_result=task_result,
    )


def _policy_was_observed(task_result, injection, expected_action, events):
    if events != (injection,):
        return False
    item = next(
        (
            result
            for result in task_result.items
            if result.object_id == injection.object_id
        ),
        None,
    )
    if item is None or not item.policy_actions:
        return False
    expected_disposition = {
        "rescan": "completed",
        "retry_once": "completed",
        "skip_and_report": "skipped",
        "stop": "stopped",
    }[expected_action]
    expected_attempts = 2 if expected_action in {"rescan", "retry_once"} else 1
    return (
        item.policy_actions[0] == expected_action
        and item.disposition == expected_disposition
        and len(item.attempts) == expected_attempts
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inject one warehouse failure and verify its recovery policy."
    )
    parser.add_argument(
        "--failure",
        choices=tuple(INJECTABLE_FAILURE_STATES),
        default="object_missing",
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
    result = run_fault_demo(args.failure, args.object, args.viewer)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if result.policy_verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
