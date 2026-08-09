from dataclasses import dataclass
from math import isfinite
from typing import Optional

from warehouse.scenario import WarehouseScenario
from warehouse.schemas import TaskPlan
from warehouse.validation import PlanValidator


@dataclass(frozen=True)
class PreparedCycle:
    operation: str
    object_id: str
    target_body_id: int
    destination_id: str
    place_xy: tuple[float, float]


@dataclass(frozen=True)
class ExecutionAttempt:
    attempt_number: int
    success: bool
    error_message: Optional[str] = None
    failure_code: Optional[str] = None
    failed_state: Optional[str] = None
    verification: Optional[dict[str, object]] = None


@dataclass(frozen=True)
class ItemExecutionResult:
    operation: str
    object_id: str
    destination_id: str
    success: bool
    disposition: str
    error_message: Optional[str] = None
    failure_code: Optional[str] = None
    failed_state: Optional[str] = None
    verification: Optional[dict[str, object]] = None
    policy_actions: tuple[str, ...] = ()
    attempts: tuple[ExecutionAttempt, ...] = ()


@dataclass(frozen=True)
class TaskExecutionResult:
    task_id: str
    success: bool
    items: tuple[ItemExecutionResult, ...]
    remaining_objects: tuple[str, ...]


class WarehouseTaskExecutor:
    """将高层仓储操作编译为控制器执行周期。"""

    RETRY_ACTIONS = frozenset(("rescan", "retry_once"))

    def __init__(self, scenario, controller, object_catalog):
        self.scenario: WarehouseScenario = scenario
        self.controller = controller
        self.body_ids = self._build_body_ids(object_catalog)
        self.validator = PlanValidator(scenario)

    def prepare(self, plan: TaskPlan) -> tuple[PreparedCycle, ...]:
        self.validator.validate(plan)
        motion_steps = tuple(
            step for step in plan.steps if step.skill in {"relocate", "pick"}
        )
        self._require_body_ids(motion_steps)

        cycles = []
        for step in motion_steps:
            item = self.scenario.item_for(step.object_id)
            destination_id = (
                step.destination_id
                if step.skill == "relocate"
                else item.destination_id
            )
            destination = self.scenario.destinations[destination_id]
            cycles.append(
                PreparedCycle(
                    operation=step.skill,
                    object_id=step.object_id,
                    target_body_id=self.body_ids[step.object_id],
                    destination_id=destination_id,
                    place_xy=self._normalize_place_xy(destination["place_xy"]),
                )
            )
        return tuple(cycles)

    def execute(self, plan: TaskPlan) -> TaskExecutionResult:
        if self.controller is None:
            raise RuntimeError("A controller is required for warehouse execution")
        cycles = self.prepare(plan)

        item_results = []
        for index, cycle in enumerate(cycles):
            item_result, continue_batch = self._execute_cycle(
                cycle,
                plan.failure_policy,
            )
            item_results.append(item_result)
            if not continue_batch:
                return TaskExecutionResult(
                    task_id=plan.task_id,
                    success=False,
                    items=tuple(item_results),
                    remaining_objects=tuple(
                        remaining.object_id for remaining in cycles[index + 1 :]
                    ),
                )

        return TaskExecutionResult(
            task_id=plan.task_id,
            success=all(item.success for item in item_results),
            items=tuple(item_results),
            remaining_objects=(),
        )

    def execute_next(self, plan: TaskPlan) -> TaskExecutionResult:
        """仅执行已验证计划中的下一个运动操作。"""
        if self.controller is None:
            raise RuntimeError("A controller is required for warehouse execution")
        cycles = self.prepare(plan)
        if not cycles:
            raise ValueError("Plan contains no executable warehouse operation")

        item_result, _ = self._execute_cycle(
            cycles[0],
            plan.failure_policy,
        )
        return TaskExecutionResult(
            task_id=plan.task_id,
            success=item_result.success,
            items=(item_result,),
            remaining_objects=tuple(cycle.object_id for cycle in cycles[1:]),
        )

    def _execute_cycle(self, cycle, failure_policy):
        attempts = []
        policy_actions = []
        used_retry_actions = set()

        while True:
            controller_result = self.controller.run_cycle(
                target_body_id=cycle.target_body_id,
                place_xy=cycle.place_xy,
            )
            attempt = self._build_attempt(
                len(attempts) + 1,
                controller_result,
            )
            attempts.append(attempt)

            if attempt.success:
                return (
                    self._build_item_result(
                        cycle,
                        "completed",
                        policy_actions,
                        attempts,
                    ),
                    True,
                )

            action = failure_policy.get(attempt.failure_code, "stop")
            if action in self.RETRY_ACTIONS:
                if action not in used_retry_actions:
                    used_retry_actions.add(action)
                    policy_actions.append(action)
                    continue
                action = "stop"

            policy_actions.append(action)
            disposition = "skipped" if action == "skip_and_report" else "stopped"
            return (
                self._build_item_result(
                    cycle,
                    disposition,
                    policy_actions,
                    attempts,
                ),
                action == "skip_and_report",
            )

    @staticmethod
    def _build_attempt(attempt_number, controller_result):
        success = bool(controller_result.get("success"))
        error_message = None
        failure_code = None
        failed_state = None
        if not success:
            error_message = str(
                controller_result.get("error_message", "Controller cycle failed")
            )
            failure_code = str(
                controller_result.get("failure_code") or "internal_error"
            )
            state = controller_result.get("failed_state")
            failed_state = None if state is None else str(state)
        return ExecutionAttempt(
            attempt_number=attempt_number,
            success=success,
            error_message=error_message,
            failure_code=failure_code,
            failed_state=failed_state,
            verification=controller_result.get("verification"),
        )

    @staticmethod
    def _build_item_result(cycle, disposition, policy_actions, attempts):
        final_attempt = attempts[-1]
        return ItemExecutionResult(
            operation=cycle.operation,
            object_id=cycle.object_id,
            destination_id=cycle.destination_id,
            success=final_attempt.success,
            disposition=disposition,
            error_message=final_attempt.error_message,
            failure_code=final_attempt.failure_code,
            failed_state=final_attempt.failed_state,
            verification=final_attempt.verification,
            policy_actions=tuple(policy_actions),
            attempts=tuple(attempts),
        )

    def _require_body_ids(self, pick_steps):
        missing = sorted(
            {
                step.object_id
                for step in pick_steps
                if step.object_id not in self.body_ids
            }
        )
        if missing:
            raise ValueError(f"Missing body ids for warehouse objects: {missing}")

    @staticmethod
    def _normalize_place_xy(place_xy):
        try:
            values = tuple(float(value) for value in place_xy)
        except (TypeError, ValueError) as error:
            raise ValueError("place_xy must contain two finite values") from error
        if len(values) != 2 or not all(isfinite(value) for value in values):
            raise ValueError("place_xy must contain two finite values")
        return values

    @staticmethod
    def _build_body_ids(object_catalog):
        body_ids = {}
        for item in object_catalog:
            object_id = str(item["class_name"])
            if object_id in body_ids:
                raise ValueError(f"Duplicate object in catalog: {object_id}")
            body_id = int(item["body_id"])
            if body_id < 0:
                raise ValueError(f"Invalid body id for {object_id}: {body_id}")
            body_ids[object_id] = body_id
        return body_ids
