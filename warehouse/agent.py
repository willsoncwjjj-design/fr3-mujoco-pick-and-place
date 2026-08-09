import json
from dataclasses import dataclass
from typing import Optional

from warehouse.execution import ItemExecutionResult, WarehouseTaskExecutor
from warehouse.scenario import WarehouseScenario
from warehouse.schemas import TaskPlan
from warehouse.state import WarehouseStateSnapshot


@dataclass(frozen=True)
class AgentIteration:
    iteration_id: int
    replan_reason: str
    state_before: WarehouseStateSnapshot
    plan: TaskPlan
    execution: ItemExecutionResult
    state_after: WarehouseStateSnapshot
    state_changed: bool


@dataclass(frozen=True)
class AgentRunResult:
    task_id: str
    success: bool
    termination_reason: str
    initial_state: WarehouseStateSnapshot
    final_state: WarehouseStateSnapshot
    iterations: tuple[AgentIteration, ...]
    planning_calls: int
    planner_requests: int
    plan_rejections: int
    error_message: Optional[str] = None


class ClosedLoopWarehouseAgent:
    """每次高层机器人操作后，根据最新观测状态重新规划。"""

    def __init__(
        self,
        scenario,
        observer,
        planner,
        executor,
        max_iterations=8,
        max_no_progress=2,
    ):
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if max_no_progress < 1:
            raise ValueError("max_no_progress must be positive")
        self.scenario: WarehouseScenario = scenario
        self.observer = observer
        self.planner = planner
        self.executor: WarehouseTaskExecutor = executor
        self.max_iterations = int(max_iterations)
        self.max_no_progress = int(max_no_progress)

    def run(self) -> AgentRunResult:
        self._planning_calls = 0
        self._planner_request_start = int(
            getattr(self.planner, "request_count", 0)
        )
        self._planner_rejection_start = int(
            getattr(self.planner, "rejection_count", 0)
        )
        initial_state = self.observer.observe()
        current_state = initial_state
        iterations = []
        no_progress_count = 0
        replan_reason = "initial_plan"

        if self._goal_satisfied(current_state):
            return self._result(
                initial_state,
                current_state,
                iterations,
                True,
                "goal_satisfied",
            )

        for iteration_id in range(1, self.max_iterations + 1):
            planner_state = current_state.to_planner_state(self.scenario)
            context = self._planning_context(
                iteration_id,
                replan_reason,
                iterations,
            )
            try:
                self._planning_calls += 1
                plan = self.planner.plan(
                    self.scenario.goal,
                    planner_state,
                    execution_context=context,
                )
                task_result = self.executor.execute_next(plan)
            except (RuntimeError, ValueError) as error:
                reason = (
                    "no_executable_action"
                    if str(error) == "Plan contains no executable warehouse operation"
                    else "planning_or_execution_failed"
                )
                return self._result(
                    initial_state,
                    current_state,
                    iterations,
                    False,
                    reason,
                    str(error),
                )

            execution = task_result.items[0]
            next_state = self.observer.observe()
            before_signature = self._state_signature(current_state)
            after_signature = self._state_signature(next_state)
            state_changed = before_signature != after_signature
            no_progress_count = 0 if state_changed else no_progress_count + 1
            iterations.append(
                AgentIteration(
                    iteration_id=iteration_id,
                    replan_reason=replan_reason,
                    state_before=current_state,
                    plan=plan,
                    execution=execution,
                    state_after=next_state,
                    state_changed=state_changed,
                )
            )
            current_state = next_state

            if self._goal_satisfied(current_state):
                return self._result(
                    initial_state,
                    current_state,
                    iterations,
                    True,
                    "goal_satisfied",
                )
            if execution.disposition == "stopped":
                return self._result(
                    initial_state,
                    current_state,
                    iterations,
                    False,
                    "failure_policy_stop",
                    execution.error_message,
                )
            if no_progress_count >= self.max_no_progress:
                return self._result(
                    initial_state,
                    current_state,
                    iterations,
                    False,
                    "no_progress",
                )
            replan_reason = (
                "after_success" if execution.success else "after_failure"
            )

        return self._result(
            initial_state,
            current_state,
            iterations,
            False,
            "max_iterations",
        )

    def _planning_context(self, iteration_id, reason, iterations):
        history = []
        for item in iterations:
            execution = item.execution
            history.append(
                {
                    "iteration_id": item.iteration_id,
                    "operation": execution.operation,
                    "object_id": execution.object_id,
                    "destination_id": execution.destination_id,
                    "success": execution.success,
                    "disposition": execution.disposition,
                    "failure_code": execution.failure_code,
                }
            )
        return {
            "iteration_id": iteration_id,
            "replan_reason": reason,
            "execution_history": history,
        }

    def _goal_satisfied(self, state):
        return set(state.completed_objects) == set(self.scenario.object_ids)

    def _state_signature(self, state):
        payload = state.to_planner_state(self.scenario)
        payload.pop("snapshot_id", None)
        payload.pop("sim_time_s", None)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def _result(
        self,
        initial_state,
        final_state,
        iterations,
        success,
        termination_reason,
        error_message=None,
    ):
        planner_requests = (
            int(self.planner.request_count) - self._planner_request_start
            if hasattr(self.planner, "request_count")
            else self._planning_calls
        )
        plan_rejections = (
            int(self.planner.rejection_count) - self._planner_rejection_start
            if hasattr(self.planner, "rejection_count")
            else 0
        )
        return AgentRunResult(
            task_id=self.scenario.task_id,
            success=success,
            termination_reason=termination_reason,
            initial_state=initial_state,
            final_state=final_state,
            iterations=tuple(iterations),
            planning_calls=self._planning_calls,
            planner_requests=planner_requests,
            plan_rejections=plan_rejections,
            error_message=error_message,
        )
