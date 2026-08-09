from typing import Any

from warehouse.scenario import WarehouseScenario
from warehouse.schemas import (
    ALLOWED_FAILURE_ACTIONS,
    ALLOWED_SKILLS,
    FORBIDDEN_CONTROL_FIELDS,
    TaskPlan,
)


class PlanValidator:
    def __init__(self, scenario: WarehouseScenario):
        self.scenario = scenario

    def parse_and_validate(self, payload: dict[str, Any]) -> TaskPlan:
        self._reject_direct_control(payload)
        plan = TaskPlan.from_dict(payload)
        self.validate(plan)
        return plan

    def validate(self, plan: TaskPlan):
        if plan.task_id != self.scenario.task_id:
            raise ValueError(f"Unexpected task_id: {plan.task_id}")
        if not plan.goal.strip():
            raise ValueError("Plan goal must not be empty")
        if not plan.steps:
            raise ValueError("Plan must contain at least one step")

        expected_ids = list(range(1, len(plan.steps) + 1))
        actual_ids = [step.step_id for step in plan.steps]
        if actual_ids != expected_ids:
            raise ValueError("Plan step ids must be consecutive and start at 1")
        scan_steps = [step for step in plan.steps if step.skill == "scan"]
        if len(scan_steps) != 1 or plan.steps[0].skill != "scan":
            raise ValueError("Plan must start with exactly one scan step")

        picked = set()
        placed = set()
        verified = set()
        relocated = set()
        for step in plan.steps:
            if step.skill not in ALLOWED_SKILLS:
                raise ValueError(f"Unsupported skill: {step.skill}")
            if step.skill == "scan":
                self._require_empty_target(step)
                continue

            if step.object_id not in self.scenario.object_ids:
                raise ValueError(f"Unknown object in plan: {step.object_id}")
            if step.skill == "relocate":
                if step.destination_id not in self.scenario.buffer_destination_ids:
                    raise ValueError("Relocate destination must be a buffer")
                if step.object_id in relocated:
                    raise ValueError(
                        f"Object relocated more than once: {step.object_id}"
                    )
                if step.object_id in picked:
                    raise ValueError(
                        f"Object relocated after sorting started: {step.object_id}"
                    )
                relocated.add(step.object_id)
            elif step.skill == "pick":
                if step.destination_id is not None:
                    raise ValueError("Pick steps must not set destination_id")
                if step.object_id in picked:
                    raise ValueError(f"Object picked more than once: {step.object_id}")
                picked.add(step.object_id)
            elif step.skill == "place":
                if step.object_id not in picked:
                    raise ValueError(f"Object placed before pick: {step.object_id}")
                if step.destination_id not in self.scenario.destination_ids:
                    raise ValueError(
                        f"Unknown destination in plan: {step.destination_id}"
                    )
                expected = self.scenario.item_for(step.object_id).destination_id
                if step.destination_id != expected:
                    raise ValueError(
                        f"Object {step.object_id} must be placed at {expected}"
                    )
                placed.add(step.object_id)
            elif step.skill == "verify":
                if step.object_id not in placed:
                    raise ValueError(f"Object verified before place: {step.object_id}")
                if step.object_id in verified:
                    raise ValueError(
                        f"Object verified more than once: {step.object_id}"
                    )
                expected = self.scenario.item_for(step.object_id).destination_id
                if step.destination_id != expected:
                    raise ValueError(
                        f"Verification destination must be {expected}"
                    )
                verified.add(step.object_id)

        if picked != placed or placed != verified:
            raise ValueError("Every picked object must be placed and verified")
        if not relocated <= picked:
            raise ValueError("Every relocated object must later be sorted")
        if plan.failure_policy != self.scenario.failure_policy:
            raise ValueError("Plan failure policy must match the scenario policy")
        if set(plan.failure_policy.values()) - set(ALLOWED_FAILURE_ACTIONS):
            raise ValueError("Plan contains an unsupported failure action")

    def validate_against_state(self, plan: TaskPlan, scene_state: dict[str, Any]):
        self.validate(plan)
        available = set(scene_state["available_objects"])
        destinations = scene_state["destinations"]
        occupancy = {
            destination_id: set(destination.get("occupied_by", []))
            for destination_id, destination in destinations.items()
        }

        motion_steps = (
            step for step in plan.steps if step.skill in {"relocate", "pick"}
        )
        for step in motion_steps:
            if step.object_id not in available:
                raise ValueError(
                    f"Plan uses an unavailable object: {step.object_id}"
                )
            destination_id = (
                step.destination_id
                if step.skill == "relocate"
                else self.scenario.item_for(step.object_id).destination_id
            )
            blockers = occupancy[destination_id] - {step.object_id}
            if blockers:
                raise ValueError(
                    f"Destination {destination_id} is occupied by "
                    f"{sorted(blockers)}"
                )
            for occupied in occupancy.values():
                occupied.discard(step.object_id)
            occupancy[destination_id].add(step.object_id)

    @staticmethod
    def _require_empty_target(step):
        if step.object_id is not None or step.destination_id is not None:
            raise ValueError("Scan steps must not name an object or destination")

    def _reject_direct_control(self, value: Any):
        if isinstance(value, dict):
            forbidden = set(value) & FORBIDDEN_CONTROL_FIELDS
            if forbidden:
                raise ValueError(
                    "Direct robot-control fields are forbidden: "
                    f"{sorted(forbidden)}"
                )
            for nested in value.values():
                self._reject_direct_control(nested)
        elif isinstance(value, list):
            for nested in value:
                self._reject_direct_control(nested)
