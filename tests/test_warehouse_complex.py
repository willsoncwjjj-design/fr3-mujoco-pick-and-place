from pathlib import Path

import pytest

from warehouse.execution import WarehouseTaskExecutor
from warehouse.planners import ConstraintAwarePlanner
from warehouse.scenario import WarehouseScenario
from warehouse.validation import PlanValidator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_PATH = (
    PROJECT_ROOT / "warehouse" / "configs" / "warehouse_sorting_complex.json"
)
OBJECT_CATALOG = [
    {"class_name": "cube_red", "body_id": 11},
    {"class_name": "cube_blue", "body_id": 12},
]


class ControllerSpy:
    def __init__(self):
        self.calls = []

    def run_cycle(self, target_body_id, place_xy):
        self.calls.append((target_body_id, tuple(place_xy)))
        return {"success": True}


@pytest.fixture()
def scenario():
    return WarehouseScenario.load(SCENARIO_PATH)


@pytest.fixture()
def occupied_state(scenario):
    state = scenario.scene_state(["cube_red", "cube_blue"])
    for destination in state["destinations"].values():
        destination["occupied_by"] = []
        destination["available"] = True
    state["destinations"]["priority_bin.slot_1"].update(
        {"occupied_by": ["cube_blue"], "available": False}
    )
    return state


def test_constraint_planner_relocates_blocker_before_sorting(
    scenario,
    occupied_state,
):
    plan = ConstraintAwarePlanner(scenario).plan(
        scenario.goal,
        occupied_state,
    )

    assert [step.skill for step in plan.steps] == [
        "scan",
        "relocate",
        "pick",
        "place",
        "verify",
        "pick",
        "place",
        "verify",
    ]
    relocate = plan.steps[1]
    assert relocate.object_id == "cube_blue"
    assert relocate.destination_id == "buffer.slot_1"
    assert [step.object_id for step in plan.steps if step.skill == "pick"] == [
        "cube_red",
        "cube_blue",
    ]


def test_validator_rejects_relocation_to_non_buffer_destination(
    scenario,
    occupied_state,
):
    payload = ConstraintAwarePlanner(scenario).plan(
        scenario.goal,
        occupied_state,
    ).to_dict()
    payload["steps"][1]["destination_id"] = "standard_bin.slot_1"

    with pytest.raises(ValueError, match="Relocate destination must be a buffer"):
        PlanValidator(scenario).parse_and_validate(payload)


def test_executor_compiles_relocation_and_sorting_in_plan_order(
    scenario,
    occupied_state,
):
    plan = ConstraintAwarePlanner(scenario).plan(
        scenario.goal,
        occupied_state,
    )
    executor = WarehouseTaskExecutor(scenario, None, OBJECT_CATALOG)

    cycles = executor.prepare(plan)

    assert [cycle.operation for cycle in cycles] == [
        "relocate",
        "pick",
        "pick",
    ]
    assert [cycle.object_id for cycle in cycles] == [
        "cube_blue",
        "cube_red",
        "cube_blue",
    ]
    assert [cycle.destination_id for cycle in cycles] == [
        "buffer.slot_1",
        "priority_bin.slot_1",
        "standard_bin.slot_1",
    ]
    assert [cycle.place_xy for cycle in cycles] == [
        (0.52, 0.12),
        (0.30, -0.18),
        (0.46, -0.18),
    ]


def test_executor_reuses_controller_for_three_complex_cycles(
    scenario,
    occupied_state,
):
    plan = ConstraintAwarePlanner(scenario).plan(
        scenario.goal,
        occupied_state,
    )
    controller = ControllerSpy()
    executor = WarehouseTaskExecutor(scenario, controller, OBJECT_CATALOG)

    result = executor.execute(plan)

    assert result.success
    assert controller.calls == [
        (12, (0.52, 0.12)),
        (11, (0.30, -0.18)),
        (12, (0.46, -0.18)),
    ]
    assert [item.operation for item in result.items] == [
        "relocate",
        "pick",
        "pick",
    ]


def test_constraint_planner_rejects_occupied_buffer(
    scenario,
    occupied_state,
):
    occupied_state["destinations"]["buffer.slot_1"].update(
        {"occupied_by": ["cube_red"], "available": False}
    )

    with pytest.raises(ValueError, match="No available buffer destination"):
        ConstraintAwarePlanner(scenario).plan(scenario.goal, occupied_state)
