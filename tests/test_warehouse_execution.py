from pathlib import Path

import pytest

from warehouse.execution import WarehouseTaskExecutor
from warehouse.planners import RuleBasedPlanner
from warehouse.scenario import WarehouseScenario

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_PATH = (
    PROJECT_ROOT / "warehouse" / "configs" / "warehouse_sorting_minimal.json"
)
OBJECT_CATALOG = [
    {"class_name": "cube_red", "body_id": 11},
    {"class_name": "cube_blue", "body_id": 12},
]


class ControllerSpy:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def run_cycle(self, target_body_id, place_xy):
        self.calls.append(
            {
                "target_body_id": target_body_id,
                "place_xy": tuple(place_xy),
            }
        )
        return self.results.pop(0)


@pytest.fixture()
def scenario():
    return WarehouseScenario.load(SCENARIO_PATH)


@pytest.fixture()
def plan(scenario):
    return RuleBasedPlanner(scenario).plan(
        "分拣红色和蓝色货物",
        scenario.scene_state(),
    )


def test_prepare_builds_cycles_without_controller(plan, scenario):
    executor = WarehouseTaskExecutor(
        scenario,
        None,
        OBJECT_CATALOG,
    )

    cycles = executor.prepare(plan)

    assert [cycle.object_id for cycle in cycles] == ["cube_red", "cube_blue"]
    assert [cycle.target_body_id for cycle in cycles] == [11, 12]
    assert [cycle.destination_id for cycle in cycles] == [
        "priority_bin.slot_1",
        "standard_bin.slot_1",
    ]
    assert [cycle.place_xy for cycle in cycles] == [
        (0.30, -0.18),
        (0.46, -0.18),
    ]


def test_executor_maps_each_object_cycle_to_controller(plan, scenario):
    verification = {
        "target_visible": True,
        "moved_distance_m": 0.32,
        "place_error_m": 0.01,
    }
    controller = ControllerSpy(
        [
            {"success": True, "verification": verification},
            {"success": True, "verification": verification},
        ]
    )
    executor = WarehouseTaskExecutor(
        scenario,
        controller,
        OBJECT_CATALOG,
    )

    result = executor.execute(plan)

    assert result.success
    assert controller.calls == [
        {"target_body_id": 11, "place_xy": (0.30, -0.18)},
        {"target_body_id": 12, "place_xy": (0.46, -0.18)},
    ]
    assert [item.object_id for item in result.items] == [
        "cube_red",
        "cube_blue",
    ]
    assert result.remaining_objects == ()
    assert result.items[0].verification == verification
    assert result.items[0].disposition == "completed"
    assert result.items[0].policy_actions == ()
    assert len(result.items[0].attempts) == 1
    assert result.items[0].attempts[0].success


def test_execute_next_runs_only_first_planned_cycle(plan, scenario):
    controller = ControllerSpy([{"success": True}])
    executor = WarehouseTaskExecutor(scenario, controller, OBJECT_CATALOG)

    result = executor.execute_next(plan)

    assert controller.calls == [
        {"target_body_id": 11, "place_xy": (0.30, -0.18)}
    ]
    assert result.success
    assert [item.object_id for item in result.items] == ["cube_red"]
    assert result.remaining_objects == ("cube_blue",)


def test_executor_skips_ik_failure_and_continues(plan, scenario):
    controller = ControllerSpy(
        [
            {
                "success": False,
                "error_message": "same diagnostic message",
                "failure_code": "ik_failed",
                "failed_state": "PLAN",
            },
            {"success": True, "verification": {"target_visible": True}},
        ]
    )
    executor = WarehouseTaskExecutor(
        scenario,
        controller,
        OBJECT_CATALOG,
    )

    result = executor.execute(plan)

    assert not result.success
    assert len(controller.calls) == 2
    assert result.items[0].object_id == "cube_red"
    assert result.items[0].error_message == "same diagnostic message"
    assert result.items[0].failure_code == "ik_failed"
    assert result.items[0].failed_state == "PLAN"
    assert result.items[0].disposition == "skipped"
    assert result.items[0].policy_actions == ("skip_and_report",)
    assert not result.items[0].attempts[0].success
    assert result.items[1].object_id == "cube_blue"
    assert result.items[1].success
    assert result.remaining_objects == ()


def test_executor_rescans_once_then_completes(plan, scenario):
    controller = ControllerSpy(
        [
            {
                "success": False,
                "error_message": "not visible",
                "failure_code": "object_missing",
                "failed_state": "LOCALIZE",
            },
            {"success": True},
            {"success": True},
        ]
    )
    executor = WarehouseTaskExecutor(scenario, controller, OBJECT_CATALOG)

    result = executor.execute(plan)

    assert result.success
    assert len(controller.calls) == 3
    assert result.items[0].success
    assert result.items[0].disposition == "completed"
    assert result.items[0].policy_actions == ("rescan",)
    assert len(result.items[0].attempts) == 2


def test_executor_retries_pick_once_then_completes(plan, scenario):
    controller = ControllerSpy(
        [
            {
                "success": False,
                "error_message": "grasp lost",
                "failure_code": "pick_failed",
                "failed_state": "EXECUTE",
            },
            {"success": True},
            {"success": True},
        ]
    )
    executor = WarehouseTaskExecutor(scenario, controller, OBJECT_CATALOG)

    result = executor.execute(plan)

    assert result.success
    assert len(controller.calls) == 3
    assert result.items[0].success
    assert result.items[0].policy_actions == ("retry_once",)
    assert len(result.items[0].attempts) == 2


def test_executor_stops_when_retry_is_exhausted(plan, scenario):
    failure = {
        "success": False,
        "error_message": "grasp lost",
        "failure_code": "pick_failed",
        "failed_state": "EXECUTE",
    }
    controller = ControllerSpy([failure, failure, {"success": True}])
    executor = WarehouseTaskExecutor(scenario, controller, OBJECT_CATALOG)

    result = executor.execute(plan)

    assert not result.success
    assert len(controller.calls) == 2
    assert result.items[0].disposition == "stopped"
    assert result.items[0].policy_actions == ("retry_once", "stop")
    assert len(result.items[0].attempts) == 2
    assert result.remaining_objects == ("cube_blue",)


def test_executor_stops_on_verification_failure(plan, scenario):
    controller = ControllerSpy(
        [
            {
                "success": False,
                "error_message": "outside target zone",
                "failure_code": "verification_failed",
                "failed_state": "VERIFY",
            },
            {"success": True},
        ]
    )
    executor = WarehouseTaskExecutor(scenario, controller, OBJECT_CATALOG)

    result = executor.execute(plan)

    assert not result.success
    assert len(controller.calls) == 1
    assert result.items[0].disposition == "stopped"
    assert result.items[0].policy_actions == ("stop",)
    assert result.remaining_objects == ("cube_blue",)


def test_executor_rechecks_policy_after_rescan(plan, scenario):
    controller = ControllerSpy(
        [
            {
                "success": False,
                "failure_code": "object_missing",
                "failed_state": "LOCALIZE",
            },
            {
                "success": False,
                "failure_code": "ik_failed",
                "failed_state": "PLAN",
            },
            {"success": True},
        ]
    )
    executor = WarehouseTaskExecutor(scenario, controller, OBJECT_CATALOG)

    result = executor.execute(plan)

    assert not result.success
    assert result.items[0].disposition == "skipped"
    assert result.items[0].policy_actions == ("rescan", "skip_and_report")
    assert len(result.items[0].attempts) == 2
    assert result.items[1].success
    assert result.remaining_objects == ()


def test_executor_stops_on_unmapped_failure_code(plan, scenario):
    controller = ControllerSpy(
        [
            {
                "success": False,
                "failure_code": "internal_error",
                "failed_state": "DONE",
            },
            {"success": True},
        ]
    )
    executor = WarehouseTaskExecutor(scenario, controller, OBJECT_CATALOG)

    result = executor.execute(plan)

    assert not result.success
    assert len(controller.calls) == 1
    assert result.items[0].disposition == "stopped"
    assert result.items[0].policy_actions == ("stop",)
    assert result.remaining_objects == ("cube_blue",)


def test_executor_rejects_missing_body_id_before_motion(plan, scenario):
    controller = ControllerSpy([{"success": True}])
    executor = WarehouseTaskExecutor(
        scenario,
        controller,
        [{"class_name": "cube_red", "body_id": 11}],
    )

    with pytest.raises(ValueError, match="Missing body ids.*cube_blue"):
        executor.execute(plan)

    assert controller.calls == []


def test_prepare_rejects_invalid_place_coordinates(plan, scenario):
    scenario.destinations["priority_bin.slot_1"]["place_xy"] = [float("nan"), 0]
    executor = WarehouseTaskExecutor(
        scenario,
        None,
        OBJECT_CATALOG,
    )

    with pytest.raises(ValueError, match="place_xy must contain two finite values"):
        executor.prepare(plan)
