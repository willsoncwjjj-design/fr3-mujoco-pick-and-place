from pathlib import Path

from warehouse.agent import ClosedLoopWarehouseAgent
from warehouse.execution import WarehouseTaskExecutor
from warehouse.planners import ConstraintAwarePlanner, RuleBasedPlanner
from warehouse.scenario import WarehouseScenario
from warehouse.state import DestinationState, ObjectState, WarehouseStateSnapshot

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MINIMAL_SCENARIO_PATH = (
    PROJECT_ROOT / "warehouse" / "configs" / "warehouse_sorting_minimal.json"
)
COMPLEX_SCENARIO_PATH = (
    PROJECT_ROOT / "warehouse" / "configs" / "warehouse_sorting_complex.json"
)
OBJECT_CATALOG = [
    {"class_name": "cube_red", "body_id": 11},
    {"class_name": "cube_blue", "body_id": 12},
]


class ObserverStub:
    def __init__(self, snapshots):
        self.snapshots = list(snapshots)

    def observe(self):
        return self.snapshots.pop(0)


class ControllerStub:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def run_cycle(self, target_body_id, place_xy):
        self.calls.append((target_body_id, tuple(place_xy)))
        return self.results.pop(0)


class RecordingConstraintPlanner(ConstraintAwarePlanner):
    def __init__(self, scenario):
        super().__init__(scenario)
        self.contexts = []

    def plan(self, task_text, scene_state, execution_context=None):
        self.contexts.append(execution_context)
        return super().plan(task_text, scene_state, execution_context)


class FailureAwarePlanner:
    def __init__(self, scenario):
        self.delegate = RuleBasedPlanner(scenario)
        self.contexts = []

    def plan(self, task_text, scene_state, execution_context=None):
        self.contexts.append(execution_context)
        history = execution_context["execution_history"]
        if history and not history[-1]["success"]:
            scene_state = dict(scene_state)
            scene_state["available_objects"] = ["cube_blue"]
        return self.delegate.plan(task_text, scene_state, execution_context)


def make_snapshot(scenario, sequence_id, locations):
    objects = []
    for item in scenario.inventory:
        location = locations[item.object_id]
        assigned = item.destination_id
        at_destination = location == assigned
        position = tuple(
            scenario.destinations.get(location, {}).get(
                "place_xy",
                [0.20, 0.15] if item.object_id == "cube_red" else [0.45, 0.15],
            )
        ) + (0.02,)
        objects.append(
            ObjectState(
                object_id=item.object_id,
                visible=True,
                position=position,
                assigned_destination_id=assigned,
                at_destination=at_destination,
                destination_error_m=0.0 if at_destination else 0.2,
            )
        )

    destinations = []
    for destination_id, destination in scenario.destinations.items():
        occupied_by = tuple(
            object_id
            for object_id, location in locations.items()
            if location == destination_id
        )
        destinations.append(
            DestinationState(
                destination_id=destination_id,
                place_xy=tuple(destination["place_xy"]),
                occupied_by=occupied_by,
                available=not occupied_by,
                acceptance_radius_m=0.05,
            )
        )
    return WarehouseStateSnapshot(
        sequence_id=sequence_id,
        sim_time_s=float(sequence_id),
        objects=tuple(objects),
        destinations=tuple(destinations),
    )


def test_agent_replans_after_each_complex_operation():
    scenario = WarehouseScenario.load(COMPLEX_SCENARIO_PATH)
    snapshots = [
        make_snapshot(
            scenario,
            1,
            {
                "cube_red": "source",
                "cube_blue": "priority_bin.slot_1",
            },
        ),
        make_snapshot(
            scenario,
            2,
            {"cube_red": "source", "cube_blue": "buffer.slot_1"},
        ),
        make_snapshot(
            scenario,
            3,
            {
                "cube_red": "priority_bin.slot_1",
                "cube_blue": "buffer.slot_1",
            },
        ),
        make_snapshot(
            scenario,
            4,
            {
                "cube_red": "priority_bin.slot_1",
                "cube_blue": "standard_bin.slot_1",
            },
        ),
    ]
    planner = RecordingConstraintPlanner(scenario)
    controller = ControllerStub([{"success": True}] * 3)
    executor = WarehouseTaskExecutor(scenario, controller, OBJECT_CATALOG)

    result = ClosedLoopWarehouseAgent(
        scenario,
        ObserverStub(snapshots),
        planner,
        executor,
    ).run()

    assert result.success
    assert result.termination_reason == "goal_satisfied"
    assert [item.execution.operation for item in result.iterations] == [
        "relocate",
        "pick",
        "pick",
    ]
    assert [item.execution.object_id for item in result.iterations] == [
        "cube_blue",
        "cube_red",
        "cube_blue",
    ]
    assert [context["replan_reason"] for context in planner.contexts] == [
        "initial_plan",
        "after_success",
        "after_success",
    ]
    assert len(controller.calls) == 3
    assert result.planning_calls == 3
    assert result.planner_requests == 3
    assert result.plan_rejections == 0


def test_agent_passes_failure_history_into_replanning():
    scenario = WarehouseScenario.load(MINIMAL_SCENARIO_PATH)
    initial = make_snapshot(
        scenario,
        1,
        {"cube_red": "source", "cube_blue": "source"},
    )
    snapshots = [
        initial,
        make_snapshot(
            scenario,
            2,
            {"cube_red": "source", "cube_blue": "source"},
        ),
        make_snapshot(
            scenario,
            3,
            {"cube_red": "source", "cube_blue": "standard_bin.slot_1"},
        ),
        make_snapshot(
            scenario,
            4,
            {
                "cube_red": "priority_bin.slot_1",
                "cube_blue": "standard_bin.slot_1",
            },
        ),
    ]
    planner = FailureAwarePlanner(scenario)
    controller = ControllerStub(
        [
            {
                "success": False,
                "failure_code": "ik_failed",
                "failed_state": "PLAN",
                "error_message": "red target temporarily unreachable",
            },
            {"success": True},
            {"success": True},
        ]
    )
    executor = WarehouseTaskExecutor(scenario, controller, OBJECT_CATALOG)

    result = ClosedLoopWarehouseAgent(
        scenario,
        ObserverStub(snapshots),
        planner,
        executor,
    ).run()

    assert result.success
    assert [item.execution.object_id for item in result.iterations] == [
        "cube_red",
        "cube_blue",
        "cube_red",
    ]
    assert planner.contexts[1]["replan_reason"] == "after_failure"
    previous = planner.contexts[1]["execution_history"][-1]
    assert previous["failure_code"] == "ik_failed"
    assert previous["disposition"] == "skipped"


def test_agent_stops_after_repeated_success_without_state_progress():
    scenario = WarehouseScenario.load(MINIMAL_SCENARIO_PATH)
    snapshots = [
        make_snapshot(
            scenario,
            sequence_id,
            {"cube_red": "source", "cube_blue": "source"},
        )
        for sequence_id in range(1, 4)
    ]
    planner = RuleBasedPlanner(scenario)
    controller = ControllerStub([{"success": True}, {"success": True}])
    executor = WarehouseTaskExecutor(scenario, controller, OBJECT_CATALOG)

    result = ClosedLoopWarehouseAgent(
        scenario,
        ObserverStub(snapshots),
        planner,
        executor,
        max_no_progress=2,
    ).run()

    assert not result.success
    assert result.termination_reason == "no_progress"
    assert len(result.iterations) == 2
