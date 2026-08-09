import json
from pathlib import Path

import pytest

from warehouse.planners import ConstraintAwarePlanner, OllamaPlanner, RuleBasedPlanner
from warehouse.scenario import WarehouseScenario
from warehouse.validation import PlanValidator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPLEX_SCENARIO_PATH = (
    PROJECT_ROOT / "warehouse" / "configs" / "warehouse_sorting_complex.json"
)


@pytest.fixture()
def scenario():
    return WarehouseScenario.load()


def test_rule_planner_builds_valid_pick_place_verify_sequence(scenario):
    planner = RuleBasedPlanner(scenario)

    plan = planner.plan(
        "分拣红色和蓝色货物",
        scenario.scene_state(["cube_red", "cube_blue"]),
    )

    assert [step.skill for step in plan.steps] == [
        "scan",
        "pick",
        "place",
        "verify",
        "pick",
        "place",
        "verify",
    ]
    assert plan.steps[2].destination_id == "priority_bin.slot_1"
    assert plan.steps[5].destination_id == "standard_bin.slot_1"


def test_empty_available_object_list_produces_scan_only_plan(scenario):
    state = scenario.scene_state([])

    plan = RuleBasedPlanner(scenario).plan(scenario.goal, state)

    assert state["available_objects"] == []
    assert [step.skill for step in plan.steps] == ["scan"]


def test_validator_rejects_direct_joint_control_fields(scenario):
    payload = {
        "task_id": scenario.task_id,
        "goal": "unsafe",
        "steps": [
            {
                "step_id": 1,
                "skill": "scan",
                "object_id": None,
                "destination_id": None,
                "qpos": [0.0] * 7,
            }
        ],
        "failure_policy": {},
    }

    with pytest.raises(ValueError, match="Direct robot-control fields"):
        PlanValidator(scenario).parse_and_validate(payload)


def test_validator_rejects_wrong_destination(scenario):
    payload = RuleBasedPlanner(scenario).plan(
        "分拣红色货物", scenario.scene_state(["cube_red"])
    ).to_dict()
    payload["steps"][2]["destination_id"] = "standard_bin.slot_1"

    with pytest.raises(ValueError, match="must be placed"):
        PlanValidator(scenario).parse_and_validate(payload)


def test_validator_requires_complete_pick_place_verify_cycle(scenario):
    payload = RuleBasedPlanner(scenario).plan(
        "分拣红色货物", scenario.scene_state(["cube_red"])
    ).to_dict()
    payload["steps"] = list(payload["steps"])
    payload["steps"].pop()

    with pytest.raises(ValueError, match="placed and verified"):
        PlanValidator(scenario).parse_and_validate(payload)


def test_validator_requires_single_leading_scan(scenario):
    payload = RuleBasedPlanner(scenario).plan(
        "分拣红色货物", scenario.scene_state(["cube_red"])
    ).to_dict()
    payload["steps"] = list(payload["steps"])
    payload["steps"].pop(0)
    for index, step in enumerate(payload["steps"], start=1):
        step["step_id"] = index

    with pytest.raises(ValueError, match="exactly one scan"):
        PlanValidator(scenario).parse_and_validate(payload)


def test_ollama_planner_validates_mocked_structured_response(scenario):
    expected = RuleBasedPlanner(scenario).plan(
        "分拣红色货物", scenario.scene_state(["cube_red"])
    )

    def fake_transport(url, payload, timeout):
        assert url.endswith("/api/chat")
        assert payload["format"]["type"] == "object"
        properties = payload["format"]["properties"]
        assert properties["task_id"]["const"] == scenario.task_id
        step_variants = properties["steps"]["items"]["oneOf"]
        assert step_variants[0]["properties"]["skill"]["const"] == "scan"
        assert step_variants[0]["properties"]["object_id"]["type"] == "null"
        pick_variant = step_variants[1]["properties"]
        assert pick_variant["skill"]["const"] == "pick"
        assert pick_variant["object_id"]["const"] == "cube_red"
        assert pick_variant["destination_id"]["type"] == "null"
        assert timeout == 120
        return {"message": {"content": json.dumps(expected.to_dict())}}

    planner = OllamaPlanner(scenario, transport=fake_transport)

    actual = planner.plan(
        "分拣红色货物", scenario.scene_state(["cube_red"])
    )

    assert actual == expected


def test_ollama_planner_receives_replan_context_and_relocation_schema():
    scenario = WarehouseScenario.load(COMPLEX_SCENARIO_PATH)
    state = scenario.scene_state(["cube_red", "cube_blue"])
    for destination in state["destinations"].values():
        destination.update({"occupied_by": [], "available": True})
    state["destinations"]["priority_bin.slot_1"].update(
        {"occupied_by": ["cube_blue"], "available": False}
    )
    expected = {
        "task_id": scenario.task_id,
        "goal": scenario.goal,
        "steps": [
            {
                "step_id": 1,
                "skill": "scan",
                "object_id": None,
                "destination_id": None,
            },
            {
                "step_id": 2,
                "skill": "relocate",
                "object_id": "cube_blue",
                "destination_id": "buffer.slot_1",
            },
            {
                "step_id": 3,
                "skill": "pick",
                "object_id": "cube_red",
                "destination_id": None,
            },
            {
                "step_id": 4,
                "skill": "place",
                "object_id": "cube_red",
                "destination_id": "priority_bin.slot_1",
            },
            {
                "step_id": 5,
                "skill": "verify",
                "object_id": "cube_red",
                "destination_id": "priority_bin.slot_1",
            },
            {
                "step_id": 6,
                "skill": "pick",
                "object_id": "cube_blue",
                "destination_id": None,
            },
            {
                "step_id": 7,
                "skill": "place",
                "object_id": "cube_blue",
                "destination_id": "standard_bin.slot_1",
            },
            {
                "step_id": 8,
                "skill": "verify",
                "object_id": "cube_blue",
                "destination_id": "standard_bin.slot_1",
            },
        ],
        "failure_policy": scenario.failure_policy,
    }
    context = {
        "iteration_id": 2,
        "replan_reason": "after_failure",
        "execution_history": [{"failure_code": "ik_failed"}],
    }

    def fake_transport(url, payload, timeout):
        prompt = json.loads(payload["messages"][1]["content"])
        assert prompt["execution_context"] == context
        assert "relocate" in prompt["constraints"]["allowed_skills"]
        variants = payload["format"]["properties"]["steps"]["items"][
            "oneOf"
        ]
        assert any(
            item["properties"]["skill"].get("const") == "relocate"
            and item["properties"]["destination_id"].get("const")
            == "buffer.slot_1"
            for item in variants
        )
        return {"message": {"content": json.dumps(expected)}}

    plan = OllamaPlanner(scenario, transport=fake_transport).plan(
        scenario.goal,
        state,
        execution_context=context,
    )

    assert plan.steps[1].skill == "relocate"


def test_ollama_repairs_plan_rejected_by_dynamic_occupancy_check():
    scenario = WarehouseScenario.load(COMPLEX_SCENARIO_PATH)
    state = scenario.scene_state(["cube_red", "cube_blue"])
    for destination in state["destinations"].values():
        destination.update({"occupied_by": [], "available": True})
    state["destinations"]["priority_bin.slot_1"].update(
        {"occupied_by": ["cube_blue"], "available": False}
    )
    unsafe_plan = RuleBasedPlanner(scenario).plan(
        scenario.goal,
        {**state, "available_objects": ["cube_red"]},
    )
    safe_plan = ConstraintAwarePlanner(scenario).plan(scenario.goal, state)
    requests = []

    def fake_transport(url, payload, timeout):
        requests.append(payload)
        response_plan = unsafe_plan if len(requests) == 1 else safe_plan
        return {"message": {"content": json.dumps(response_plan.to_dict())}}

    planner = OllamaPlanner(scenario, transport=fake_transport)

    actual = planner.plan(scenario.goal, state)

    assert actual == safe_plan
    assert len(requests) == 2
    assert planner.request_count == 2
    assert planner.rejection_count == 1
    repair_message = json.loads(requests[1]["messages"][-1]["content"])
    assert "occupied by ['cube_blue']" in repair_message["validation_error"]
    assert "Do not weaken or bypass" in repair_message["instruction"]


def test_ollama_stops_after_bounded_plan_repairs():
    scenario = WarehouseScenario.load(COMPLEX_SCENARIO_PATH)
    state = scenario.scene_state(["cube_red", "cube_blue"])
    for destination in state["destinations"].values():
        destination.update({"occupied_by": [], "available": True})
    state["destinations"]["priority_bin.slot_1"].update(
        {"occupied_by": ["cube_blue"], "available": False}
    )
    unsafe_plan = RuleBasedPlanner(scenario).plan(
        scenario.goal,
        {**state, "available_objects": ["cube_red"]},
    )
    calls = []

    def fake_transport(url, payload, timeout):
        calls.append(payload)
        return {"message": {"content": json.dumps(unsafe_plan.to_dict())}}

    planner = OllamaPlanner(
        scenario,
        max_repair_attempts=1,
        transport=fake_transport,
    )

    with pytest.raises(RuntimeError, match="rejected after 2 attempt"):
        planner.plan(scenario.goal, state)

    assert len(calls) == 2
    assert planner.request_count == 2
    assert planner.rejection_count == 2
