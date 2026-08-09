import pytest

from main import run
from warehouse.planners import ConstraintAwarePlanner
from warehouse_agent_run import run_ollama_agent
from warehouse_batch_run import run_observed_batch
from warehouse_complex_run import run_complex


@pytest.mark.integration
def test_baseline_pick_and_place():
    result = run(target_name="cube_red", use_viewer=False)
    assert result["success"], result


@pytest.mark.integration
def test_warehouse_red_pick_and_place():
    result = run(
        target_name="cube_red",
        use_viewer=False,
        scene_path="robot/franka_fr3/warehouse_scene.xml",
        place_xy=(0.30, -0.18),
    )

    assert result["success"], result
    assert tuple(result["place_xy"]) == (0.30, -0.18)
    assert result["verification"]["target_visible"]
    assert result["verification"]["moved_distance_m"] >= 0.02
    assert result["verification"]["place_error_m"] <= 0.05


@pytest.mark.integration
def test_warehouse_blue_pick_and_place():
    result = run(
        target_name="cube_blue",
        use_viewer=False,
        scene_path="robot/franka_fr3/warehouse_scene.xml",
        place_xy=(0.46, -0.18),
    )

    assert result["success"], result
    assert tuple(result["place_xy"]) == (0.46, -0.18)
    assert result["verification"]["target_visible"]
    assert result["verification"]["moved_distance_m"] >= 0.02
    assert result["verification"]["place_error_m"] <= 0.05


@pytest.mark.integration
def test_warehouse_red_blue_batch():
    report = run_observed_batch()
    result = report.task_result

    assert result.success, result
    assert report.goal_satisfied, report
    assert report.initial_state.available_objects == ("cube_red", "cube_blue")
    assert report.final_state.completed_objects == ("cube_red", "cube_blue")
    assert [item.object_id for item in result.items] == [
        "cube_red",
        "cube_blue",
    ]
    assert result.remaining_objects == ()
    for item in result.items:
        assert item.verification["target_visible"]
        assert item.verification["moved_distance_m"] >= 0.02
        assert item.verification["place_error_m"] <= 0.05


@pytest.mark.integration
def test_complex_warehouse_relocation_sequence():
    result = run_complex()
    report = result.report

    assert result.goal_satisfied, result
    assert report.initial_state.destination_for(
        "priority_bin.slot_1"
    ).occupied_by == ("cube_blue",)
    assert [step.skill for step in report.plan.steps] == [
        "scan",
        "relocate",
        "pick",
        "place",
        "verify",
        "pick",
        "place",
        "verify",
    ]
    assert [item.operation for item in report.task_result.items] == [
        "relocate",
        "pick",
        "pick",
    ]
    assert report.final_state.completed_objects == ("cube_red", "cube_blue")
    assert report.final_state.destination_for("buffer.slot_1").available


@pytest.mark.integration
def test_closed_loop_agent_replans_in_complex_scene():
    result = run_ollama_agent(planner_factory=ConstraintAwarePlanner)

    assert result.success, result
    assert result.termination_reason == "goal_satisfied"
    assert [item.execution.operation for item in result.iterations] == [
        "relocate",
        "pick",
        "pick",
    ]
    assert [item.replan_reason for item in result.iterations] == [
        "initial_plan",
        "after_success",
        "after_success",
    ]
    assert all(item.state_changed for item in result.iterations)
