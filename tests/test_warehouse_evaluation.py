import csv
import json
from types import SimpleNamespace

import pytest

from warehouse.evaluation import (
    agent_result_to_record,
    distribution,
    runner_error_record,
    summarize_agent_trials,
    wilson_interval,
    write_agent_evaluation,
)
from warehouse_agent_evaluate import sample_initial_positions


def make_result(
    success,
    termination_reason,
    planning_calls,
    planner_requests,
    plan_rejections,
    errors,
    executions,
):
    initial_state = SimpleNamespace(sim_time_s=0.4)
    final_state = SimpleNamespace(
        sim_time_s=20.4,
        objects=[
            SimpleNamespace(
                object_id=object_id,
                destination_error_m=error,
            )
            for object_id, error in errors.items()
        ],
    )
    iterations = tuple(
        SimpleNamespace(execution=SimpleNamespace(**execution))
        for execution in executions
    )
    return SimpleNamespace(
        success=success,
        termination_reason=termination_reason,
        planning_calls=planning_calls,
        planner_requests=planner_requests,
        plan_rejections=plan_rejections,
        initial_state=initial_state,
        final_state=final_state,
        iterations=iterations,
        error_message=None,
    )


def test_agent_evaluation_uses_explicit_denominators_and_success_errors():
    success = make_result(
        True,
        "goal_satisfied",
        3,
        4,
        1,
        {"cube_red": 0.01, "cube_blue": 0.02},
        [
            {"operation": "relocate", "success": True, "failure_code": None},
            {"operation": "pick", "success": True, "failure_code": None},
            {"operation": "pick", "success": True, "failure_code": None},
        ],
    )
    failed = make_result(
        False,
        "failure_policy_stop",
        1,
        2,
        1,
        {"cube_red": 0.30, "cube_blue": 0.16},
        [
            {
                "operation": "pick",
                "success": False,
                "failure_code": "verification_failed",
            }
        ],
    )
    rows = [
        agent_result_to_record(success, 1, "ollama", 30.0),
        agent_result_to_record(failed, 2, "ollama", 10.0),
    ]

    summary = summarize_agent_trials(rows, {"planner": "ollama"})

    assert summary["task_success"]["rate"] == 0.5
    assert summary["task_success"]["denominator"].startswith("all trials")
    assert summary["planning"] == {
        "high_level_calls": 4,
        "planner_requests": 6,
        "rejected_plans": 2,
        "rejection_rate": pytest.approx(1 / 3),
        "trials_with_rejection": 2,
        "trial_repair_rate": 1.0,
    }
    assert summary["execution"]["relocation_actions"] == 1
    assert summary["execution"]["execution_failures"] == 1
    assert summary["execution"]["failure_code_counts"] == {
        "verification_failed": 1
    }
    assert summary["successful_final_place_error_m"]["n"] == 2
    assert summary["successful_final_place_error_m"]["max"] == 0.02
    assert summary["termination_counts"] == {
        "goal_satisfied": 1,
        "failure_policy_stop": 1,
    }


def test_runner_error_is_counted_as_failed_trial():
    row = runner_error_record(
        1,
        "constraint",
        0.2,
        RuntimeError("renderer unavailable"),
        {"cube_red": (0.3, 0.15)},
    )

    summary = summarize_agent_trials([row], {"planner": "constraint"})

    assert not row["success"]
    assert row["termination_reason"] == "runner_error"
    assert summary["task_success"]["count"] == 0
    assert summary["termination_counts"] == {"runner_error": 1}


def test_write_agent_evaluation_outputs_csv_and_json(tmp_path):
    result = make_result(
        True,
        "goal_satisfied",
        1,
        1,
        0,
        {"cube_red": 0.01},
        [{"operation": "pick", "success": True, "failure_code": None}],
    )
    rows = [agent_result_to_record(result, 1, "constraint", 1.2)]
    summary = summarize_agent_trials(rows, {"planner": "constraint"})

    csv_path, json_path = write_agent_evaluation(rows, summary, tmp_path)

    with csv_path.open(encoding="utf-8") as handle:
        written_rows = list(csv.DictReader(handle))
    assert written_rows[0]["planner"] == "constraint"
    assert json.loads(written_rows[0]["failure_codes"]) == {}
    assert json.loads(json_path.read_text(encoding="utf-8")) == summary


def test_fixed_seed_position_sampling_is_reproducible_and_bounded():
    first = sample_initial_positions(3, 42, 0.02, 0.01)
    second = sample_initial_positions(3, 42, 0.02, 0.01)

    assert first == second
    for sample in first:
        red_x, red_y = sample["cube_red"]
        blue_x, blue_y = sample["cube_blue"]
        assert 0.28 <= red_x <= 0.32
        assert 0.13 <= red_y <= 0.17
        assert 0.29 <= blue_x <= 0.31
        assert -0.19 <= blue_y <= -0.17


def test_distribution_uses_linear_p95_interpolation():
    assert distribution([1, 2, 3, 4])["p95"] == pytest.approx(3.85)


def test_wilson_interval_is_clamped_to_probability_range():
    lower, upper = wilson_interval(5, 5)

    assert 0 <= lower <= 1
    assert upper == 1.0
