import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median


def agent_result_to_record(
    result,
    trial_id,
    planner_name,
    wall_time_s,
    initial_xy_by_object=None,
):
    failure_codes = Counter(
        item.execution.failure_code
        for item in result.iterations
        if item.execution.failure_code
    )
    record = {
        "trial_id": int(trial_id),
        "planner": planner_name,
        "success": bool(result.success),
        "termination_reason": result.termination_reason,
        "iteration_count": len(result.iterations),
        "planning_calls": result.planning_calls,
        "planner_requests": result.planner_requests,
        "plan_rejections": result.plan_rejections,
        "relocation_count": sum(
            item.execution.operation == "relocate" for item in result.iterations
        ),
        "execution_failure_count": sum(
            not item.execution.success for item in result.iterations
        ),
        "sim_time_s": max(
            0.0,
            result.final_state.sim_time_s - result.initial_state.sim_time_s,
        ),
        "wall_time_s": float(wall_time_s),
        "failure_codes": dict(failure_codes),
        "error_message": result.error_message or "",
    }
    for object_state in result.final_state.objects:
        record[f"final_error_{object_state.object_id}_m"] = (
            object_state.destination_error_m
        )
    for object_id, xy in (initial_xy_by_object or {}).items():
        record[f"start_{object_id}_x_m"] = float(xy[0])
        record[f"start_{object_id}_y_m"] = float(xy[1])
    return record


def runner_error_record(
    trial_id,
    planner_name,
    wall_time_s,
    error,
    initial_xy_by_object=None,
):
    record = {
        "trial_id": int(trial_id),
        "planner": planner_name,
        "success": False,
        "termination_reason": "runner_error",
        "iteration_count": 0,
        "planning_calls": 0,
        "planner_requests": 0,
        "plan_rejections": 0,
        "relocation_count": 0,
        "execution_failure_count": 0,
        "sim_time_s": 0.0,
        "wall_time_s": float(wall_time_s),
        "failure_codes": {},
        "error_message": str(error),
    }
    for object_id, xy in (initial_xy_by_object or {}).items():
        record[f"start_{object_id}_x_m"] = float(xy[0])
        record[f"start_{object_id}_y_m"] = float(xy[1])
        record[f"final_error_{object_id}_m"] = None
    return record


def summarize_agent_trials(rows, scope):
    if not rows:
        raise ValueError("At least one trial record is required")
    successes = sum(bool(row["success"]) for row in rows)
    lower, upper = wilson_interval(successes, len(rows))
    total_requests = sum(int(row["planner_requests"]) for row in rows)
    total_rejections = sum(int(row["plan_rejections"]) for row in rows)
    trials_with_rejection = sum(int(row["plan_rejections"]) > 0 for row in rows)
    final_errors = []
    for row in rows:
        if not row["success"]:
            continue
        final_errors.extend(
            float(value)
            for key, value in row.items()
            if key.startswith("final_error_") and value is not None
        )
    failure_codes = Counter()
    for row in rows:
        failure_codes.update(row["failure_codes"])

    return {
        "scope": scope,
        "trial_count": len(rows),
        "task_success": {
            "count": successes,
            "rate": successes / len(rows),
            "wilson_95_ci": [lower, upper],
            "denominator": "all trials, including runner errors",
        },
        "planning": {
            "high_level_calls": sum(int(row["planning_calls"]) for row in rows),
            "planner_requests": total_requests,
            "rejected_plans": total_rejections,
            "rejection_rate": (
                total_rejections / total_requests if total_requests else 0.0
            ),
            "trials_with_rejection": trials_with_rejection,
            "trial_repair_rate": trials_with_rejection / len(rows),
        },
        "execution": {
            "iterations": distribution(
                [float(row["iteration_count"]) for row in rows]
            ),
            "relocation_actions": sum(
                int(row["relocation_count"]) for row in rows
            ),
            "execution_failures": sum(
                int(row["execution_failure_count"]) for row in rows
            ),
            "failure_code_counts": dict(failure_codes),
        },
        "timing": {
            "sim_time_s": distribution([float(row["sim_time_s"]) for row in rows]),
            "wall_time_s": distribution(
                [float(row["wall_time_s"]) for row in rows]
            ),
        },
        "successful_final_place_error_m": (
            distribution(final_errors) if final_errors else None
        ),
        "termination_counts": dict(
            Counter(str(row["termination_reason"]) for row in rows)
        ),
        "metric_definitions": {
            "task_success_rate": "successful trials / all attempted trials",
            "planning_rejection_rate": (
                "plans rejected by deterministic validators / all planner requests"
            ),
            "trial_repair_rate": (
                "trials with at least one rejected plan / all trials"
            ),
            "successful_final_place_error_m": (
                "object-to-assigned-destination error from successful trials only"
            ),
        },
    }


def write_agent_evaluation(rows, summary, output_directory):
    if not rows:
        raise ValueError("At least one trial record is required")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "agent_trials.csv"
    json_path = output_directory / "agent_evaluation_summary.json"
    fieldnames = _ordered_fieldnames(rows)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            serialized["failure_codes"] = json.dumps(
                row["failure_codes"], ensure_ascii=False, sort_keys=True
            )
            writer.writerow(serialized)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return csv_path, json_path


def wilson_interval(successes, total, z=1.96):
    if total <= 0:
        raise ValueError("total must be positive")
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z**2 / (4 * total**2)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def distribution(values):
    values = sorted(float(value) for value in values)
    if not values:
        raise ValueError("Distribution requires at least one value")
    return {
        "n": len(values),
        "mean": mean(values),
        "median": median(values),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def percentile(sorted_values, quantile):
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between 0 and 1")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = quantile * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _ordered_fieldnames(rows):
    names = []
    for row in rows:
        for key in row:
            if key not in names:
                names.append(key)
    return names
