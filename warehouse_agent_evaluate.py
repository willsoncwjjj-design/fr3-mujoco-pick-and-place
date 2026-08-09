import argparse
import json
import random
import time
from pathlib import Path

from warehouse.evaluation import (
    agent_result_to_record,
    runner_error_record,
    summarize_agent_trials,
    write_agent_evaluation,
)
from warehouse.planners import DEFAULT_OLLAMA_MODEL, ConstraintAwarePlanner
from warehouse_agent_run import run_ollama_agent

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "artifacts" / "agent_evaluation"


def sample_initial_positions(trials, seed, source_jitter_m, blocker_jitter_m):
    generator = random.Random(seed)
    samples = []
    for _ in range(trials):
        samples.append(
            {
                "cube_red": (
                    0.30 + generator.uniform(-source_jitter_m, source_jitter_m),
                    0.15 + generator.uniform(-source_jitter_m, source_jitter_m),
                ),
                "cube_blue": (
                    0.30 + generator.uniform(-blocker_jitter_m, blocker_jitter_m),
                    -0.18 + generator.uniform(-blocker_jitter_m, blocker_jitter_m),
                ),
            }
        )
    return samples


def run_evaluation(
    trials=5,
    seed=20260809,
    planner_name="constraint",
    model=DEFAULT_OLLAMA_MODEL,
    source_jitter_m=0.02,
    blocker_jitter_m=0.01,
    output_directory=None,
):
    if trials < 1:
        raise ValueError("trials must be positive")
    if planner_name not in {"constraint", "ollama"}:
        raise ValueError(f"Unsupported planner: {planner_name}")
    if source_jitter_m < 0 or blocker_jitter_m < 0:
        raise ValueError("position jitter must not be negative")
    positions = sample_initial_positions(
        trials,
        seed,
        source_jitter_m,
        blocker_jitter_m,
    )
    planner_factory = ConstraintAwarePlanner if planner_name == "constraint" else None
    rows = []
    for trial_id, initial_xy in enumerate(positions, start=1):
        started = time.perf_counter()
        try:
            result = run_ollama_agent(
                model=model,
                initial_xy_by_object=initial_xy,
                planner_factory=planner_factory,
            )
            row = agent_result_to_record(
                result,
                trial_id,
                planner_name,
                time.perf_counter() - started,
                initial_xy,
            )
        except Exception as error:
            row = runner_error_record(
                trial_id,
                planner_name,
                time.perf_counter() - started,
                error,
                initial_xy,
            )
        rows.append(row)

    scope = {
        "scenario": "warehouse_sorting_complex_v1",
        "planner": planner_name,
        "model": model if planner_name == "ollama" else None,
        "seed": seed,
        "randomization": {
            "cube_red_source_jitter_m": source_jitter_m,
            "cube_blue_blocker_jitter_m": blocker_jitter_m,
        },
    }
    summary = summarize_agent_trials(rows, scope)
    output_directory = output_directory or DEFAULT_OUTPUT_DIRECTORY / planner_name
    csv_path, json_path = write_agent_evaluation(
        rows,
        summary,
        output_directory,
    )
    return rows, summary, csv_path, json_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the closed-loop warehouse agent across fixed-seed trials."
    )
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument(
        "--planner",
        choices=("constraint", "ollama"),
        default="constraint",
    )
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--source-jitter", type=float, default=0.02)
    parser.add_argument("--blocker-jitter", type=float, default=0.01)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    _, summary, csv_path, json_path = run_evaluation(
        trials=args.trials,
        seed=args.seed,
        planner_name=args.planner,
        model=args.model,
        source_jitter_m=args.source_jitter,
        blocker_jitter_m=args.blocker_jitter,
        output_directory=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    return 0 if summary["task_success"]["count"] == args.trials else 1


if __name__ == "__main__":
    raise SystemExit(main())
