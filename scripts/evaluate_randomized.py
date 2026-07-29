from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from control.controller import Controller, RobotState
from main import build_object_catalog, initialize_home
from perception.camera import SimCamera
from perception.detector import Detector
from perception.localizer import ObjectLocalizer
from planning.grasp_planner import GraspPlanner
from planning.trajectory_planner import TrajectoryPlanner
from robot.gripper import PandaGripper
from robot.kinematics import DLSIKSolver
from robot.panda import PandaRobot
from simulation.scene import SimScene

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "robot" / "franka_fr3" / "myscene.xml"
TARGET_NAME = "cube_red"
TARGET_Z = 0.85
PLACE_XY = np.array([0.3, -0.15])


def set_free_body_position(model, data, body_name, position):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    joint_id = int(model.body_jntadr[body_id])
    qpos_address = int(model.jnt_qposadr[joint_id])
    dof_address = int(model.jnt_dofadr[joint_id])
    data.qpos[qpos_address : qpos_address + 3] = position
    data.qpos[qpos_address + 3 : qpos_address + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[dof_address : dof_address + 6] = 0.0
    mujoco.mj_forward(model, data)


def create_controller(model, data):
    scene = SimScene()
    scene.model = model
    scene.data = data
    catalog = build_object_catalog(model)
    robot = PandaRobot(model, data)
    camera = SimCamera(model, data)
    detector = Detector(model)
    localizer = ObjectLocalizer(camera.camera_matrix)
    ik_solver = DLSIKSolver(model, data, "attachment_site", 0.1)
    grasp_planner = GraspPlanner(model, data, ik_solver=ik_solver)
    trajectory_planner = TrajectoryPlanner(ik_solver)
    gripper = PandaGripper(model, data)
    gripper.open()
    for _ in range(200):
        scene.step()
    return (
        Controller(
            catalog,
            scene,
            robot,
            camera,
            detector,
            localizer,
            grasp_planner,
            trajectory_planner,
            gripper,
        ),
        camera,
        catalog,
    )


def sample_positions(generator, count):
    fixed_objects = (
        np.array([0.45, -0.05]),
        np.array([0.15, 0.30]),
        np.array([0.50, 0.15]),
    )
    positions = []
    while len(positions) < count:
        candidate = np.array(
            [generator.uniform(0.12, 0.50), generator.uniform(0.03, 0.32)]
        )
        if any(np.linalg.norm(candidate - item) < 0.12 for item in fixed_objects):
            continue
        positions.append(np.array([candidate[0], candidate[1], TARGET_Z]))
    return positions


def run_trial(model, trial_id, target_position):
    started = time.perf_counter()
    data = mujoco.MjData(model)
    initialize_home(model, data)
    set_free_body_position(model, data, TARGET_NAME, target_position)
    controller = camera = None
    row = {
        "trial_id": trial_id,
        "start_x_m": float(target_position[0]),
        "start_y_m": float(target_position[1]),
        "perception_success": False,
        "localization_success": False,
        "planning_success": False,
        "execution_success": False,
        "e2e_success": False,
        "localization_xy_error_m": math.nan,
        "trajectory_points": 0,
        "moved_distance_m": math.nan,
        "place_error_m": math.nan,
        "failure_reason": "",
    }

    try:
        controller, camera, catalog = create_controller(model, data)
        target_body_id = next(
            item["body_id"]
            for item in catalog
            if item["class_name"] == TARGET_NAME
        )
        initial_xy = data.body(target_body_id).xpos[:2].copy()

        controller._perceive()
        row["perception_success"] = any(
            item["body_id"] == target_body_id for item in controller.detections
        )
        if not row["perception_success"]:
            raise RuntimeError("Target was not detected")

        controller.state = RobotState.LOCALIZE
        controller.localized_objects = controller.localizer.localize(
            controller.detections, controller.Depth, controller.camera_pose
        )
        target = next(
            (
                item
                for item in controller.localized_objects
                if item["body_id"] == target_body_id
            ),
            None,
        )
        if target is None:
            raise RuntimeError("Target was not localized")
        controller.target_object = target
        controller.object_position = target["position"].copy()
        row["localization_success"] = True
        row["localization_xy_error_m"] = float(
            np.linalg.norm(target["position"][:2] - initial_xy)
        )

        controller._plan()
        controller._check_reachability()
        row["planning_success"] = True
        row["trajectory_points"] = len(controller.trajectory)

        controller._execute()
        row["execution_success"] = True
        controller._verify()

        final_position = data.body(target_body_id).xpos.copy()
        row["moved_distance_m"] = float(
            np.linalg.norm(final_position[:2] - initial_xy)
        )
        row["place_error_m"] = float(
            np.linalg.norm(final_position[:2] - PLACE_XY)
        )
        row["e2e_success"] = row["place_error_m"] <= 0.05
    except Exception as error:
        row["failure_reason"] = str(error)
    finally:
        if camera is not None:
            camera.close()
        row["wall_time_s"] = time.perf_counter() - started
    return row


def wilson_interval(successes, total, z=1.96):
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z**2 / (4 * total**2)
    ) / denominator
    return center - margin, center + margin


def distribution(values):
    values = np.asarray(values, dtype=float)
    return {
        "n": int(values.size),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def write_results(rows, output_directory, seed):
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "randomized_trials.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    successes = sum(bool(row["e2e_success"]) for row in rows)
    lower, upper = wilson_interval(successes, len(rows))
    successful_rows = [row for row in rows if row["e2e_success"]]
    stage_names = (
        "perception_success",
        "localization_success",
        "planning_success",
        "execution_success",
        "e2e_success",
    )
    summary = {
        "seed": seed,
        "trial_count": len(rows),
        "success": {
            "count": successes,
            "rate": successes / len(rows),
            "wilson_95_ci": [lower, upper],
        },
        "stage_counts": {
            name: sum(bool(row[name]) for row in rows) for name in stage_names
        },
        "localization_xy_error_m": distribution(
            [row["localization_xy_error_m"] for row in rows]
        ),
        "successful_place_error_m": (
            distribution([row["place_error_m"] for row in successful_rows])
            if successful_rows
            else None
        ),
        "failure_counts": dict(
            Counter(
                row["failure_reason"]
                for row in rows
                if row["failure_reason"]
            )
        ),
    }
    json_path = output_directory / "evaluation_summary.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return csv_path, json_path, summary


def main():
    parser = argparse.ArgumentParser(description="Run randomized FR3 evaluations")
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "evaluation",
    )
    args = parser.parse_args()
    if args.trials <= 0:
        raise SystemExit("--trials must be positive")

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    positions = sample_positions(np.random.default_rng(args.seed), args.trials)
    rows = [
        run_trial(model, index, position)
        for index, position in enumerate(positions, start=1)
    ]
    csv_path, json_path, summary = write_results(rows, args.output_dir, args.seed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
