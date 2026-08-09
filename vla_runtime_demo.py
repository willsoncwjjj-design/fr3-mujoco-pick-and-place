import argparse
import json
from dataclasses import asdict

import numpy as np

from simulation.runtime import RobotRuntime
from vla_runtime.adapter import DeltaEEActionAdapter
from vla_runtime.contracts import DeltaEEAction
from vla_runtime.manager import ActionManager
from vla_runtime.policies import ScriptedPolicy
from vla_runtime.runtime import ActionChunkRuntime, RuntimeObservationBuilder


def run_demo(use_viewer=False):
    instruction = "根据最新视觉状态移动机械臂，并取消过期动作"
    with RobotRuntime(use_viewer=use_viewer) as robot_runtime:
        manager = ActionManager()
        adapter = DeltaEEActionAdapter(
            robot_runtime.robot,
            robot_runtime.ik_solver,
            max_translation_m=0.01,
            max_rotation_rad=0.10,
        )
        runtime = ActionChunkRuntime(
            manager,
            adapter,
            robot_runtime.controller,
            RuntimeObservationBuilder(robot_runtime),
        )
        action_up = DeltaEEAction((0.0, 0.0, 0.005), (0.0, 0.0, 0.0))
        action_side = DeltaEEAction((0.0, 0.005, 0.0), (0.0, 0.0, 0.0))
        policy = ScriptedPolicy(
            {
                1: (action_up, action_up, action_up),
                2: (action_side, action_side),
            }
        )

        start_position, _ = robot_runtime.robot.get_end_effector_pose()
        observation_1 = runtime.observe(instruction, "先沿 Z 轴移动")
        runtime.submit_chunk(policy.infer(observation_1))
        first_result = runtime.step()

        observation_2 = runtime.observe(instruction, "改为沿 Y 轴移动")
        runtime.submit_chunk(policy.infer(observation_2))
        remaining_results = runtime.run_until_idle()
        end_position, _ = robot_runtime.robot.get_end_effector_pose()

        if robot_runtime.viewer and robot_runtime.viewer.is_running():
            robot_runtime.viewer.wait_until_closed()

    events = tuple(asdict(event) for event in manager.events)
    executed = (first_result,) + remaining_results
    return {
        "success": all(result is not None and result.success for result in executed),
        "executed_actions": len(executed),
        "preempted_actions": next(
            (
                event["remaining_actions"]
                for event in events
                if event["event_type"] == "chunk_cancelled"
            ),
            0,
        ),
        "start_ee_position": np.asarray(start_position).tolist(),
        "end_ee_position": np.asarray(end_position).tolist(),
        "events": events,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the deterministic FR3 action-chunk preemption demo."
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="open the passive MuJoCo viewer (use mjpython on macOS)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_demo(args.viewer)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["success"] and result["preempted_actions"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
