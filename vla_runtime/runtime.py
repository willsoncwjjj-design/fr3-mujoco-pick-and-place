from dataclasses import dataclass
from typing import Optional

import numpy as np

from vla_runtime.contracts import ActionChunk, RGBFrame, VLAObservation
from vla_runtime.manager import ActionDispatch, ActionManager


class RuntimeObservationBuilder:
    """从现有 RobotRuntime 采集 VLA 所需的多模态观测。"""

    def __init__(self, runtime, camera_name="top_cam"):
        self.runtime = runtime
        self.camera_name = camera_name
        self.sequence_id = 0

    def capture(self, instruction, subgoal):
        if not self.runtime.is_open:
            raise RuntimeError("RobotRuntime is not open")
        rgb, _, _, _ = self.runtime.camera.capture()
        ee_position, ee_quaternion = (
            self.runtime.robot.get_end_effector_pose()
        )
        qpos = self.runtime.data.qpos[
            self.runtime.robot.arm_qposadr
        ].copy()
        qvel = self.runtime.data.qvel[
            self.runtime.ik_solver.arm_dofadr
        ].copy()
        finger_positions = self.runtime.data.qpos[
            self.runtime.gripper.finger_qposadr
        ]
        self.sequence_id += 1
        return VLAObservation(
            observation_id=self.sequence_id,
            timestamp_s=float(self.runtime.data.time),
            frames=(RGBFrame(self.camera_name, rgb),),
            qpos=tuple(qpos),
            qvel=tuple(qvel),
            ee_position=tuple(ee_position),
            ee_quaternion=tuple(ee_quaternion),
            gripper_position=float(np.mean(finger_positions)),
            instruction=instruction,
            subgoal=subgoal,
        )


@dataclass(frozen=True)
class ActionExecutionResult:
    """一个动作从分发到控制器执行后的结果。"""

    dispatch: ActionDispatch
    success: bool
    clipped: bool = False
    error_message: Optional[str] = None


class ActionChunkRuntime:
    """连接观测、动作块管理、动作适配与低层控制器。"""

    def __init__(
        self,
        manager: ActionManager,
        adapter,
        controller,
        observation_builder=None,
        control_steps_per_action=10,
    ):
        if control_steps_per_action < 1:
            raise ValueError("control_steps_per_action must be positive")
        self.manager = manager
        self.adapter = adapter
        self.controller = controller
        self.observation_builder = observation_builder
        self.control_steps_per_action = int(control_steps_per_action)

    def observe(self, instruction, subgoal):
        if self.observation_builder is None:
            raise RuntimeError("An observation builder is required")
        observation = self.observation_builder.capture(instruction, subgoal)
        self.manager.update_observation(observation.observation_id)
        return observation

    def submit_chunk(self, chunk: ActionChunk):
        return self.manager.submit(chunk)

    def step(self):
        dispatch = self.manager.dispatch_next()
        if dispatch is None:
            return None
        try:
            command = self.adapter.adapt(dispatch.action)
            self.controller.execute_joint_target(
                command.joint_target,
                gripper_command=command.gripper_command,
                control_steps=self.control_steps_per_action,
            )
        except Exception as error:
            self.manager.complete_action(
                dispatch,
                success=False,
                reason=str(error),
            )
            return ActionExecutionResult(
                dispatch=dispatch,
                success=False,
                error_message=str(error),
            )
        self.manager.complete_action(dispatch, success=True)
        return ActionExecutionResult(
            dispatch=dispatch,
            success=True,
            clipped=command.clipped,
        )

    def run_until_idle(self, max_actions=100):
        if max_actions < 1:
            raise ValueError("max_actions must be positive")
        results = []
        for _ in range(max_actions):
            result = self.step()
            if result is None:
                break
            results.append(result)
            if not result.success:
                break
        else:
            raise RuntimeError("Action runtime exceeded max_actions")
        return tuple(results)
