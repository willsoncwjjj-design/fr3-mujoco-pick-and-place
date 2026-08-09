from math import cos, sin

import mujoco
import numpy as np

from vla_runtime.contracts import DeltaEEAction, FR3Command


class ActionAdapterError(ValueError):
    """VLA 动作无法安全转换为 FR3 命令。"""


class DeltaEEActionAdapter:
    """将末端增量动作转换为经过约束检查的 FR3 关节目标。"""

    def __init__(
        self,
        robot,
        ik_solver,
        max_translation_m=0.03,
        max_rotation_rad=0.20,
        workspace_low=None,
        workspace_high=None,
    ):
        if max_translation_m <= 0:
            raise ValueError("max_translation_m must be positive")
        if max_rotation_rad <= 0:
            raise ValueError("max_rotation_rad must be positive")
        self.robot = robot
        self.ik_solver = ik_solver
        self.max_translation_m = float(max_translation_m)
        self.max_rotation_rad = float(max_rotation_rad)
        self.workspace_low = self._workspace_bound(
            workspace_low,
            (-np.inf, -np.inf, -np.inf),
            "workspace_low",
        )
        self.workspace_high = self._workspace_bound(
            workspace_high,
            (np.inf, np.inf, np.inf),
            "workspace_high",
        )
        if np.any(self.workspace_low >= self.workspace_high):
            raise ValueError("workspace_low must be below workspace_high")

    def adapt(self, action: DeltaEEAction) -> FR3Command:
        current_position, current_quaternion = (
            self.robot.get_end_effector_pose()
        )
        translation, translation_clipped = self._limit_norm(
            action.delta_position,
            self.max_translation_m,
        )
        rotation, rotation_clipped = self._limit_norm(
            action.delta_rotation,
            self.max_rotation_rad,
        )
        target_position = np.asarray(current_position, dtype=float) + translation
        if np.any(target_position < self.workspace_low) or np.any(
            target_position > self.workspace_high
        ):
            raise ActionAdapterError("Target end-effector position is out of bounds")

        target_quaternion = self._apply_world_rotation(
            current_quaternion,
            rotation,
        )
        solution = self.ik_solver.solve(
            target_position,
            target_quat=target_quaternion,
            initial_pos=self.ik_solver.robot_qpos,
        )
        if solution is None:
            raise ActionAdapterError("Delta end-effector action is not IK feasible")
        solution = np.asarray(solution, dtype=float)
        if solution.shape != (7,) or not np.all(np.isfinite(solution)):
            raise ActionAdapterError("IK returned an invalid joint target")
        limits = np.asarray(self.robot.joint_limits, dtype=float)
        if limits.shape != (7, 2) or np.any(solution < limits[:, 0]) or np.any(
            solution > limits[:, 1]
        ):
            raise ActionAdapterError("IK solution violates FR3 joint limits")

        return FR3Command(
            joint_target=tuple(solution),
            gripper_command=action.gripper_command,
            target_ee_position=tuple(target_position),
            target_ee_quaternion=tuple(target_quaternion),
            clipped=translation_clipped or rotation_clipped,
        )

    @staticmethod
    def _limit_norm(values, maximum):
        vector = np.asarray(values, dtype=float)
        norm = float(np.linalg.norm(vector))
        if norm <= maximum:
            return vector, False
        return vector * (maximum / norm), True

    @staticmethod
    def _apply_world_rotation(current_quaternion, rotation_vector):
        angle = float(np.linalg.norm(rotation_vector))
        if angle < 1e-12:
            delta_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            axis = np.asarray(rotation_vector, dtype=float) / angle
            delta_quaternion = np.concatenate(
                ([cos(angle / 2.0)], axis * sin(angle / 2.0))
            )
        target = np.empty(4)
        mujoco.mju_mulQuat(
            target,
            delta_quaternion,
            np.asarray(current_quaternion, dtype=float),
        )
        norm = float(np.linalg.norm(target))
        if norm < 1e-12:
            raise ActionAdapterError("Target quaternion is invalid")
        return target / norm

    @staticmethod
    def _workspace_bound(value, default, field_name):
        bound = np.asarray(default if value is None else value, dtype=float)
        if bound.shape != (3,) or np.any(np.isnan(bound)):
            raise ValueError(f"{field_name} must contain three valid values")
        return bound
