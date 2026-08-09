import mujoco
import numpy as np


class DLSIKSolver:
    """用于 FR3 七个机械臂关节的阻尼最小二乘逆运动学。"""

    ARM_JOINT_NAMES = tuple(f"fr3_joint{i}" for i in range(1, 8))

    def __init__(
        self,
        robot_model,
        robot_data,
        ee_name,
        lambda_damping=0.1,
        max_iterations=200,
        tolerance=1e-5,
    ):
        self.model = robot_model
        self.sim_data = robot_data
        # IK 使用独立 MjData，避免可行性检查修改仿真状态。
        self.data = mujoco.MjData(robot_model)
        self.lambda_damping = float(lambda_damping)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)

        self.ee_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, ee_name
        )
        if self.ee_id < 0:
            raise ValueError(f"Site '{ee_name}' does not exist")

        self.arm_joint_ids = [
            self._joint_id(name) for name in self.ARM_JOINT_NAMES
        ]
        self.arm_qposadr = np.array(
            [self.model.jnt_qposadr[jid] for jid in self.arm_joint_ids], dtype=int
        )
        self.arm_dofadr = np.array(
            [self.model.jnt_dofadr[jid] for jid in self.arm_joint_ids], dtype=int
        )

    def _joint_id(self, name):
        joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if joint_id < 0:
            raise ValueError(f"Joint '{name}' does not exist")
        return joint_id

    @property
    def robot_qpos(self):
        return self.sim_data.qpos[self.arm_qposadr].copy()

    def solve(self, target_pos, target_quat=None, initial_pos=None):
        target_pos = np.asarray(target_pos, dtype=float)
        if target_pos.shape != (3,):
            raise ValueError("target_pos must contain three values")

        initial = self.robot_qpos if initial_pos is None else np.asarray(initial_pos)
        if initial.shape != (7,):
            raise ValueError("initial_pos must contain seven joint values")
        self.data.qpos[self.arm_qposadr] = initial
        mujoco.mj_forward(self.model, self.data)

        target_rotation = self._rotation_from_quaternion(target_quat)
        for _ in range(self.max_iterations):
            error, jacobian = self._error_and_jacobian(target_pos, target_rotation)
            if np.linalg.norm(error) < self.tolerance:
                return self._limit_aware_result(target_pos)

            # DLS：dq = J^T (J J^T + lambda^2 I)^-1 error。
            jj_t = jacobian @ jacobian.T
            regularized = jj_t + self.lambda_damping**2 * np.eye(jj_t.shape[0])
            delta_q = jacobian.T @ np.linalg.solve(regularized, error)
            self.data.qpos[self.arm_qposadr] += delta_q
            mujoco.mj_forward(self.model, self.data)
        return None

    def _error_and_jacobian(self, target_pos, target_rotation):
        current_site = self.data.site(self.ee_id)
        position_error = target_pos - current_site.xpos
        jac_pos = np.zeros((3, self.model.nv))

        if target_rotation is None:
            mujoco.mj_jacSite(
                self.model, self.data, jac_pos, None, self.ee_id
            )
            return position_error, jac_pos[:, self.arm_dofadr]

        current_rotation = current_site.xmat.reshape(3, 3)
        rotation_error = target_rotation @ current_rotation.T
        error_quaternion = np.empty(4)
        mujoco.mju_mat2Quat(error_quaternion, rotation_error.ravel())
        axis_angle = np.empty(3)
        mujoco.mju_quat2Vel(axis_angle, error_quaternion, 1.0)

        jac_rot = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(
            self.model, self.data, jac_pos, jac_rot, self.ee_id
        )
        error = np.concatenate((position_error, axis_angle))
        jacobian = np.vstack((jac_pos, jac_rot))[:, self.arm_dofadr]
        return error, jacobian

    @staticmethod
    def _rotation_from_quaternion(quaternion):
        if quaternion is None:
            return None
        quaternion = np.asarray(quaternion, dtype=float)
        if quaternion.shape != (4,):
            raise ValueError("target_quat must contain four values")
        norm = np.linalg.norm(quaternion)
        if norm < 1e-12:
            raise ValueError("target_quat must be non-zero")
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, quaternion / norm)
        return rotation.reshape(3, 3)

    def _limit_aware_result(self, target_pos):
        solution = self.data.qpos[self.arm_qposadr].copy()
        limits = self.model.jnt_range[self.arm_joint_ids]
        if np.all((solution >= limits[:, 0]) & (solution <= limits[:, 1])):
            return solution

        # 在迭代循环内执行限幅可能导致结果在关节限位附近振荡。
        clamped = np.clip(solution, limits[:, 0], limits[:, 1])
        self.data.qpos[self.arm_qposadr] = clamped
        mujoco.mj_forward(self.model, self.data)
        position_error = np.linalg.norm(
            target_pos - self.data.site(self.ee_id).xpos
        )
        return clamped.copy() if position_error < 0.02 else None
