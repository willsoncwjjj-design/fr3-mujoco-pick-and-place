import mujoco
import numpy as np


class PandaRobot:
    ARM_JOINT_NAMES = tuple(f"fr3_joint{i}" for i in range(1, 8))
    END_EFFECTOR_SITE = "attachment_site"

    def __init__(self, model, data):
        self.model = model
        self.data = data

        self.arm_joint_ids = [self._joint_id(name) for name in self.ARM_JOINT_NAMES]
        self.arm_qposadr = [int(model.jnt_qposadr[jid]) for jid in self.arm_joint_ids]
        self.arm_act_ids = [self._actuator_for_joint(jid) for jid in self.arm_joint_ids]
        self.joint_limits = [tuple(model.jnt_range[jid]) for jid in self.arm_joint_ids]

        self.ee_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, self.END_EFFECTOR_SITE
        )
        if self.ee_id < 0:
            raise ValueError(f"Site '{self.END_EFFECTOR_SITE}' does not exist")

    def _joint_id(self, name):
        # A missing name returns -1; using it as an index corrupts qpos[-1].
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Joint '{name}' does not exist")
        return joint_id

    def _actuator_for_joint(self, joint_id):
        for actuator_id in range(self.model.nu):
            if (
                self.model.actuator_trntype[actuator_id]
                == mujoco.mjtTrn.mjTRN_JOINT
                and self.model.actuator_trnid[actuator_id, 0] == joint_id
            ):
                return actuator_id
        raise ValueError(f"Joint id={joint_id} has no actuator")

    def set_joint_positions(self, positions):
        positions = self._validate_positions(positions)
        for address, position in zip(
            self.arm_qposadr, positions, strict=True
        ):
            self.data.qpos[address] = position
        mujoco.mj_forward(self.model, self.data)

    def set_ctrl(self, positions):
        """Set position-servo targets without teleporting the simulated robot."""
        positions = self._validate_positions(positions)
        for actuator_id, position in zip(
            self.arm_act_ids, positions, strict=True
        ):
            self.data.ctrl[actuator_id] = position

    def _validate_positions(self, positions):
        values = np.asarray(positions, dtype=float)
        if values.shape != (len(self.arm_joint_ids),):
            raise ValueError(
                f"Expected {len(self.arm_joint_ids)} joint values, got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("Joint positions contain non-finite values")
        return values

    def get_end_effector_pose(self):
        position = self.data.site(self.ee_id).xpos.copy()
        rotation = self.data.site(self.ee_id).xmat.copy()
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, rotation)
        return position, quaternion

    def get_end_effect_position(self):
        """Backward-compatible alias for earlier project versions."""
        return self.get_end_effector_pose()
