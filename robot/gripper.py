import mujoco


class PandaGripper:
    """Control both fingers through MuJoCo position servos."""

    OPEN_POSITION = 0.04
    CLOSED_POSITION = 0.0
    FINGER_JOINT_NAMES = ("finger_joint1", "finger_joint2")

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.act_ids = []
        self.finger_qposadr = []

        for joint_name in self.FINGER_JOINT_NAMES:
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if joint_id < 0:
                raise ValueError(f"Joint '{joint_name}' does not exist")
            self.finger_qposadr.append(int(model.jnt_qposadr[joint_id]))
            self.act_ids.append(self._actuator_for_joint(joint_id))

    def _actuator_for_joint(self, joint_id):
        for actuator_id in range(self.model.nu):
            if (
                self.model.actuator_trntype[actuator_id]
                == mujoco.mjtTrn.mjTRN_JOINT
                and self.model.actuator_trnid[actuator_id, 0] == joint_id
            ):
                return actuator_id
        raise ValueError(f"Joint id={joint_id} has no actuator")

    def execute(self, command):
        if command is None:
            return
        targets = {
            "open": self.OPEN_POSITION,
            "close": self.CLOSED_POSITION,
        }
        if command not in targets:
            raise ValueError(f"Unknown gripper command: {command}")
        for actuator_id in self.act_ids:
            self.data.ctrl[actuator_id] = targets[command]

    def open(self):
        self.execute("open")

    def close(self):
        self.execute("close")
