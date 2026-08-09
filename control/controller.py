import logging
from enum import Enum, auto

import mujoco
import numpy as np

LOGGER = logging.getLogger(__name__)


class RobotState(Enum):
    IDLE = auto()
    PERCEIVE = auto()
    LOCALIZE = auto()
    PLAN = auto()
    CHECK_REACHABILITY = auto()
    EXECUTE = auto()
    VERIFY = auto()
    FAIL = auto()
    DONE = auto()


FAILURE_CODE_BY_STATE = {
    RobotState.IDLE: "invalid_request",
    RobotState.PERCEIVE: "object_missing",
    RobotState.LOCALIZE: "object_missing",
    RobotState.PLAN: "ik_failed",
    RobotState.CHECK_REACHABILITY: "ik_failed",
    RobotState.EXECUTE: "pick_failed",
    RobotState.VERIFY: "verification_failed",
}


class Controller:
    GRIPPER_SETTLE_STEPS = 100
    STEPS_PER_TRAJECTORY_POINT = 10
    MINIMUM_OBJECT_MOVEMENT_M = 0.02
    MAXIMUM_PLACE_ERROR_M = 0.05

    def __init__(
        self,
        object_catalog,
        env,
        robot,
        camera,
        detector,
        localizer,
        grasp_planner,
        trajectory_planner,
        gripper,
    ):
        self.object_catalog = object_catalog
        self.env = env
        self.robot = robot
        self.camera = camera
        self.detector = detector
        self.localizer = localizer
        self.grasp_planner = grasp_planner
        self.trajectory_planner = trajectory_planner
        self.gripper = gripper

        self.state = RobotState.IDLE
        self.RGB = None
        self.Depth = None
        self.Segmentation = None
        self.camera_pose = None
        self.detections = None
        self.localized_objects = None
        self.target_object = None
        self.object_position = None
        self.grasp_pose = None
        self.trajectory = None
        self.gripper_cmds = None
        self.target_place_xy = np.asarray(
            self.grasp_planner.place_xy, dtype=float
        ).copy()
        self.result = None
        self.error_info = None
        self.verification_metrics = None

    def run_cycle(
        self,
        target_object_idx=0,
        target_body_id=None,
        place_xy=None,
    ):
        self.state = RobotState.IDLE
        self.verification_metrics = None
        try:
            self.target_place_xy = self._resolve_place_xy(place_xy)
            self._perceive()
            self._localize(target_object_idx, target_body_id)
            self._plan()
            self._check_reachability()
            self._execute()
            self._verify()
        except Exception as error:
            LOGGER.exception("Robot cycle failed in state %s", self.state.name)
            return self._fail(str(error))

        self.state = RobotState.DONE
        self.result = {
            "success": True,
            "state": self.state,
            "target_object": self.target_object,
            "place_xy": self.target_place_xy.copy(),
            "failure_code": None,
            "failed_state": None,
            "verification": self.verification_metrics,
        }
        return self.result

    def _resolve_place_xy(self, place_xy):
        target = self.grasp_planner.place_xy if place_xy is None else place_xy
        values = np.asarray(target, dtype=float)
        if values.shape != (2,) or not np.all(np.isfinite(values)):
            raise ValueError("place_xy must contain two finite values")
        return values.copy()

    def _perceive(self):
        self.state = RobotState.PERCEIVE
        self.RGB, self.Depth, self.Segmentation, self.camera_pose = (
            self.camera.capture()
        )
        self.detections = self.detector.detect_sim(
            self.Segmentation, self.object_catalog
        )
        if not self.detections:
            raise RuntimeError("No objects detected")
        LOGGER.info("Detected %d objects", len(self.detections))

    def _localize(self, target_object_idx=0, target_body_id=None):
        self.state = RobotState.LOCALIZE
        self.localized_objects = self.localizer.localize(
            self.detections, self.Depth, self.camera_pose
        )
        if not self.localized_objects:
            raise RuntimeError("No objects localized")
        if target_body_id is not None:
            self.target_object = next(
                (
                    item
                    for item in self.localized_objects
                    if item["body_id"] == target_body_id
                ),
                None,
            )
            if self.target_object is None:
                raise RuntimeError(f"Target body {target_body_id} was not localized")
        else:
            if not 0 <= target_object_idx < len(self.localized_objects):
                raise IndexError(
                    f"Invalid target object index: {target_object_idx}"
                )
            self.target_object = self.localized_objects[target_object_idx]
        self.object_position = self.target_object["position"].copy()
        LOGGER.info(
            "Selected %s at %s",
            self.target_object["class_name"],
            np.round(self.object_position, 4),
        )

    def _plan(self):
        self.state = RobotState.PLAN
        self.grasp_pose = self.grasp_planner.compute_grasp_pose(
            self.object_position,
            place_xy=self.target_place_xy,
        )
        if not self.grasp_planner.check_ik_feasible(self.grasp_pose):
            raise RuntimeError("IK feasibility check failed")

        self.trajectory, self.gripper_cmds = self.trajectory_planner.plan(
            self.grasp_pose
        )
        if not self.trajectory or self.gripper_cmds is None:
            raise RuntimeError("Trajectory planning failed")
        LOGGER.info("Planned %d trajectory points", len(self.trajectory))

    def _check_reachability(self):
        self.state = RobotState.CHECK_REACHABILITY
        if len(self.trajectory) != len(self.gripper_cmds):
            raise RuntimeError("Trajectory and command lengths differ")
        if not all(self._is_valid_qpos(qpos) for qpos in self.trajectory):
            raise RuntimeError("Trajectory contains invalid joint values")
        if not self._check_joint_limits():
            raise RuntimeError("Trajectory violates joint limits")

    def _execute(self):
        self.state = RobotState.EXECUTE
        for qpos, command in zip(
            self.trajectory, self.gripper_cmds, strict=True
        ):
            # 位置伺服能够保留物理接触，直接写入 qpos 则不能。
            self.robot.set_ctrl(qpos)
            if command is not None:
                self.gripper.execute(command)
                self._step(self.GRIPPER_SETTLE_STEPS)
            self._step(self.STEPS_PER_TRAJECTORY_POINT)

    def execute_joint_target(
        self,
        target_qpos,
        gripper_command=None,
        control_steps=10,
    ):
        """执行一个经过上层适配的关节目标，不启动完整抓放状态机。"""
        values = np.asarray(target_qpos, dtype=float)
        if values.shape != (7,) or not np.all(np.isfinite(values)):
            raise ValueError("target_qpos must contain seven finite values")
        limits = np.asarray(self.robot.joint_limits, dtype=float)
        if limits.shape != (7, 2) or np.any(values < limits[:, 0]) or np.any(
            values > limits[:, 1]
        ):
            raise ValueError("target_qpos violates joint limits")
        if gripper_command not in {None, "open", "close"}:
            raise ValueError("Invalid gripper command")
        if int(control_steps) != control_steps or control_steps < 1:
            raise ValueError("control_steps must be a positive integer")
        self.robot.set_ctrl(values)
        self.gripper.execute(gripper_command)
        self._step(int(control_steps))
        return {
            "success": True,
            "joint_target": values.copy(),
            "gripper_command": gripper_command,
            "control_steps": int(control_steps),
        }

    def _verify(self):
        self.state = RobotState.VERIFY
        self._retreat_to_home()
        if not self._verify_grasp_success():
            raise RuntimeError("Placement verification failed")

    def _retreat_to_home(self, lift_height=0.15):
        """返回初始位前先垂直抬升，避免机械臂横向扫动物体。"""
        end_effector_position, _ = self.robot.get_end_effector_pose()
        lift_target = end_effector_position + np.array([0.0, 0.0, lift_height])
        lift_qpos = self.grasp_planner.ik_solver.solve(
            lift_target, target_quat=np.array([0.0, 1.0, 0.0, 0.0])
        )
        if lift_qpos is None:
            raise RuntimeError("Safe retreat lift IK failed")
        self._servo_to(lift_qpos, interpolation_points=60, steps_per_point=4)

        key_id = mujoco.mj_name2id(
            self.robot.model, mujoco.mjtObj.mjOBJ_KEY, "home"
        )
        if key_id < 0:
            raise RuntimeError("Home keyframe does not exist")
        home_qpos = self.robot.model.key_qpos[key_id][
            self.robot.arm_qposadr
        ]
        self._servo_to(home_qpos, interpolation_points=100, steps_per_point=5)

    def _servo_to(self, target_qpos, interpolation_points, steps_per_point):
        current = self.robot.data.qpos[self.robot.arm_qposadr].copy()
        for qpos in np.linspace(current, target_qpos, interpolation_points):
            self.robot.set_ctrl(qpos)
            self._step(steps_per_point)

    def _step(self, count):
        for _ in range(count):
            self.env.step()

    @staticmethod
    def _is_valid_qpos(qpos):
        values = np.asarray(qpos)
        return values.shape == (7,) and np.all(np.isfinite(values))

    def _check_joint_limits(self):
        limits = np.asarray(self.robot.joint_limits)
        return all(
            len(qpos) == len(limits)
            and np.all(np.asarray(qpos) >= limits[:, 0])
            and np.all(np.asarray(qpos) <= limits[:, 1])
            for qpos in self.trajectory
        )

    def _verify_grasp_success(self):
        _, depth, segmentation, camera_pose = self.camera.capture()
        detections = self.detector.detect_sim(
            segmentation, self.object_catalog
        )
        localized = self.localizer.localize(detections, depth, camera_pose)
        current = next(
            (
                obj["position"]
                for obj in localized
                if obj["body_id"] == self.target_object["body_id"]
            ),
            None,
        )
        if current is None:
            self.verification_metrics = {
                "target_visible": False,
                "moved_distance_m": None,
                "place_error_m": None,
                "minimum_movement_m": self.MINIMUM_OBJECT_MOVEMENT_M,
                "maximum_place_error_m": self.MAXIMUM_PLACE_ERROR_M,
            }
            LOGGER.warning("Target is not visible after execution")
            return False

        moved_distance = float(
            np.linalg.norm(current[:2] - self.object_position[:2])
        )
        place_error = float(
            np.linalg.norm(current[:2] - self.target_place_xy)
        )
        self.verification_metrics = {
            "target_visible": True,
            "moved_distance_m": moved_distance,
            "place_error_m": place_error,
            "minimum_movement_m": self.MINIMUM_OBJECT_MOVEMENT_M,
            "maximum_place_error_m": self.MAXIMUM_PLACE_ERROR_M,
        }
        LOGGER.info(
            "Verification: moved=%.1f mm, place error=%.1f mm",
            moved_distance * 1000,
            place_error * 1000,
        )
        return (
            moved_distance >= self.MINIMUM_OBJECT_MOVEMENT_M
            and place_error <= self.MAXIMUM_PLACE_ERROR_M
        )

    def _fail(self, error_message):
        failed_state = self.state
        failure_code = FAILURE_CODE_BY_STATE.get(
            failed_state,
            "internal_error",
        )
        self.state = RobotState.FAIL
        self.error_info = error_message
        self.result = {
            "success": False,
            "state": self.state,
            "error_message": error_message,
            "failure_code": failure_code,
            "failed_state": failed_state.name,
            "verification": self.verification_metrics,
        }
        return self.result
