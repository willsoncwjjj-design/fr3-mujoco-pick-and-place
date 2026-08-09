from contextlib import ExitStack
from math import isfinite

import mujoco

from control.controller import Controller
from perception.camera import SimCamera
from perception.detector import Detector
from perception.localizer import ObjectLocalizer
from planning.grasp_planner import GraspPlanner
from planning.trajectory_planner import TrajectoryPlanner
from robot.gripper import PandaGripper
from robot.kinematics import DLSIKSolver
from robot.panda import PandaRobot
from simulation.scene import SimScene
from view.viewer import Viewer

OBJECT_NAMES = ("cube_red", "cube_blue", "cyl_green", "sphere_yellow")
DEFAULT_SCENE_PATH = "robot/franka_fr3/myscene.xml"


def build_object_catalog(model, object_names=OBJECT_NAMES):
    catalog = []
    for name in object_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"Body '{name}' does not exist")
        catalog.append(
            {"class_name": name, "body_name": name, "body_id": body_id}
        )
    return catalog


def initialize_home(model, data):
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id < 0:
        raise ValueError("Home keyframe does not exist")
    data.qpos[:7] = model.key_qpos[key_id][:7]
    data.ctrl[:7] = data.qpos[:7]
    mujoco.mj_forward(model, data)


class RobotRuntime:
    def __init__(
        self,
        scene_path=DEFAULT_SCENE_PATH,
        use_viewer=False,
        object_names=OBJECT_NAMES,
        settle_steps=200,
    ):
        if settle_steps < 0:
            raise ValueError("settle_steps must not be negative")
        self.scene_path = scene_path
        self.use_viewer = use_viewer
        self.object_names = tuple(object_names)
        self.settle_steps = int(settle_steps)

        self.scene = None
        self.model = None
        self.data = None
        self.viewer = None
        self.camera = None
        self.object_catalog = None
        self.robot = None
        self.detector = None
        self.localizer = None
        self.ik_solver = None
        self.grasp_planner = None
        self.trajectory_planner = None
        self.gripper = None
        self.controller = None
        self.is_open = False
        self._resources = ExitStack()

    def open(self):
        if self.is_open:
            raise RuntimeError("RobotRuntime is already open")
        self._resources = ExitStack()
        try:
            self.scene = SimScene()
            self.model, self.data = self.scene.setup(self.scene_path)
            initialize_home(self.model, self.data)

            if self.use_viewer:
                self.viewer = Viewer(self.model, self.data)
                self._resources.callback(self.viewer.close)
                self.viewer.attach(self.scene)

            self.object_catalog = build_object_catalog(
                self.model, self.object_names
            )
            self.robot = PandaRobot(self.model, self.data)
            self.camera = SimCamera(self.model, self.data)
            self._resources.callback(self.camera.close)
            self.detector = Detector(self.model)
            self.localizer = ObjectLocalizer(self.camera.camera_matrix)
            self.ik_solver = DLSIKSolver(
                self.model, self.data, "attachment_site", 0.1
            )
            self.grasp_planner = GraspPlanner(
                self.model,
                self.data,
                ik_solver=self.ik_solver,
            )
            self.trajectory_planner = TrajectoryPlanner(self.ik_solver)
            self.gripper = PandaGripper(self.model, self.data)

            self.gripper.open()
            for _ in range(self.settle_steps):
                self.scene.step()

            self.controller = Controller(
                object_catalog=self.object_catalog,
                env=self.scene,
                robot=self.robot,
                camera=self.camera,
                detector=self.detector,
                localizer=self.localizer,
                grasp_planner=self.grasp_planner,
                trajectory_planner=self.trajectory_planner,
                gripper=self.gripper,
            )
            self.is_open = True
            return self
        except Exception:
            self.close()
            raise

    def close(self):
        try:
            self._resources.close()
        finally:
            self.camera = None
            self.viewer = None
            self.controller = None
            self.is_open = False

    def body_id_for(self, object_name):
        if not self.is_open or self.object_catalog is None:
            raise RuntimeError("RobotRuntime is not open")
        try:
            return next(
                item["body_id"]
                for item in self.object_catalog
                if item["class_name"] == object_name
            )
        except StopIteration as error:
            raise KeyError(f"Unknown runtime object: {object_name}") from error

    def set_free_body_xy(self, object_name, xy):
        if not self.is_open or self.model is None or self.data is None:
            raise RuntimeError("RobotRuntime is not open")
        try:
            values = tuple(float(value) for value in xy)
        except (TypeError, ValueError) as error:
            raise ValueError("xy must contain two finite values") from error
        if len(values) != 2 or not all(isfinite(value) for value in values):
            raise ValueError("xy must contain two finite values")

        body_id = self.body_id_for(object_name)
        joint_id = int(self.model.body_jntadr[body_id])
        if joint_id < 0 or self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"Body '{object_name}' does not have a free joint")
        qpos_address = int(self.model.jnt_qposadr[joint_id])
        dof_address = int(self.model.jnt_dofadr[joint_id])
        self.data.qpos[qpos_address : qpos_address + 2] = values
        self.data.qvel[dof_address : dof_address + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
