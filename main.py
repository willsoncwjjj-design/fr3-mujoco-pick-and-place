import argparse
import logging

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


def build_object_catalog(model):
    catalog = []
    for name in OBJECT_NAMES:
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
    # A full keyframe reset zero-fills free-joint object poses; set only the arm.
    data.qpos[:7] = model.key_qpos[key_id][:7]
    data.ctrl[:7] = data.qpos[:7]
    mujoco.mj_forward(model, data)


def run(target_name="cube_red", use_viewer=False):
    scene = SimScene()
    model, data = scene.setup("robot/franka_fr3/myscene.xml")
    initialize_home(model, data)

    viewer = Viewer(model, data) if use_viewer else None
    if viewer:
        viewer.attach(scene)

    camera = None
    try:
        catalog = build_object_catalog(model)
        target_body_id = next(
            item["body_id"]
            for item in catalog
            if item["class_name"] == target_name
        )

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

        controller = Controller(
            object_catalog=catalog,
            env=scene,
            robot=robot,
            camera=camera,
            detector=detector,
            localizer=localizer,
            grasp_planner=grasp_planner,
            trajectory_planner=trajectory_planner,
            gripper=gripper,
        )
        result = controller.run_cycle(target_body_id=target_body_id)
        logging.info("Result: %s", result)
        if viewer and viewer.is_running():
            viewer.wait_until_closed()
        return result
    finally:
        if camera is not None:
            camera.close()
        if viewer is not None:
            viewer.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one FR3 tabletop pick-and-place cycle."
    )
    parser.add_argument("--target", choices=OBJECT_NAMES, default="cube_red")
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="open the passive MuJoCo viewer (use mjpython on macOS)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = run(target_name=args.target, use_viewer=args.viewer)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
