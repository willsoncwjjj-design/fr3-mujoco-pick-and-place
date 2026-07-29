import mujoco
import numpy as np


class SimCamera:
    """Render aligned RGB, depth, and segmentation frames from a fixed camera."""

    def __init__(self, model, data, height=480, width=640, camera_name="top_cam"):
        self.model = model
        self.data = data
        self.height = height
        self.width = width
        self.renderer = mujoco.Renderer(model, height, width)

        self.cam_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
        )
        if self.cam_id < 0:
            raise ValueError(f"Camera '{camera_name}' does not exist in the model")

        fovy_deg = model.cam_fovy[self.cam_id]
        if fovy_deg <= 0:
            fovy_deg = model.vis.global_.fovy
        focal_length = (height / 2) / np.tan(np.radians(fovy_deg) / 2)
        self.camera_matrix = np.array(
            [
                [focal_length, 0, width / 2],
                [0, focal_length, height / 2],
                [0, 0, 1],
            ]
        )

    def capture(self):
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera.fixedcamid = self.cam_id
        self.renderer.update_scene(self.data, camera)

        camera_pose = self.get_camera_pose()
        rgb = self.renderer.render()

        self.renderer.enable_depth_rendering()
        try:
            depth = self.renderer.render()
        finally:
            self.renderer.disable_depth_rendering()

        self.renderer.enable_segmentation_rendering()
        try:
            segmentation = self.renderer.render()
        finally:
            self.renderer.disable_segmentation_rendering()

        return rgb, depth, segmentation, camera_pose

    def get_camera_pose(self):
        # cam_xmat columns follow OpenGL camera axes: right, up, backward.
        return {
            "position": self.data.cam_xpos[self.cam_id].copy(),
            "rotation": self.data.cam_xmat[self.cam_id].copy().reshape(3, 3),
        }

    def close(self):
        self.renderer.close()
