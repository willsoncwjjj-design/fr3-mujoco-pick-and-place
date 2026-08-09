import numpy as np


class ObjectLocalizer:
    """将分割区域内的深度像素反投影到世界坐标系。"""

    TOP_SURFACE_BAND_M = 0.02

    def __init__(self, camera_matrix):
        self.K = np.asarray(camera_matrix, dtype=float)

    def localize(self, detections, depth, camera_pose):
        results = []
        camera_position = camera_pose["position"]
        camera_rotation = camera_pose["rotation"]

        for detection in detections:
            x0, y0, x1, y1 = detection["bbox"]
            roi = depth[y0 : y1 + 1, x0 : x1 + 1]
            valid = roi[roi > 0]
            if not valid.size:
                continue

            nearest_depth = float(valid.min())
            top_mask = (roi > 0) & (
                roi < nearest_depth + self.TOP_SURFACE_BAND_M
            )
            ys, xs = np.where(top_mask)
            if not len(xs):
                continue

            u = int(xs.mean()) + x0
            v = int(ys.mean()) + y0
            point_cv = self.pixel_to_camera(u, v, nearest_depth)

            # 针孔反投影使用 OpenCV 坐标轴（右、下、前），
            # MuJoCo 相机位姿使用 OpenGL 坐标轴（右、上、后）。
            point_gl = point_cv * np.array([1.0, -1.0, -1.0])
            point_world = self.camera_to_world(
                point_gl, camera_position, camera_rotation
            )
            results.append(
                {
                    "class_name": detection["class_name"],
                    "body_id": detection["body_id"],
                    "position": point_world,
                }
            )
        return results

    def pixel_to_camera(self, u, v, depth):
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        return np.array(
            [
                (u - cx) / fx * depth,
                (v - cy) / fy * depth,
                depth,
            ]
        )

    @staticmethod
    def camera_to_world(point_camera, camera_position, camera_rotation):
        return camera_rotation @ point_camera + camera_position
