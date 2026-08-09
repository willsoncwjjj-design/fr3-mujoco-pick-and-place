import numpy as np

from robot.kinematics import DLSIKSolver


class GraspPlanner:
    WAYPOINTS = ("pre", "grasp", "lift", "place_pre", "place")

    def __init__(self, robot_model, robot_data, ik_solver=None):
        self.ik_solver = ik_solver or DLSIKSolver(
            robot_model, robot_data, "attachment_site", lambda_damping=0.1
        )
        self.grasp_offset = 0.02
        self.approach_height = 0.10
        # 与 myscene.xml 中的标记位置一致，并处于已验证工作空间内。
        self.place_xy = np.array([0.3, -0.15])

    def compute_grasp_pose(self, object_position, place_xy=None):
        x, y, z = np.asarray(object_position, dtype=float)
        target_place_xy = (
            self.place_xy
            if place_xy is None
            else np.asarray(place_xy, dtype=float)
        )
        if target_place_xy.shape != (2,) or not np.all(
            np.isfinite(target_place_xy)
        ):
            raise ValueError("place_xy must contain two finite values")
        return {
            "pre": np.array([x, y, z + self.approach_height]),
            "grasp": np.array([x, y, z + self.grasp_offset]),
            "lift": np.array([x, y, z + self.approach_height]),
            "place_pre": np.array(
                [
                    target_place_xy[0],
                    target_place_xy[1],
                    z + self.approach_height,
                ]
            ),
            # 放置高度接近 FR3 工作空间边界；此处若叠加抓取偏移量，
            # 锁定姿态后的目标位姿将不可达。
            "place": np.array([target_place_xy[0], target_place_xy[1], z]),
            "orn": np.array([0.0, 1.0, 0.0, 0.0]),
        }

    def check_ik_feasible(self, grasp_pose):
        for name in self.WAYPOINTS:
            if self.ik_solver.solve(
                grasp_pose[name], target_quat=grasp_pose["orn"]
            ) is None:
                return False
        return True

    def check_Ik_feasible(self, grasp_pose):
        """用于兼容项目早期版本的别名。"""
        return self.check_ik_feasible(grasp_pose)
