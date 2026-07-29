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
        # Matches the marker in myscene.xml and stays within the verified workspace.
        self.place_xy = np.array([0.3, -0.15])

    def compute_grasp_pose(self, object_position):
        x, y, z = np.asarray(object_position, dtype=float)
        return {
            "pre": np.array([x, y, z + self.approach_height]),
            "grasp": np.array([x, y, z + self.grasp_offset]),
            "lift": np.array([x, y, z + self.approach_height]),
            "place_pre": np.array(
                [self.place_xy[0], self.place_xy[1], z + self.approach_height]
            ),
            # The placement height is near the FR3 workspace boundary; adding the
            # grasp offset here makes the locked-orientation pose unreachable.
            "place": np.array([self.place_xy[0], self.place_xy[1], z]),
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
        """Backward-compatible alias for earlier project versions."""
        return self.check_ik_feasible(grasp_pose)
