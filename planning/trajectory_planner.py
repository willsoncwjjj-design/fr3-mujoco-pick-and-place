import numpy as np


class TrajectoryPlanner:
    WAYPOINTS = ("pre", "grasp", "lift", "place_pre", "place")

    def __init__(self, ik_solver):
        self.ik_solver = ik_solver

    def plan(self, grasp_pose, steps_per_segment=50):
        if steps_per_segment < 2:
            raise ValueError("steps_per_segment must be at least 2")

        trajectory = []
        commands = []
        current_joints = self.ik_solver.robot_qpos

        for name in self.WAYPOINTS:
            target_joints = self.ik_solver.solve(
                target_pos=grasp_pose[name],
                target_quat=grasp_pose["orn"],
                initial_pos=current_joints,
            )
            if target_joints is None:
                return None, None

            for alpha in np.linspace(0.0, 1.0, steps_per_segment + 1)[1:]:
                trajectory.append(
                    current_joints + alpha * (target_joints - current_joints)
                )
                commands.append(None)
            current_joints = target_joints.copy()

        commands[2 * steps_per_segment - 1] = "close"
        commands[5 * steps_per_segment - 1] = "open"
        return trajectory, commands
