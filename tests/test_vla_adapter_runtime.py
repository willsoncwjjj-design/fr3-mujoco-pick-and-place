from types import SimpleNamespace

import numpy as np
import pytest

from vla_runtime.adapter import ActionAdapterError, DeltaEEActionAdapter
from vla_runtime.contracts import ActionChunk, DeltaEEAction, FR3Command
from vla_runtime.manager import ActionManager
from vla_runtime.runtime import ActionChunkRuntime


class RobotStub:
    def __init__(self):
        self.joint_limits = [(-1.0, 1.0)] * 7

    def get_end_effector_pose(self):
        return np.array([0.4, 0.0, 0.5]), np.array([1.0, 0.0, 0.0, 0.0])


class IKSolverStub:
    def __init__(self, solution=None):
        self.robot_qpos = np.zeros(7)
        self.solution = np.full(7, 0.1) if solution is None else solution
        self.received_position = None

    def solve(self, target_position, target_quat=None, initial_pos=None):
        self.received_position = np.asarray(target_position).copy()
        return self.solution


class ControllerSpy:
    def __init__(self):
        self.calls = []

    def execute_joint_target(
        self,
        joint_target,
        gripper_command=None,
        control_steps=10,
    ):
        self.calls.append((tuple(joint_target), gripper_command, control_steps))


class AdapterStub:
    def adapt(self, action):
        return FR3Command(
            joint_target=(action.delta_position[0],) * 7,
            gripper_command=action.gripper_command,
            target_ee_position=(0.4, 0.0, 0.5),
            target_ee_quaternion=(1.0, 0.0, 0.0, 0.0),
        )


def make_chunk(actions):
    return ActionChunk(
        chunk_id="chunk-1",
        source_observation_id=1,
        actions=tuple(actions),
        action_dt_s=0.02,
        created_at_s=1.0,
        policy_version="scripted-v1",
    )


def test_adapter_limits_translation_before_ik():
    solver = IKSolverStub()
    adapter = DeltaEEActionAdapter(RobotStub(), solver, max_translation_m=0.01)

    command = adapter.adapt(
        DeltaEEAction((0.03, 0.0, 0.0), (0.0, 0.0, 0.0))
    )

    assert command.clipped
    assert solver.received_position == pytest.approx((0.41, 0.0, 0.5))


def test_adapter_rejects_workspace_and_ik_failures():
    workspace_adapter = DeltaEEActionAdapter(
        RobotStub(),
        IKSolverStub(),
        workspace_low=(0.3, -0.2, 0.3),
        workspace_high=(0.405, 0.2, 0.8),
    )
    with pytest.raises(ActionAdapterError, match="out of bounds"):
        workspace_adapter.adapt(
            DeltaEEAction((0.01, 0.0, 0.0), (0.0, 0.0, 0.0))
        )

    ik_solver = IKSolverStub()
    ik_solver.solution = None
    ik_adapter = DeltaEEActionAdapter(RobotStub(), ik_solver)
    with pytest.raises(ActionAdapterError, match="IK feasible"):
        ik_adapter.adapt(
            DeltaEEAction((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        )


def test_runtime_executes_chunk_in_order():
    manager = ActionManager()
    manager.update_observation(1)
    actions = (
        DeltaEEAction((0.1, 0.0, 0.0), (0.0, 0.0, 0.0)),
        DeltaEEAction((0.2, 0.0, 0.0), (0.0, 0.0, 0.0), "close"),
    )
    manager.submit(make_chunk(actions))
    controller = ControllerSpy()
    runtime = ActionChunkRuntime(
        manager,
        AdapterStub(),
        controller,
        control_steps_per_action=4,
    )

    results = runtime.run_until_idle()

    assert [result.dispatch.action_index for result in results] == [0, 1]
    assert controller.calls == [
        ((0.1,) * 7, None, 4),
        ((0.2,) * 7, "close", 4),
    ]


def test_runtime_marks_adapter_failure_and_stops_chunk():
    manager = ActionManager()
    manager.update_observation(1)
    manager.submit(
        make_chunk(
            (DeltaEEAction((0.1, 0.0, 0.0), (0.0, 0.0, 0.0)),)
        )
    )
    adapter = SimpleNamespace(
        adapt=lambda action: (_ for _ in ()).throw(ActionAdapterError("bad action"))
    )
    runtime = ActionChunkRuntime(manager, adapter, ControllerSpy())

    result = runtime.step()

    assert not result.success
    assert result.error_message == "bad action"
    assert manager.active_chunk is None
    assert [event.event_type for event in manager.events[-2:]] == [
        "action_failed",
        "chunk_cancelled",
    ]
