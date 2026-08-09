from dataclasses import asdict

import pytest

import warehouse_fault_demo
from warehouse.fault_injection import FaultInjectingController, FaultInjection

OBJECT_CATALOG = [
    {"class_name": "cube_red", "body_id": 11},
    {"class_name": "cube_blue", "body_id": 12},
]


class ControllerSpy:
    def __init__(self):
        self.calls = []

    def run_cycle(self, target_body_id, place_xy):
        self.calls.append((target_body_id, tuple(place_xy)))
        return {"success": True, "verification": {"target_visible": True}}


class RuntimeStub:
    instances = []

    def __init__(self, scene_path, use_viewer, object_names):
        self.scene_path = scene_path
        self.use_viewer = use_viewer
        self.object_names = tuple(object_names)
        self.object_catalog = OBJECT_CATALOG
        self.controller = ControllerSpy()
        self.viewer = None
        self.closed = False
        self.__class__.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.closed = True


def test_fault_injection_validates_configuration():
    with pytest.raises(ValueError, match="Unsupported injected failure"):
        FaultInjection("cube_red", "unknown_failure")
    with pytest.raises(ValueError, match="attempt_number must be positive"):
        FaultInjection("cube_red", "object_missing", attempt_number=0)


def test_wrapper_injects_only_configured_object_and_attempt():
    controller = ControllerSpy()
    injection = FaultInjection(
        "cube_red",
        "object_missing",
        attempt_number=2,
    )
    wrapper = FaultInjectingController(
        controller,
        OBJECT_CATALOG,
        [injection],
    )

    first = wrapper.run_cycle(11, (0.30, -0.18))
    blue = wrapper.run_cycle(12, (0.46, -0.18))
    second = wrapper.run_cycle(11, (0.30, -0.18))

    assert first["success"]
    assert blue["success"]
    assert not second["success"]
    assert second["failure_code"] == "object_missing"
    assert second["failed_state"] == "LOCALIZE"
    assert controller.calls == [
        (11, (0.30, -0.18)),
        (12, (0.46, -0.18)),
    ]
    assert wrapper.events == (injection,)


@pytest.mark.parametrize(
    ("failure_code", "expected_action", "expected_disposition", "task_success"),
    [
        ("object_missing", "rescan", "completed", True),
        ("pick_failed", "retry_once", "completed", True),
        ("ik_failed", "skip_and_report", "skipped", False),
        ("verification_failed", "stop", "stopped", False),
    ],
)
def test_fault_demo_verifies_each_policy(
    monkeypatch,
    failure_code,
    expected_action,
    expected_disposition,
    task_success,
):
    RuntimeStub.instances.clear()
    monkeypatch.setattr(warehouse_fault_demo, "RobotRuntime", RuntimeStub)

    result = warehouse_fault_demo.run_fault_demo(
        failure_code=failure_code,
        object_id="cube_red",
    )

    assert result.policy_verified
    assert result.expected_action == expected_action
    assert result.task_result.success is task_success
    item = result.task_result.items[0]
    assert item.disposition == expected_disposition
    assert item.policy_actions[0] == expected_action
    assert result.injected_events == (result.injection,)
    assert RuntimeStub.instances[0].closed
    assert asdict(result)["policy_verified"]
