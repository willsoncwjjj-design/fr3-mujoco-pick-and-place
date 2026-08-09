import warehouse_batch_run
from warehouse_batch_run import build_batch_plan


class ControllerSpy:
    def __init__(self):
        self.calls = []

    def run_cycle(self, target_body_id, place_xy):
        self.calls.append((target_body_id, tuple(place_xy)))
        return {
            "success": True,
            "verification": {
                "target_visible": True,
                "moved_distance_m": 0.2,
                "place_error_m": 0.01,
            },
        }


class RuntimeStub:
    instances = []

    def __init__(self, scene_path, use_viewer, object_names):
        self.scene_path = scene_path
        self.use_viewer = use_viewer
        self.object_names = tuple(object_names)
        self.object_catalog = [
            {"class_name": "cube_red", "body_id": 11},
            {"class_name": "cube_blue", "body_id": 12},
        ]
        self.controller = ControllerSpy()
        self.viewer = None
        self.closed = False
        self.__class__.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.closed = True


class SnapshotStub:
    def __init__(self, available, completed=(), missing=()):
        self.available_objects = tuple(available)
        self.completed_objects = tuple(completed)
        self.missing_objects = tuple(missing)

    def to_planner_state(self, scenario):
        state = scenario.scene_state(list(self.available_objects))
        state["completed_objects"] = list(self.completed_objects)
        state["missing_objects"] = list(self.missing_objects)
        return state


class ObserverStub:
    def __init__(self, snapshots):
        self.snapshots = list(snapshots)

    def observe(self):
        return self.snapshots.pop(0)


def test_build_batch_plan_orders_red_then_blue():
    scenario, plan = build_batch_plan()

    assert scenario.object_ids == ("cube_red", "cube_blue")
    assert [step.object_id for step in plan.steps if step.skill == "pick"] == [
        "cube_red",
        "cube_blue",
    ]


def test_run_batch_reuses_one_runtime_for_both_cycles(monkeypatch):
    RuntimeStub.instances.clear()
    monkeypatch.setattr(warehouse_batch_run, "RobotRuntime", RuntimeStub)
    observer = ObserverStub(
        [
            SnapshotStub(("cube_red", "cube_blue")),
            SnapshotStub((), completed=("cube_red", "cube_blue")),
        ]
    )

    result = warehouse_batch_run.run_batch(
        observer_factory=lambda scenario, runtime: observer
    )

    assert len(RuntimeStub.instances) == 1
    runtime = RuntimeStub.instances[0]
    assert runtime.scene_path == "robot/franka_fr3/warehouse_scene.xml"
    assert runtime.object_names == ("cube_red", "cube_blue")
    assert runtime.controller.calls == [
        (11, (0.30, -0.18)),
        (12, (0.46, -0.18)),
    ]
    assert runtime.closed
    assert result.success
    assert [item.object_id for item in result.items] == [
        "cube_red",
        "cube_blue",
    ]
    assert result.remaining_objects == ()


def test_observed_batch_skips_object_already_at_destination(monkeypatch):
    RuntimeStub.instances.clear()
    monkeypatch.setattr(warehouse_batch_run, "RobotRuntime", RuntimeStub)
    initial = SnapshotStub(("cube_blue",), completed=("cube_red",))
    final = SnapshotStub((), completed=("cube_red", "cube_blue"))
    observer = ObserverStub([initial, final])

    report = warehouse_batch_run.run_observed_batch(
        observer_factory=lambda scenario, runtime: observer
    )

    runtime = RuntimeStub.instances[0]
    assert runtime.controller.calls == [(12, (0.46, -0.18))]
    assert report.initial_state is initial
    assert report.final_state is final
    assert [step.object_id for step in report.plan.steps if step.skill == "pick"] == [
        "cube_blue"
    ]
    assert report.task_result.success
    assert report.goal_satisfied
