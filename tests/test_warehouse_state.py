import json
from pathlib import Path

import numpy as np
import pytest

from warehouse.planners import RuleBasedPlanner
from warehouse.scenario import WarehouseScenario
from warehouse.state import WarehouseStateObserver

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_PATH = (
    PROJECT_ROOT / "warehouse" / "configs" / "warehouse_sorting_minimal.json"
)
OBJECT_CATALOG = [
    {"class_name": "cube_red", "body_id": 11},
    {"class_name": "cube_blue", "body_id": 12},
]


class CameraStub:
    def capture(self):
        return None, "depth", "segmentation", "camera_pose"


class DetectorStub:
    def detect_sim(self, segmentation, object_catalog):
        return [{"body_id": item["body_id"]} for item in object_catalog]


class LocalizerStub:
    def __init__(self, positions):
        self.positions = positions

    def localize(self, detections, depth, camera_pose):
        return [
            {
                "body_id": body_id,
                "position": np.asarray(position, dtype=float),
            }
            for body_id, position in self.positions.items()
        ]


@pytest.fixture()
def scenario():
    return WarehouseScenario.load(SCENARIO_PATH)


def observe(scenario, positions, sim_time=1.25):
    return WarehouseStateObserver(
        scenario=scenario,
        camera=CameraStub(),
        detector=DetectorStub(),
        localizer=LocalizerStub(positions),
        object_catalog=OBJECT_CATALOG,
        time_source=lambda: sim_time,
    ).observe()


def test_initial_snapshot_marks_visible_objects_and_empty_destinations(scenario):
    snapshot = observe(
        scenario,
        {
            11: (0.30, 0.15, 0.88),
            12: (0.45, 0.12, 0.88),
        },
    )

    assert snapshot.sequence_id == 1
    assert snapshot.sim_time_s == 1.25
    assert snapshot.visible_objects == ("cube_red", "cube_blue")
    assert snapshot.available_objects == ("cube_red", "cube_blue")
    assert snapshot.completed_objects == ()
    assert snapshot.missing_objects == ()
    assert snapshot.destination_for("priority_bin.slot_1").available
    assert snapshot.destination_for("standard_bin.slot_1").occupied_by == ()


def test_snapshot_detects_completed_object_and_drives_remaining_plan(scenario):
    snapshot = observe(
        scenario,
        {
            11: (0.305, -0.175, 0.85),
            12: (0.45, 0.12, 0.88),
        },
    )

    red = snapshot.object_for("cube_red")
    priority_bin = snapshot.destination_for("priority_bin.slot_1")
    assert red.at_destination
    assert red.destination_error_m == pytest.approx(0.007071, abs=1e-6)
    assert snapshot.completed_objects == ("cube_red",)
    assert snapshot.available_objects == ("cube_blue",)
    assert priority_bin.occupied_by == ("cube_red",)
    assert not priority_bin.available

    planner_state = snapshot.to_planner_state(scenario)
    plan = RuleBasedPlanner(scenario).plan(scenario.goal, planner_state)
    assert [step.object_id for step in plan.steps if step.skill == "pick"] == [
        "cube_blue"
    ]
    assert json.loads(json.dumps(planner_state))["completed_objects"] == [
        "cube_red"
    ]


def test_snapshot_does_not_infer_state_for_missing_object(scenario):
    snapshot = observe(scenario, {11: (0.30, 0.15, 0.88)})

    blue = snapshot.object_for("cube_blue")
    assert not blue.visible
    assert blue.position is None
    assert not blue.at_destination
    assert blue.destination_error_m is None
    assert snapshot.missing_objects == ("cube_blue",)
    assert snapshot.available_objects == ("cube_red",)


def test_observer_increments_snapshot_sequence(scenario):
    observer = WarehouseStateObserver(
        scenario=scenario,
        camera=CameraStub(),
        detector=DetectorStub(),
        localizer=LocalizerStub({11: (0.30, 0.15, 0.88)}),
        object_catalog=OBJECT_CATALOG,
    )

    assert observer.observe().sequence_id == 1
    assert observer.observe().sequence_id == 2
