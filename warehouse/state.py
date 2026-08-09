from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from warehouse.scenario import WarehouseScenario


@dataclass(frozen=True)
class ObjectState:
    object_id: str
    visible: bool
    position: Optional[tuple[float, float, float]]
    assigned_destination_id: str
    at_destination: bool
    destination_error_m: Optional[float]


@dataclass(frozen=True)
class DestinationState:
    destination_id: str
    place_xy: tuple[float, float]
    occupied_by: tuple[str, ...]
    available: bool
    acceptance_radius_m: float


@dataclass(frozen=True)
class WarehouseStateSnapshot:
    sequence_id: int
    sim_time_s: float
    objects: tuple[ObjectState, ...]
    destinations: tuple[DestinationState, ...]

    @property
    def visible_objects(self):
        return tuple(item.object_id for item in self.objects if item.visible)

    @property
    def available_objects(self):
        return tuple(
            item.object_id
            for item in self.objects
            if item.visible and not item.at_destination
        )

    @property
    def completed_objects(self):
        return tuple(
            item.object_id for item in self.objects if item.at_destination
        )

    @property
    def missing_objects(self):
        return tuple(item.object_id for item in self.objects if not item.visible)

    def object_for(self, object_id):
        try:
            return next(item for item in self.objects if item.object_id == object_id)
        except StopIteration as error:
            raise KeyError(f"Unknown observed object: {object_id}") from error

    def destination_for(self, destination_id):
        try:
            return next(
                item
                for item in self.destinations
                if item.destination_id == destination_id
            )
        except StopIteration as error:
            raise KeyError(
                f"Unknown observed destination: {destination_id}"
            ) from error

    def to_planner_state(self, scenario):
        state = scenario.scene_state(list(self.available_objects))
        state.update(
            {
                "snapshot_id": self.sequence_id,
                "sim_time_s": self.sim_time_s,
                "visible_objects": list(self.visible_objects),
                "completed_objects": list(self.completed_objects),
                "missing_objects": list(self.missing_objects),
                "objects": {
                    item.object_id: asdict(item) for item in self.objects
                },
            }
        )
        destinations = deepcopy(state["destinations"])
        for observed in self.destinations:
            destinations[observed.destination_id].update(
                {
                    "occupied_by": list(observed.occupied_by),
                    "available": observed.available,
                    "acceptance_radius_m": observed.acceptance_radius_m,
                }
            )
        state["destinations"] = destinations
        return state


class WarehouseStateObserver:
    def __init__(
        self,
        scenario,
        camera,
        detector,
        localizer,
        object_catalog,
        time_source=None,
        acceptance_radius_m=0.05,
    ):
        if acceptance_radius_m <= 0:
            raise ValueError("acceptance_radius_m must be positive")
        self.scenario: WarehouseScenario = scenario
        self.camera = camera
        self.detector = detector
        self.localizer = localizer
        self.time_source = time_source or (lambda: 0.0)
        self.acceptance_radius_m = float(acceptance_radius_m)
        self.body_ids = self._build_body_ids(object_catalog)
        missing = set(scenario.object_ids) - set(self.body_ids)
        if missing:
            raise ValueError(f"Missing observed objects: {sorted(missing)}")
        self.sequence_id = 0

    @classmethod
    def from_runtime(cls, scenario, runtime, acceptance_radius_m=0.05):
        return cls(
            scenario=scenario,
            camera=runtime.camera,
            detector=runtime.detector,
            localizer=runtime.localizer,
            object_catalog=runtime.object_catalog,
            time_source=lambda: runtime.data.time,
            acceptance_radius_m=acceptance_radius_m,
        )

    def observe(self):
        _, depth, segmentation, camera_pose = self.camera.capture()
        detections = self.detector.detect_sim(
            segmentation,
            self._object_catalog(),
        )
        localized = self.localizer.localize(detections, depth, camera_pose)
        positions = {
            int(item["body_id"]): self._normalize_position(item["position"])
            for item in localized
        }
        objects = self._build_object_states(positions)
        destinations = self._build_destination_states(objects)
        self.sequence_id += 1
        return WarehouseStateSnapshot(
            sequence_id=self.sequence_id,
            sim_time_s=float(self.time_source()),
            objects=objects,
            destinations=destinations,
        )

    def _build_object_states(self, positions):
        states = []
        for item in self.scenario.inventory:
            position = positions.get(self.body_ids[item.object_id])
            destination_xy = self._destination_xy(item.destination_id)
            error = None
            at_destination = False
            if position is not None:
                error = float(np.linalg.norm(position[:2] - destination_xy))
                at_destination = error <= self.acceptance_radius_m
            states.append(
                ObjectState(
                    object_id=item.object_id,
                    visible=position is not None,
                    position=None if position is None else tuple(position),
                    assigned_destination_id=item.destination_id,
                    at_destination=at_destination,
                    destination_error_m=error,
                )
            )
        return tuple(states)

    def _build_destination_states(self, objects):
        states = []
        for destination_id in self.scenario.destination_ids:
            destination_xy = self._destination_xy(destination_id)
            occupied_by = tuple(
                item.object_id
                for item in objects
                if item.position is not None
                and np.linalg.norm(
                    np.asarray(item.position[:2]) - destination_xy
                )
                <= self.acceptance_radius_m
            )
            states.append(
                DestinationState(
                    destination_id=destination_id,
                    place_xy=tuple(destination_xy),
                    occupied_by=occupied_by,
                    available=not occupied_by,
                    acceptance_radius_m=self.acceptance_radius_m,
                )
            )
        return tuple(states)

    def _destination_xy(self, destination_id):
        values = np.asarray(
            self.scenario.destinations[destination_id]["place_xy"],
            dtype=float,
        )
        if values.shape != (2,) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"Invalid destination coordinates: {destination_id}"
            )
        return values

    def _object_catalog(self):
        return [
            {"class_name": object_id, "body_id": body_id}
            for object_id, body_id in self.body_ids.items()
        ]

    @staticmethod
    def _normalize_position(position):
        values = np.asarray(position, dtype=float)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            raise ValueError("Localized object position must contain three values")
        return values

    @staticmethod
    def _build_body_ids(object_catalog):
        body_ids = {}
        used_body_ids = set()
        for item in object_catalog:
            object_id = str(item["class_name"])
            body_id = int(item["body_id"])
            if object_id in body_ids or body_id in used_body_ids:
                raise ValueError("Observed object catalog must be one-to-one")
            body_ids[object_id] = body_id
            used_body_ids.add(body_id)
        return body_ids
