import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

DEFAULT_SCENARIO_PATH = (
    Path(__file__).resolve().parent / "configs" / "warehouse_sorting.json"
)


@dataclass(frozen=True)
class InventoryItem:
    object_id: str
    sku: str
    category: str
    destination_id: str


class WarehouseScenario:
    def __init__(self, payload: dict[str, Any]):
        self.task_id = str(payload["task_id"])
        self.goal = str(payload["goal"])
        self.inventory = tuple(
            InventoryItem(
                object_id=str(item["object_id"]),
                sku=str(item["sku"]),
                category=str(item["category"]),
                destination_id=str(item["destination_id"]),
            )
            for item in payload["inventory"]
        )
        self.destinations = payload["destinations"]
        self.failure_policy = {
            str(key): str(value)
            for key, value in payload["failure_policy"].items()
        }
        self._validate()

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "WarehouseScenario":
        scenario_path = path or DEFAULT_SCENARIO_PATH
        with scenario_path.open(encoding="utf-8") as handle:
            return cls(json.load(handle))

    @property
    def object_ids(self) -> tuple[str, ...]:
        return tuple(item.object_id for item in self.inventory)

    @property
    def destination_ids(self) -> tuple[str, ...]:
        return tuple(self.destinations)

    @property
    def buffer_destination_ids(self) -> tuple[str, ...]:
        return tuple(
            destination_id
            for destination_id, destination in self.destinations.items()
            if destination.get("purpose") == "buffer"
        )

    def item_for(self, object_id: str) -> InventoryItem:
        try:
            return next(item for item in self.inventory if item.object_id == object_id)
        except StopIteration as error:
            raise KeyError(f"Unknown warehouse object: {object_id}") from error

    def scene_state(self, available_objects: Optional[list[str]] = None):
        available = (
            list(self.object_ids)
            if available_objects is None
            else list(available_objects)
        )
        unknown = set(available) - set(self.object_ids)
        if unknown:
            raise ValueError(f"Unknown available objects: {sorted(unknown)}")
        return {
            "available_objects": available,
            "inventory": [item.__dict__ for item in self.inventory],
            "destinations": deepcopy(self.destinations),
        }

    def _validate(self):
        object_ids = self.object_ids
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("Warehouse object ids must be unique")
        missing = {
            item.destination_id
            for item in self.inventory
            if item.destination_id not in self.destinations
        }
        if missing:
            raise ValueError(f"Unknown inventory destinations: {sorted(missing)}")
