from dataclasses import dataclass

INJECTABLE_FAILURE_STATES = {
    "object_missing": "LOCALIZE",
    "pick_failed": "EXECUTE",
    "ik_failed": "PLAN",
    "verification_failed": "VERIFY",
}


@dataclass(frozen=True)
class FaultInjection:
    object_id: str
    failure_code: str
    attempt_number: int = 1

    def __post_init__(self):
        if not self.object_id.strip():
            raise ValueError("Injected fault object_id must not be empty")
        if self.failure_code not in INJECTABLE_FAILURE_STATES:
            raise ValueError(
                f"Unsupported injected failure: {self.failure_code}"
            )
        if self.attempt_number < 1:
            raise ValueError("attempt_number must be positive")

    @property
    def failed_state(self):
        return INJECTABLE_FAILURE_STATES[self.failure_code]

    def to_controller_result(self):
        verification = None
        if self.failure_code == "verification_failed":
            verification = {
                "target_visible": True,
                "moved_distance_m": 0.0,
                "place_error_m": 0.20,
                "minimum_movement_m": 0.02,
                "maximum_place_error_m": 0.05,
            }
        return {
            "success": False,
            "error_message": (
                f"Injected {self.failure_code} for {self.object_id} "
                f"on attempt {self.attempt_number}"
            ),
            "failure_code": self.failure_code,
            "failed_state": self.failed_state,
            "verification": verification,
        }


class FaultInjectingController:
    def __init__(self, controller, object_catalog, injections):
        self.controller = controller
        self.object_ids_by_body_id = {
            int(item["body_id"]): str(item["class_name"])
            for item in object_catalog
        }
        known_objects = set(self.object_ids_by_body_id.values())
        schedule = {}
        for injection in injections:
            if injection.object_id not in known_objects:
                raise ValueError(
                    f"Unknown injected fault object: {injection.object_id}"
                )
            key = (injection.object_id, injection.attempt_number)
            if key in schedule:
                raise ValueError(f"Duplicate injected fault: {key}")
            schedule[key] = injection
        self.schedule = schedule
        self.attempts = {}
        self._events = []

    @property
    def events(self):
        return tuple(self._events)

    def run_cycle(self, target_body_id, place_xy):
        try:
            object_id = self.object_ids_by_body_id[int(target_body_id)]
        except KeyError as error:
            raise ValueError(
                f"Unknown controller target body id: {target_body_id}"
            ) from error

        attempt_number = self.attempts.get(object_id, 0) + 1
        self.attempts[object_id] = attempt_number
        injection = self.schedule.get((object_id, attempt_number))
        if injection is not None:
            self._events.append(injection)
            return injection.to_controller_result()
        return self.controller.run_cycle(
            target_body_id=target_body_id,
            place_xy=place_xy,
        )
