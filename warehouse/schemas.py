from dataclasses import asdict, dataclass
from typing import Any, Optional

ALLOWED_SKILLS = ("scan", "relocate", "pick", "place", "verify")
ALLOWED_FAILURE_ACTIONS = ("rescan", "retry_once", "skip_and_report", "stop")
FORBIDDEN_CONTROL_FIELDS = {
    "actuator",
    "ctrl",
    "joint_angle",
    "joint_angles",
    "qpos",
    "torque",
    "trajectory",
    "velocity",
}


@dataclass(frozen=True)
class PlanStep:
    step_id: int
    skill: str
    object_id: Optional[str] = None
    destination_id: Optional[str] = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PlanStep":
        allowed = {"step_id", "skill", "object_id", "destination_id"}
        extras = set(value) - allowed
        if extras:
            raise ValueError(f"Unsupported plan step fields: {sorted(extras)}")
        try:
            return cls(
                step_id=int(value["step_id"]),
                skill=str(value["skill"]),
                object_id=value.get("object_id"),
                destination_id=value.get("destination_id"),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid plan step: {value}") from error


@dataclass(frozen=True)
class TaskPlan:
    task_id: str
    goal: str
    steps: tuple[PlanStep, ...]
    failure_policy: dict[str, str]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TaskPlan":
        allowed = {"task_id", "goal", "steps", "failure_policy"}
        extras = set(value) - allowed
        if extras:
            raise ValueError(f"Unsupported task plan fields: {sorted(extras)}")
        try:
            steps = tuple(PlanStep.from_dict(item) for item in value["steps"])
            failure_policy = {
                str(key): str(action)
                for key, action in value["failure_policy"].items()
            }
            return cls(
                task_id=str(value["task_id"]),
                goal=str(value["goal"]),
                steps=steps,
                failure_policy=failure_policy,
            )
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, ValueError):
                raise
            raise ValueError("Invalid task plan structure") from error

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


TASK_PLAN_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["task_id", "goal", "steps", "failure_policy"],
    "properties": {
        "task_id": {"type": "string"},
        "goal": {"type": "string"},
        "steps": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "step_id",
                    "skill",
                    "object_id",
                    "destination_id",
                ],
                "properties": {
                    "step_id": {"type": "integer", "minimum": 1},
                    "skill": {"type": "string", "enum": list(ALLOWED_SKILLS)},
                    "object_id": {"type": ["string", "null"]},
                    "destination_id": {"type": ["string", "null"]},
                },
            },
        },
        "failure_policy": {
            "type": "object",
            "additionalProperties": {
                "type": "string",
                "enum": list(ALLOWED_FAILURE_ACTIONS),
            },
        },
    },
}
