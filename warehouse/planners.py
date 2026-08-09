import json
import logging
from copy import deepcopy
from typing import Any, Callable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from warehouse.scenario import WarehouseScenario
from warehouse.schemas import TASK_PLAN_JSON_SCHEMA, PlanStep, TaskPlan
from warehouse.validation import PlanValidator

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:14b"
LOGGER = logging.getLogger(__name__)


class RuleBasedPlanner:
    def __init__(self, scenario: WarehouseScenario):
        self.scenario = scenario
        self.validator = PlanValidator(scenario)

    def plan(
        self,
        task_text: str,
        scene_state: dict[str, Any],
        execution_context: Optional[dict[str, Any]] = None,
    ) -> TaskPlan:
        available = set(scene_state["available_objects"])
        steps = [PlanStep(step_id=1, skill="scan")]
        next_step_id = 2
        for item in self.scenario.inventory:
            if item.object_id not in available:
                continue
            steps.extend(
                [
                    PlanStep(next_step_id, "pick", item.object_id),
                    PlanStep(
                        next_step_id + 1,
                        "place",
                        item.object_id,
                        item.destination_id,
                    ),
                    PlanStep(
                        next_step_id + 2,
                        "verify",
                        item.object_id,
                        item.destination_id,
                    ),
                ]
            )
            next_step_id += 3

        plan = TaskPlan(
            task_id=self.scenario.task_id,
            goal=task_text.strip() or self.scenario.goal,
            steps=tuple(steps),
            failure_policy=self.scenario.failure_policy.copy(),
        )
        self.validator.validate(plan)
        return plan


class ConstraintAwarePlanner:
    def __init__(self, scenario: WarehouseScenario):
        self.scenario = scenario
        self.validator = PlanValidator(scenario)

    def plan(
        self,
        task_text: str,
        scene_state: dict[str, Any],
        execution_context: Optional[dict[str, Any]] = None,
    ) -> TaskPlan:
        available = set(scene_state["available_objects"])
        destinations = scene_state["destinations"]
        steps = [PlanStep(step_id=1, skill="scan")]
        next_step_id = 2
        used_buffers = set()
        relocated = set()

        for item in self.scenario.inventory:
            if item.object_id not in available:
                continue
            destination = destinations[item.destination_id]
            blockers = [
                object_id
                for object_id in destination.get("occupied_by", [])
                if object_id != item.object_id
            ]
            for blocker in blockers:
                if blocker in relocated:
                    continue
                if blocker not in available:
                    raise ValueError(
                        f"Blocking object is not available: {blocker}"
                    )
                buffer_id = self._available_buffer(destinations, used_buffers)
                steps.append(
                    PlanStep(next_step_id, "relocate", blocker, buffer_id)
                )
                next_step_id += 1
                relocated.add(blocker)
                used_buffers.add(buffer_id)

            steps.extend(
                [
                    PlanStep(next_step_id, "pick", item.object_id),
                    PlanStep(
                        next_step_id + 1,
                        "place",
                        item.object_id,
                        item.destination_id,
                    ),
                    PlanStep(
                        next_step_id + 2,
                        "verify",
                        item.object_id,
                        item.destination_id,
                    ),
                ]
            )
            next_step_id += 3

        plan = TaskPlan(
            task_id=self.scenario.task_id,
            goal=task_text.strip() or self.scenario.goal,
            steps=tuple(steps),
            failure_policy=self.scenario.failure_policy.copy(),
        )
        self.validator.validate(plan)
        return plan

    def _available_buffer(self, destinations, used_buffers):
        for destination_id in self.scenario.buffer_destination_ids:
            destination = destinations[destination_id]
            if (
                destination_id not in used_buffers
                and destination.get("available", True)
                and not destination.get("occupied_by", [])
            ):
                return destination_id
        raise ValueError("No available buffer destination")


class OllamaPlanner:
    def __init__(
        self,
        scenario: WarehouseScenario,
        model: str = DEFAULT_OLLAMA_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout_seconds: int = 120,
        max_repair_attempts: int = 2,
        transport: Optional[Callable[..., dict[str, Any]]] = None,
    ):
        if max_repair_attempts < 0:
            raise ValueError("max_repair_attempts must not be negative")
        self.scenario = scenario
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_repair_attempts = int(max_repair_attempts)
        self.transport = transport or _post_json
        self.validator = PlanValidator(scenario)
        self.request_count = 0
        self.rejection_count = 0

    def plan(
        self,
        task_text: str,
        scene_state: dict[str, Any],
        execution_context: Optional[dict[str, Any]] = None,
    ) -> TaskPlan:
        response_schema = self._build_response_schema(scene_state)
        buffer_ids = self._available_buffer_ids(scene_state)
        allowed_skills = ["scan", "pick", "place", "verify"]
        if buffer_ids:
            allowed_skills.insert(1, "relocate")
        prompt = {
            "task": task_text,
            "scene_state": scene_state,
            "execution_context": execution_context or {},
            "required_task_id": self.scenario.task_id,
            "required_failure_policy": self.scenario.failure_policy,
            "constraints": {
                "allowed_skills": allowed_skills,
                "rule": (
                    "Return a complete plan for the remaining task, but expect "
                    "the executor to run only the first motion operation before "
                    "observing and replanning. Output exactly one scan step first. "
                    "For each selected object, output pick -> place -> verify. "
                    "Use only available objects. If a target is occupied, first "
                    "move its blocker directly to its own target when possible, "
                    "or relocate it to an available buffer. Every relocated object "
                    "must later be picked, placed at its configured target, and "
                    "verified."
                ),
                "control_boundary": (
                    "Never output joint angles, qpos, actuator commands, torque, "
                    "velocity, or trajectories."
                ),
            },
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a warehouse task planner. Return only a plan "
                    "matching the supplied JSON schema."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(prompt, ensure_ascii=False),
            },
        ]
        for attempt in range(self.max_repair_attempts + 1):
            payload = {
                "model": self.model,
                "stream": False,
                "format": response_schema,
                "options": {"temperature": 0},
                "messages": messages,
            }
            self.request_count += 1
            response = self.transport(
                f"{self.base_url}/api/chat",
                payload,
                self.timeout_seconds,
            )
            content = response.get("message", {}).get("content")
            try:
                if not isinstance(content, str):
                    raise RuntimeError(
                        "Ollama returned an invalid structured response"
                    )
                raw_plan = json.loads(content)
                LOGGER.debug("Ollama raw task plan: %s", raw_plan)
                plan = self.validator.parse_and_validate(raw_plan)
                self.validator.validate_against_state(plan, scene_state)
                return plan
            except (TypeError, json.JSONDecodeError, RuntimeError, ValueError) as error:
                self.rejection_count += 1
                if attempt >= self.max_repair_attempts:
                    raise RuntimeError(
                        "Ollama plan rejected after "
                        f"{attempt + 1} attempt(s): {error}"
                    ) from error
                LOGGER.info("Requesting Ollama plan repair: %s", error)
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": content or "{}",
                        },
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "validation_error": str(error),
                                    "instruction": (
                                        "Correct the plan against the same scene "
                                        "state. Return the complete corrected JSON "
                                        "plan only. Do not weaken or bypass the "
                                        "constraint."
                                    ),
                                },
                                ensure_ascii=False,
                            ),
                        },
                    ]
                )

    def _build_response_schema(self, scene_state):
        schema = deepcopy(TASK_PLAN_JSON_SCHEMA)
        schema["properties"]["task_id"]["const"] = self.scenario.task_id

        available_objects = list(scene_state["available_objects"])
        buffer_ids = self._available_buffer_ids(scene_state)
        schema["properties"]["steps"]["items"] = {
            "oneOf": self._build_step_variants(available_objects, buffer_ids)
        }

        failure_schema = schema["properties"]["failure_policy"]
        failure_schema["additionalProperties"] = False
        failure_schema["required"] = list(self.scenario.failure_policy)
        failure_schema["properties"] = {
            key: {"type": "string", "const": action}
            for key, action in self.scenario.failure_policy.items()
        }
        return schema

    def _build_step_variants(self, available_objects, buffer_ids):
        common = {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "step_id",
                "skill",
                "object_id",
                "destination_id",
            ],
        }
        variants = [
            {
                **common,
                "properties": {
                    "step_id": {"type": "integer", "minimum": 1},
                    "skill": {"type": "string", "const": "scan"},
                    "object_id": {"type": "null"},
                    "destination_id": {"type": "null"},
                },
            }
        ]
        for object_id in available_objects:
            destination_id = self.scenario.item_for(object_id).destination_id
            for skill in ("pick", "place", "verify"):
                destination_schema = (
                    {"type": "null"}
                    if skill == "pick"
                    else {"type": "string", "const": destination_id}
                )
                variants.append(
                    {
                        **common,
                        "properties": {
                            "step_id": {"type": "integer", "minimum": 1},
                            "skill": {"type": "string", "const": skill},
                            "object_id": {
                                "type": "string",
                                "const": object_id,
                            },
                            "destination_id": destination_schema,
                        },
                    }
                )
            for buffer_id in buffer_ids:
                variants.append(
                    {
                        **common,
                        "properties": {
                            "step_id": {"type": "integer", "minimum": 1},
                            "skill": {"type": "string", "const": "relocate"},
                            "object_id": {
                                "type": "string",
                                "const": object_id,
                            },
                            "destination_id": {
                                "type": "string",
                                "const": buffer_id,
                            },
                        },
                    }
                )
        return variants

    def _available_buffer_ids(self, scene_state):
        destinations = scene_state["destinations"]
        return [
            destination_id
            for destination_id in self.scenario.buffer_destination_ids
            if destinations[destination_id].get("available", True)
            and not destinations[destination_id].get("occupied_by", [])
        ]


def _post_json(url: str, payload: dict[str, Any], timeout: int):
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama request failed ({error.code}): {details}"
        ) from error
    except URLError as error:
        raise RuntimeError(
            "Cannot reach Ollama. Start it with `ollama serve` and retry."
        ) from error
