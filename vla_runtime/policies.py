import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, Protocol

from vla_runtime.contracts import ActionChunk, VLAObservation


class VlaPolicy(Protocol):
    """把一次 VLA 观测转换为动作块的策略接口。"""

    def infer(self, observation: VLAObservation) -> ActionChunk:
        ...


class ScriptedPolicy:
    """用于验证 Runtime 的确定性模拟策略，不代表真实 VLA。"""

    def __init__(
        self,
        actions_by_observation,
        action_dt_s=0.02,
        latency_s=0.0,
        policy_version="scripted-v1",
    ):
        if action_dt_s <= 0:
            raise ValueError("action_dt_s must be positive")
        if latency_s < 0:
            raise ValueError("latency_s must not be negative")
        self.actions_by_observation = {
            int(key): tuple(value)
            for key, value in actions_by_observation.items()
        }
        self.action_dt_s = float(action_dt_s)
        self.latency_s = float(latency_s)
        self.policy_version = str(policy_version)
        self.request_count = 0

    def infer(self, observation: VLAObservation) -> ActionChunk:
        try:
            actions = self.actions_by_observation[observation.observation_id]
        except KeyError as error:
            raise ValueError(
                "No scripted actions for observation "
                f"{observation.observation_id}"
            ) from error
        if self.latency_s:
            time.sleep(self.latency_s)
        self.request_count += 1
        return ActionChunk(
            chunk_id=(
                f"{self.policy_version}-obs{observation.observation_id}"
                f"-req{self.request_count}"
            ),
            source_observation_id=observation.observation_id,
            actions=tuple(actions),
            action_dt_s=self.action_dt_s,
            created_at_s=time.monotonic(),
            policy_version=self.policy_version,
        )


@dataclass(frozen=True)
class PolicyResult:
    """一次异步策略请求的完成结果。"""

    request_id: int
    observation_id: int
    latency_s: float
    chunk: Optional[ActionChunk] = None
    error_message: Optional[str] = None


class AsyncPolicyWorker:
    """在独立线程中推理，使控制循环无需等待策略返回。"""

    def __init__(self, policy: VlaPolicy):
        self.policy = policy
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="vla-policy",
        )
        self._pending = {}
        self._request_id = 0
        self._closed = False

    def request(self, observation: VLAObservation):
        if self._closed:
            raise RuntimeError("AsyncPolicyWorker is closed")
        self._request_id += 1
        request_id = self._request_id
        started_at = time.monotonic()
        future = self._executor.submit(self.policy.infer, observation)
        self._pending[request_id] = (
            observation.observation_id,
            started_at,
            future,
        )
        return request_id

    def poll(self):
        completed = []
        for request_id in sorted(tuple(self._pending)):
            observation_id, started_at, future = self._pending[request_id]
            if not future.done():
                continue
            del self._pending[request_id]
            latency_s = time.monotonic() - started_at
            try:
                chunk = future.result()
                completed.append(
                    PolicyResult(
                        request_id=request_id,
                        observation_id=observation_id,
                        latency_s=latency_s,
                        chunk=chunk,
                    )
                )
            except Exception as error:
                completed.append(
                    PolicyResult(
                        request_id=request_id,
                        observation_id=observation_id,
                        latency_s=latency_s,
                        error_message=str(error),
                    )
                )
        return tuple(completed)

    @property
    def pending_count(self):
        return len(self._pending)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
