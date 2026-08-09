from dataclasses import dataclass
from typing import Optional

from vla_runtime.contracts import ActionChunk, DeltaEEAction


@dataclass(frozen=True)
class ActionDispatch:
    """从动作块中取出的一个待执行动作。"""

    chunk_id: str
    source_observation_id: int
    action_index: int
    action: DeltaEEAction


@dataclass(frozen=True)
class ManagerEvent:
    """用于回放动作块生命周期的结构化事件。"""

    event_type: str
    chunk_id: Optional[str]
    observation_id: int
    action_index: Optional[int] = None
    reason: Optional[str] = None
    remaining_actions: int = 0


@dataclass(frozen=True)
class ChunkSubmission:
    """动作块提交后的接受或拒绝结果。"""

    accepted: bool
    disposition: str
    chunk_id: str
    reason: Optional[str] = None


class ActionManager:
    """在安全动作边界管理动作块、过期检测和抢占。"""

    def __init__(self):
        self.latest_observation_id = 0
        self.active_chunk: Optional[ActionChunk] = None
        self.pending_chunk: Optional[ActionChunk] = None
        self.cursor = 0
        self.inflight: Optional[ActionDispatch] = None
        self.cancel_after_inflight: Optional[str] = None
        self._events = []

    @property
    def events(self):
        return tuple(self._events)

    @property
    def remaining_action_count(self):
        if self.active_chunk is None:
            return 0
        return max(0, len(self.active_chunk.actions) - self.cursor)

    def update_observation(self, observation_id):
        observation_id = int(observation_id)
        if observation_id < self.latest_observation_id:
            raise ValueError("observation_id must be monotonic")
        if observation_id == self.latest_observation_id:
            return
        self.latest_observation_id = observation_id
        reason = f"new_observation:{observation_id}"
        if self.pending_chunk is not None and (
            self.pending_chunk.source_observation_id < observation_id
        ):
            self._record_rejection(self.pending_chunk, "stale_pending_chunk")
            self.pending_chunk = None
        if self.active_chunk is None:
            return
        if self.active_chunk.source_observation_id >= observation_id:
            return
        if self.inflight is not None:
            self.cancel_after_inflight = reason
        else:
            self._cancel_active(reason)

    def submit(self, chunk: ActionChunk) -> ChunkSubmission:
        if chunk.source_observation_id != self.latest_observation_id:
            reason = (
                "stale_observation"
                if chunk.source_observation_id < self.latest_observation_id
                else "future_observation"
            )
            self._record_rejection(chunk, reason)
            return ChunkSubmission(False, "rejected", chunk.chunk_id, reason)

        newest_id = max(
            (
                item.source_observation_id
                for item in (self.active_chunk, self.pending_chunk)
                if item is not None
            ),
            default=0,
        )
        if chunk.source_observation_id <= newest_id:
            reason = "not_newer_than_current_chunk"
            self._record_rejection(chunk, reason)
            return ChunkSubmission(False, "rejected", chunk.chunk_id, reason)

        if self.active_chunk is None and self.inflight is None:
            self._activate(chunk)
            return ChunkSubmission(True, "activated", chunk.chunk_id)

        if self.inflight is not None:
            if self.pending_chunk is not None:
                self._record_rejection(
                    self.pending_chunk,
                    "replaced_by_newer_pending_chunk",
                )
            self.pending_chunk = chunk
            self._events.append(
                ManagerEvent(
                    "chunk_pending",
                    chunk.chunk_id,
                    chunk.source_observation_id,
                )
            )
            return ChunkSubmission(True, "pending", chunk.chunk_id)

        self._cancel_active(f"replaced_by:{chunk.chunk_id}")
        self._activate(chunk)
        return ChunkSubmission(True, "activated", chunk.chunk_id)

    def dispatch_next(self):
        if self.inflight is not None:
            raise RuntimeError("An action is already in flight")
        self._activate_pending_if_ready()
        while self.active_chunk is not None:
            if (
                self.active_chunk.source_observation_id
                < self.latest_observation_id
            ):
                self._cancel_active("stale_active_chunk")
                self._activate_pending_if_ready()
                continue
            if self.cursor >= len(self.active_chunk.actions):
                self._complete_active_chunk()
                self._activate_pending_if_ready()
                continue
            dispatch = ActionDispatch(
                chunk_id=self.active_chunk.chunk_id,
                source_observation_id=(
                    self.active_chunk.source_observation_id
                ),
                action_index=self.cursor,
                action=self.active_chunk.actions[self.cursor],
            )
            self.inflight = dispatch
            self._events.append(
                ManagerEvent(
                    "action_dispatched",
                    dispatch.chunk_id,
                    dispatch.source_observation_id,
                    action_index=dispatch.action_index,
                    remaining_actions=self.remaining_action_count,
                )
            )
            return dispatch
        return None

    def complete_action(self, dispatch, success=True, reason=None):
        if self.inflight != dispatch:
            raise ValueError("Completed action does not match the in-flight action")
        if self.active_chunk is None or self.active_chunk.chunk_id != dispatch.chunk_id:
            raise RuntimeError("In-flight action has no active chunk")
        event_type = "action_completed" if success else "action_failed"
        self._events.append(
            ManagerEvent(
                event_type,
                dispatch.chunk_id,
                dispatch.source_observation_id,
                action_index=dispatch.action_index,
                reason=reason,
                remaining_actions=max(0, self.remaining_action_count - 1),
            )
        )
        self.inflight = None
        if success:
            self.cursor += 1
            if self.cursor >= len(self.active_chunk.actions):
                self._complete_active_chunk()
        else:
            self._cancel_active(reason or "action_failed")

        if self.cancel_after_inflight is not None:
            cancel_reason = self.cancel_after_inflight
            self.cancel_after_inflight = None
            if self.active_chunk is not None:
                self._cancel_active(cancel_reason)

        if self.pending_chunk is not None:
            if self.active_chunk is not None:
                self._cancel_active(
                    f"replaced_by:{self.pending_chunk.chunk_id}"
                )
            self._activate_pending_if_ready()

    def cancel(self, reason="manual_cancel"):
        if self.inflight is not None:
            self.cancel_after_inflight = reason
            return "deferred"
        self._cancel_active(reason)
        if self.pending_chunk is not None:
            self._record_rejection(self.pending_chunk, reason)
            self.pending_chunk = None
        return "cancelled"

    def _activate(self, chunk):
        self.active_chunk = chunk
        self.cursor = 0
        self._events.append(
            ManagerEvent(
                "chunk_activated",
                chunk.chunk_id,
                chunk.source_observation_id,
                remaining_actions=len(chunk.actions),
            )
        )

    def _activate_pending_if_ready(self):
        if self.inflight is not None or self.pending_chunk is None:
            return
        if self.active_chunk is not None:
            return
        pending = self.pending_chunk
        self.pending_chunk = None
        if pending.source_observation_id != self.latest_observation_id:
            self._record_rejection(pending, "stale_pending_chunk")
            return
        self._activate(pending)

    def _cancel_active(self, reason):
        if self.active_chunk is None:
            return
        self._events.append(
            ManagerEvent(
                "chunk_cancelled",
                self.active_chunk.chunk_id,
                self.active_chunk.source_observation_id,
                reason=reason,
                remaining_actions=self.remaining_action_count,
            )
        )
        self.active_chunk = None
        self.cursor = 0

    def _complete_active_chunk(self):
        if self.active_chunk is None:
            return
        self._events.append(
            ManagerEvent(
                "chunk_completed",
                self.active_chunk.chunk_id,
                self.active_chunk.source_observation_id,
            )
        )
        self.active_chunk = None
        self.cursor = 0

    def _record_rejection(self, chunk, reason):
        self._events.append(
            ManagerEvent(
                "chunk_rejected",
                chunk.chunk_id,
                chunk.source_observation_id,
                reason=reason,
                remaining_actions=len(chunk.actions),
            )
        )
