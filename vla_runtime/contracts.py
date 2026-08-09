from dataclasses import dataclass
from math import isfinite
from typing import Optional

import numpy as np


def _finite_tuple(values, length, field_name):
    try:
        normalized = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{field_name} must contain {length} finite values"
        ) from error
    if len(normalized) != length or not all(isfinite(v) for v in normalized):
        raise ValueError(f"{field_name} must contain {length} finite values")
    return normalized


@dataclass(frozen=True)
class RGBFrame:
    """一帧带相机名称的只读 RGB 图像。"""

    camera_name: str
    image: np.ndarray

    def __post_init__(self):
        if not self.camera_name.strip():
            raise ValueError("camera_name must not be empty")
        image = np.asarray(self.image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("RGB image must have shape (height, width, 3)")
        if image.dtype != np.uint8:
            raise ValueError("RGB image must use uint8 values")
        copied = np.ascontiguousarray(image).copy()
        copied.setflags(write=False)
        object.__setattr__(self, "image", copied)


@dataclass(frozen=True)
class VLAObservation:
    """由视觉、机器人本体状态和语言组成的一次策略观测。"""

    observation_id: int
    timestamp_s: float
    frames: tuple[RGBFrame, ...]
    qpos: tuple[float, ...]
    qvel: tuple[float, ...]
    ee_position: tuple[float, ...]
    ee_quaternion: tuple[float, ...]
    gripper_position: float
    instruction: str
    subgoal: str

    def __post_init__(self):
        if self.observation_id < 1:
            raise ValueError("observation_id must be positive")
        if not isfinite(float(self.timestamp_s)):
            raise ValueError("timestamp_s must be finite")
        if not self.frames:
            raise ValueError("At least one RGB frame is required")
        names = [frame.camera_name for frame in self.frames]
        if len(names) != len(set(names)):
            raise ValueError("RGB frame camera names must be unique")
        object.__setattr__(self, "qpos", _finite_tuple(self.qpos, 7, "qpos"))
        object.__setattr__(self, "qvel", _finite_tuple(self.qvel, 7, "qvel"))
        object.__setattr__(
            self,
            "ee_position",
            _finite_tuple(self.ee_position, 3, "ee_position"),
        )
        quaternion = _finite_tuple(self.ee_quaternion, 4, "ee_quaternion")
        norm = float(np.linalg.norm(quaternion))
        if norm < 1e-12:
            raise ValueError("ee_quaternion must be non-zero")
        object.__setattr__(
            self,
            "ee_quaternion",
            tuple(value / norm for value in quaternion),
        )
        if not isfinite(float(self.gripper_position)):
            raise ValueError("gripper_position must be finite")
        if not self.instruction.strip():
            raise ValueError("instruction must not be empty")
        if not self.subgoal.strip():
            raise ValueError("subgoal must not be empty")


@dataclass(frozen=True)
class DeltaEEAction:
    """世界坐标系中的末端增量动作与可选夹爪命令。"""

    delta_position: tuple[float, ...]
    delta_rotation: tuple[float, ...]
    gripper_command: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(
            self,
            "delta_position",
            _finite_tuple(self.delta_position, 3, "delta_position"),
        )
        object.__setattr__(
            self,
            "delta_rotation",
            _finite_tuple(self.delta_rotation, 3, "delta_rotation"),
        )
        if self.gripper_command not in {None, "open", "close"}:
            raise ValueError("gripper_command must be open, close, or None")


@dataclass(frozen=True)
class ActionChunk:
    """由一次观测产生、按固定周期消费的一组未来动作。"""

    chunk_id: str
    source_observation_id: int
    actions: tuple[DeltaEEAction, ...]
    action_dt_s: float
    created_at_s: float
    policy_version: str

    def __post_init__(self):
        if not self.chunk_id.strip():
            raise ValueError("chunk_id must not be empty")
        if self.source_observation_id < 1:
            raise ValueError("source_observation_id must be positive")
        if not self.actions:
            raise ValueError("ActionChunk must contain at least one action")
        if not isfinite(float(self.action_dt_s)) or self.action_dt_s <= 0:
            raise ValueError("action_dt_s must be positive and finite")
        if not isfinite(float(self.created_at_s)):
            raise ValueError("created_at_s must be finite")
        if not self.policy_version.strip():
            raise ValueError("policy_version must not be empty")


@dataclass(frozen=True)
class FR3Command:
    """经过动作适配和安全检查后的 FR3 单步关节命令。"""

    joint_target: tuple[float, ...]
    gripper_command: Optional[str]
    target_ee_position: tuple[float, ...]
    target_ee_quaternion: tuple[float, ...]
    clipped: bool = False

    def __post_init__(self):
        object.__setattr__(
            self,
            "joint_target",
            _finite_tuple(self.joint_target, 7, "joint_target"),
        )
        object.__setattr__(
            self,
            "target_ee_position",
            _finite_tuple(
                self.target_ee_position,
                3,
                "target_ee_position",
            ),
        )
        object.__setattr__(
            self,
            "target_ee_quaternion",
            _finite_tuple(
                self.target_ee_quaternion,
                4,
                "target_ee_quaternion",
            ),
        )
        if self.gripper_command not in {None, "open", "close"}:
            raise ValueError("gripper_command must be open, close, or None")
