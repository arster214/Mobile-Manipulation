#!/usr/bin/env python3
"""Shared obstacle-scene schema for cuRobo, Isaac Sim, and MoveIt2."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "simple_obstacles.yml"
DEFAULT_FRAME = "base_link_underpan"
_VALID_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class CuboidObstacle:
    name: str
    dims: tuple[float, float, float]
    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    color_rgba: tuple[float, float, float, float]

    @property
    def pose_wxyz(self) -> list[float]:
        return [*self.position, *self.quaternion_wxyz]


@dataclass(frozen=True)
class ObstacleScene:
    frame_id: str
    cuboids: tuple[CuboidObstacle, ...]

    def as_curobo_dict(self) -> dict:
        """Return only keys accepted by cuRobo's SceneCfg parser."""
        return {
            "cuboid": {
                item.name: {
                    "dims": list(item.dims),
                    "pose": item.pose_wxyz,
                }
                for item in self.cuboids
            }
        }

    @property
    def fingerprint(self) -> str:
        payload = {"frame_id": self.frame_id, **self.as_curobo_dict()}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()[:12]


def _float_tuple(value, length: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{label} must contain exactly {length} values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} contains a non-finite value")
    return result


def load_obstacle_scene(path: Path | str = DEFAULT_CONFIG) -> ObstacleScene:
    """Load and validate the canonical obstacle scene YAML."""
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text()) or {}
    frame_id = str(data.get("frame_id", DEFAULT_FRAME)).strip()
    if not frame_id or frame_id.startswith("/"):
        raise ValueError("frame_id must be a non-empty relative ROS frame name")

    cuboid_data = data.get("cuboid")
    if not isinstance(cuboid_data, dict) or not cuboid_data:
        raise ValueError("obstacle scene must contain at least one cuboid")
    visual_data = data.get("visual") or {}
    if not isinstance(visual_data, dict):
        raise ValueError("visual must be a mapping when provided")

    cuboids = []
    for name, item in cuboid_data.items():
        name = str(name)
        if not _VALID_NAME.fullmatch(name):
            raise ValueError(f"invalid obstacle name: {name!r}")
        if not isinstance(item, dict):
            raise ValueError(f"cuboid {name!r} must be a mapping")
        dims = _float_tuple(item.get("dims"), 3, f"cuboid {name!r} dims")
        if any(value <= 0.0 for value in dims):
            raise ValueError(f"cuboid {name!r} dimensions must be positive")
        pose = _float_tuple(item.get("pose"), 7, f"cuboid {name!r} pose")
        quaternion = pose[3:]
        norm = math.sqrt(sum(value * value for value in quaternion))
        if norm < 1e-8:
            raise ValueError(f"cuboid {name!r} has a zero quaternion")
        quaternion = tuple(value / norm for value in quaternion)

        visual = visual_data.get(name) or {}
        if not isinstance(visual, dict):
            raise ValueError(f"visual entry for {name!r} must be a mapping")
        color = _float_tuple(
            visual.get("color", [0.45, 0.48, 0.52, 1.0]),
            4,
            f"visual {name!r} color",
        )
        if any(value < 0.0 or value > 1.0 for value in color):
            raise ValueError(f"visual color for {name!r} must be in [0, 1]")
        cuboids.append(
            CuboidObstacle(
                name=name,
                dims=dims,
                position=pose[:3],
                quaternion_wxyz=quaternion,
                color_rgba=color,
            )
        )

    return ObstacleScene(frame_id=frame_id, cuboids=tuple(cuboids))
