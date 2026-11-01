"""Small geometry helpers shared by b2w_hardware nodes."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import PointCloud2


def quaternion_from_yaw(yaw: float) -> Quaternion:
    q = Quaternion()
    q.w = math.cos(yaw / 2.0)
    q.z = math.sin(yaw / 2.0)
    return q


def yaw_from_quaternion(q: Quaternion) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def pointcloud_xyz(msg: PointCloud2) -> Optional[np.ndarray]:
    """Return Nx3 float array from a PointCloud2 message, or None."""
    point_step = int(msg.point_step)
    row_step = int(msg.row_step)
    width = int(msg.width)
    height = int(msg.height)
    if point_step <= 0 or row_step < point_step or width * height == 0:
        return None

    fields = {field.name: field for field in msg.fields}
    if not all(name in fields for name in ("x", "y", "z")):
        return None

    raw = np.frombuffer(msg.data, dtype=np.uint8)
    count = len(raw) // point_step
    raw = raw[: count * point_step].reshape(count, point_step)

    def _column(name: str) -> np.ndarray:
        offset = int(fields[name].offset)
        return (
            raw[:, offset : offset + 4]
            .copy()
            .view(np.float32)
            .reshape(-1)
            .astype(np.float64)
        )

    xyz = np.column_stack((_column("x"), _column("y"), _column("z")))
    return xyz[np.isfinite(xyz).all(axis=1)]
