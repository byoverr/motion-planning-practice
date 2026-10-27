"""Common interface and path-tracking helpers for local planners (controllers)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from mpp.geometry import cumulative_length, resample_path
from mpp.mapping.costmap import Costmap
from mpp.robot.model import RobotParams

PATH_STEP = 0.05  # metres between points of the tracked path


@dataclass
class LocalPlan:
    v: float
    w: float
    status: str = "ok"  # "ok" | "no_valid_trajectory" | "blocked"
    best: np.ndarray | None = None  # (T, 3) predicted trajectory of the chosen command
    candidates: np.ndarray | None = None  # (K, T, 2) sampled trajectories, for drawing


class LocalPlanner(ABC):
    """Turns the global path into velocity commands at the control rate.

    ``avoids_obstacles`` tells the navigator whether this controller can drive around an
    obstacle on its own (DWA, MPPI) or only follows the path and stops (pure pursuit).
    """

    name = "base"
    avoids_obstacles = True

    def __init__(self, robot: RobotParams, dt: float = 0.1) -> None:
        self.robot = robot
        self.dt = dt
        self.path: np.ndarray | None = None
        self.s = np.zeros(0)
        self.idx = 0

    # ------------------------------------------------------------------ path tracking

    def set_path(self, path: np.ndarray) -> None:
        self.path = resample_path(np.asarray(path, float), PATH_STEP)
        self.s = cumulative_length(self.path)
        self.idx = 0
        self.reset()

    def reset(self) -> None:  # noqa: B027 (optional hook)
        """Clear internal state (warm starts etc.) when a new path arrives."""

    def update_progress(self, pose: tuple[float, float, float], window: float = 2.0) -> int:
        """Advance the index of the closest path point; never jump backwards or far ahead,
        so a path that passes close to itself is followed in order."""
        n = int(window / PATH_STEP)
        seg = self.path[self.idx : self.idx + n]
        d = np.linalg.norm(seg - np.array(pose[:2]), axis=1)
        self.idx += int(np.argmin(d))
        return self.idx

    def remaining(self) -> float:
        return float(self.s[-1] - self.s[self.idx])

    def point_at(self, ahead: float) -> np.ndarray:
        """Point on the path `ahead` metres after the closest point (clamped to the end)."""
        target = min(self.s[self.idx] + ahead, self.s[-1])
        return np.array([np.interp(target, self.s, self.path[:, k]) for k in (0, 1)])

    def local_segment(self, length: float) -> np.ndarray:
        n = int(length / PATH_STEP) + 1
        return self.path[self.idx : self.idx + n]

    def local_goal(self, costmap: Costmap, ahead: float) -> np.ndarray:
        """Look-ahead point on the path, moved further along the path if it is blocked so an
        obstacle sitting on the path does not attract the robot."""
        target = self.s[self.idx] + ahead
        k = int(np.searchsorted(self.s, target))
        k = min(k, len(self.path) - 1)
        free = costmap.is_free(self.path[k:])
        if free.size and not free[0]:
            nxt = np.nonzero(free)[0]
            if nxt.size:
                k += int(nxt[0])
        return self.path[k]

    def free_segment(
        self, costmap: Costmap, length: float, blocked_margin: float = 0.4
    ) -> tuple[np.ndarray, np.ndarray]:
        """Path points ahead (and their tangent directions) that a controller should track.

        Points inside an obstacle's inflation, and `blocked_margin` metres around them, are
        dropped: otherwise the "stay on the path" critic pulls the robot straight into the
        obstacle it is supposed to drive around.
        """
        seg = self.local_segment(length)
        if len(seg) >= 2:
            tangent = np.arctan2(np.gradient(seg[:, 1]), np.gradient(seg[:, 0]))
        else:
            tangent = np.zeros(len(seg))
        free = costmap.is_free(seg)
        if not free.all():
            blocked = seg[~free]
            near = ((seg[:, None, :] - blocked[None, :, :]) ** 2).sum(-1).min(axis=1)
            free &= near > blocked_margin**2
        return seg[free], tangent[free]

    def collision_threshold(self, costmap: Costmap, pose, safety: float) -> float:
        """Minimum clearance a predicted trajectory must keep.

        Normally robot radius + safety margin. If the robot is already closer than that (the
        lidar just revealed an obstacle next to it), demanding the full margin would forbid
        every motion and freeze the robot, so trajectories only must not get any closer.
        """
        now = float(costmap.clearance_at(np.array(pose[:2])))
        r = self.robot.radius
        return max(r + 0.005, min(r + safety, now - 0.005))

    # ------------------------------------------------------------------ interface

    @abstractmethod
    def compute(
        self, pose: tuple[float, float, float], vel: tuple[float, float], costmap: Costmap
    ) -> LocalPlan: ...

    def clamp(self, v: float, w: float, vel: tuple[float, float]) -> tuple[float, float]:
        """Respect velocity and acceleration limits."""
        r = self.robot
        v = float(np.clip(v, vel[0] - r.acc_v * self.dt, vel[0] + r.acc_v * self.dt))
        w = float(np.clip(w, vel[1] - r.acc_w * self.dt, vel[1] + r.acc_w * self.dt))
        return float(np.clip(v, r.min_v, r.max_v)), float(np.clip(w, -r.max_w, r.max_w))


def distance_to_points(pts: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Distance from every point in pts (..., 2) to the closest point of ref (M, 2)."""
    flat = pts.reshape(-1, 2)
    d2 = ((flat[:, None, :] - ref[None, :, :]) ** 2).sum(-1)
    return np.sqrt(d2.min(axis=1)).reshape(pts.shape[:-1])
