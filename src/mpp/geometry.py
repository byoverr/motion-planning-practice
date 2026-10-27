"""Small 2D geometry helpers shared by planners, the navigator and plotting."""

from __future__ import annotations

import numpy as np


def wrap_angle(a: float | np.ndarray) -> float | np.ndarray:
    """Wrap an angle (or array of angles) to [-pi, pi)."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def path_length(path: np.ndarray) -> float:
    if path is None or len(path) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def resample_path(path: np.ndarray, step: float) -> np.ndarray:
    """Return points spaced `step` metres apart along the polyline (first and last kept)."""
    path = np.asarray(path, dtype=float)
    if len(path) < 2:
        return path.copy()
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-9:
        return path[:1].copy()
    n = max(int(np.ceil(s[-1] / step)), 1)
    t = np.linspace(0.0, s[-1], n + 1)
    return np.column_stack([np.interp(t, s, path[:, 0]), np.interp(t, s, path[:, 1])])


def cumulative_length(path: np.ndarray) -> np.ndarray:
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def to_robot_frame(points: np.ndarray, pose: tuple[float, float, float]) -> np.ndarray:
    """Express world points (N, 2) in the robot frame (x forward, y left)."""
    x, y, yaw = pose
    c, s = np.cos(yaw), np.sin(yaw)
    d = np.asarray(points) - np.array([x, y])
    return np.column_stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]])


def rollout_unicycle(
    pose: tuple[float, float, float], v: np.ndarray, w: np.ndarray, dt: float
) -> np.ndarray:
    """Integrate unicycle kinematics for a batch of control sequences.

    v, w: arrays of shape (K, T) with the command applied during each step.
    Returns states of shape (K, T, 3) after each step (the start pose is not included).
    """
    x0, y0, th0 = pose
    th = th0 + np.cumsum(w * dt, axis=1)
    # use the mid-point heading of each step, it is noticeably more accurate on arcs
    th_mid = th - 0.5 * w * dt
    x = x0 + np.cumsum(v * np.cos(th_mid) * dt, axis=1)
    y = y0 + np.cumsum(v * np.sin(th_mid) * dt, axis=1)
    return np.stack([x, y, th], axis=-1)
