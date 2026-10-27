"""Common interface for global planners."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from mpp.geometry import path_length
from mpp.mapping.costmap import Costmap


@dataclass
class PlanResult:
    path: np.ndarray | None  # (N, 2) waypoints from start to goal, None on failure
    planner: str
    time: float = 0.0  # wall-clock seconds
    iterations: int = 0  # expanded cells / processed edges / tree extensions
    first_solution_time: float | None = None  # anytime planners: when the first path appeared
    tree: np.ndarray | None = None  # (E, 2, 2) edges of a sampling planner's tree, for plotting
    explored: np.ndarray | None = None  # (M, 2) cells expanded by a grid search, for plotting
    info: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.path is not None

    @property
    def length(self) -> float:
        return path_length(self.path) if self.path is not None else float("inf")


class GlobalPlanner(ABC):
    """Plans a collision-free path for a disc robot on a costmap.

    Subclasses implement `_plan`. Validation of start and goal and timing are done here.
    """

    name = "base"

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def plan(self, costmap: Costmap, start: np.ndarray, goal: np.ndarray) -> PlanResult:
        start = np.asarray(start, float)
        goal = np.asarray(goal, float)
        t0 = time.perf_counter()
        if not costmap.is_free(goal):
            return PlanResult(None, self.name, 0.0, info={"reason": "goal in collision"})
        if not costmap.is_free(start):
            return PlanResult(None, self.name, 0.0, info={"reason": "start in collision"})
        res = self._plan(costmap, start, goal)
        res.time = time.perf_counter() - t0
        res.planner = self.name
        return res

    @abstractmethod
    def _plan(self, costmap: Costmap, start: np.ndarray, goal: np.ndarray) -> PlanResult: ...


def shortcut_path(
    path: np.ndarray,
    costmap: Costmap,
    rng: np.random.Generator,
    iterations: int = 200,
    margin: float = 0.0,
) -> np.ndarray:
    """Random shortcutting (as in OMPL's PathSimplifier): try to replace sub-paths by
    straight segments that stay collision-free. Sampling planners need this, their raw
    output zig-zags."""
    pts = [p for p in np.asarray(path, float)]

    def ok(i: int, j: int) -> bool:
        m = 0.0 if i == 0 or j == len(pts) - 1 else margin
        return costmap.segment_free(pts[i], pts[j], margin=m)

    for _ in range(iterations):
        if len(pts) < 3:
            break
        i, j = sorted(rng.choice(len(pts), size=2, replace=False))
        if j - i < 2:
            continue
        if ok(i, j):
            pts = pts[: i + 1] + pts[j:]
    # final greedy pass from the start
    out = [pts[0]]
    k = 0
    while k < len(pts) - 1:
        j = len(pts) - 1
        while j > k + 1 and not ok(k, j):
            j -= 1
        out.append(pts[j])
        k = j
    return np.array(out)


def smooth_path(
    path: np.ndarray,
    costmap: Costmap,
    w_data: float = 0.2,
    w_smooth: float = 0.3,
    iterations: int = 60,
    spacing: float = 0.05,
) -> np.ndarray:
    """Gradient smoothing (like Nav2's simple smoother) that keeps every point collision-free.

    Grid planners return 45-degree staircases; this rounds them off while the data term keeps
    the result close to the original path, so it does not cut corners into the inflation.
    """
    from mpp.geometry import resample_path

    orig = resample_path(np.asarray(path, float), spacing)
    if len(orig) < 5:
        return orig
    x = orig.copy()
    for _ in range(iterations):
        upd = w_data * (orig[1:-1] - x[1:-1]) + w_smooth * (x[:-2] + x[2:] - 2 * x[1:-1])
        cand = x[1:-1] + upd
        ok = costmap.is_free(cand)
        x[1:-1][ok] = cand[ok]
    return x
