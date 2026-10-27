"""RRT-Connect (Kuffner & LaValle, 2000).

Two RRTs grow from the start and from the goal. Each iteration extends one tree one step
towards a random sample, then greedily "connects" the other tree to the new vertex. The trees
swap roles every iteration. It is fast to find *a* path but makes no attempt to find a short
one, so the raw path is shortcut afterwards.
"""

from __future__ import annotations

import time

import numpy as np

from mpp.geometry import resample_path
from mpp.mapping.costmap import Costmap
from mpp.planning.global_planners.base import (
    ClearanceChecker,
    GlobalPlanner,
    PlanResult,
    shortcut_path,
)
from mpp.planning.global_planners.sampling import Sampler, extract_path

TRAPPED, ADVANCED, REACHED = 0, 1, 2


class _Tree:
    def __init__(self, root: np.ndarray, capacity: int = 4096) -> None:
        self.pts = np.empty((capacity, 2))
        self.parent = np.empty(capacity, dtype=np.int64)
        self.pts[0] = root
        self.parent[0] = -1
        self.n = 1

    def add(self, p: np.ndarray, parent: int) -> int:
        if self.n == len(self.pts):
            self.pts = np.concatenate([self.pts, np.empty_like(self.pts)])
            self.parent = np.concatenate([self.parent, np.empty_like(self.parent)])
        self.pts[self.n] = p
        self.parent[self.n] = parent
        self.n += 1
        return self.n - 1

    def nearest(self, p: np.ndarray) -> int:
        d = self.pts[: self.n] - p
        return int(np.argmin(np.einsum("ij,ij->i", d, d)))

    def path_to_root(self, i: int) -> np.ndarray:
        return extract_path(self.pts[: self.n], self.parent[: self.n], i)

    def edges(self) -> np.ndarray:
        idx = np.arange(1, self.n)
        return np.stack([self.pts[self.parent[idx]], self.pts[idx]], axis=1)


class RRTConnect(GlobalPlanner):
    name = "rrt_connect"

    def __init__(
        self,
        step: float = 0.35,
        time_limit: float = 2.0,
        max_iterations: int = 50_000,
        simplify: bool = True,
        clearance_margin: float = 0.04,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed)
        self.margin = clearance_margin
        self.step = step
        self.time_limit = time_limit
        self.max_iterations = max_iterations
        self.simplify = simplify

    def _extend(self, cm: Costmap, tree: _Tree, target: np.ndarray) -> tuple[int, int]:
        i = tree.nearest(target)
        q = tree.pts[i]
        d = target - q
        dist = float(np.hypot(*d))
        if dist < 1e-9:
            return REACHED, i
        new = target if dist <= self.step else q + d * (self.step / dist)
        if not self._checker.segment_free(q, new):
            return TRAPPED, -1
        j = tree.add(new, i)
        return (REACHED if dist <= self.step else ADVANCED), j

    def _connect(self, cm: Costmap, tree: _Tree, target: np.ndarray) -> tuple[int, int]:
        while True:
            status, j = self._extend(cm, tree, target)
            if status != ADVANCED:
                return status, j

    def _plan(self, costmap: Costmap, start: np.ndarray, goal: np.ndarray) -> PlanResult:
        t0 = time.perf_counter()
        self._checker = ClearanceChecker(costmap, self.margin, start, goal)
        sampler = Sampler(costmap, self.rng)
        ta, tb = _Tree(start), _Tree(goal)
        a_is_start = True
        path = None
        it = 0
        for it in range(1, self.max_iterations + 1):  # noqa: B007 (`it` is reported below)
            if time.perf_counter() - t0 > self.time_limit:
                break
            q_rand = sampler.uniform(1, free_only=False)[0]
            status, j = self._extend(costmap, ta, q_rand)
            if status != TRAPPED:
                status2, k = self._connect(costmap, tb, ta.pts[j])
                if status2 == REACHED:
                    pa, pb = ta.path_to_root(j), tb.path_to_root(k)
                    sa, sb = (pa, pb) if a_is_start else (pb, pa)
                    path = np.vstack([sa, sb[::-1][1:]])
                    break
            ta, tb = tb, ta
            a_is_start = not a_is_start
        first = time.perf_counter() - t0 if path is not None else None
        edges = np.concatenate([ta.edges(), tb.edges()])
        if path is None:
            return PlanResult(None, self.name, iterations=it, tree=edges)
        raw_length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        if self.simplify:
            path = shortcut_path(path, costmap, self.rng, checker=self._checker)
        return PlanResult(
            resample_path(path, 0.05),
            self.name,
            iterations=it,
            first_solution_time=first,
            tree=edges,
            info={"raw_length": raw_length, "vertices": ta.n + tb.n},
        )
