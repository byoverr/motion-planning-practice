"""Shared pieces of the sampling-based planners: samplers, the RGG radius, path extraction."""

from __future__ import annotations

import math

import numpy as np

from mpp.mapping.costmap import Costmap
from mpp.planning.global_planners.base import ClearanceChecker


class Sampler:
    """Uniform samples in the map and informed samples inside the ellipse
    {x : |x - start| + |x - goal| <= c_best} (Gammell et al., Informed RRT*)."""

    def __init__(
        self, costmap: Costmap, rng: np.random.Generator, checker: ClearanceChecker | None = None
    ) -> None:
        self.cm = costmap
        self.rng = rng
        self.free = checker.points_free if checker is not None else costmap.is_free
        x0, x1, y0, y1 = costmap.extent
        self.lo = np.array([x0, y0])
        self.hi = np.array([x1, y1])
        self.free_area = float((costmap.clearance >= costmap.inscribed).sum()) * costmap.res**2

    def uniform(self, n: int, free_only: bool = True) -> np.ndarray:
        out = []
        need = n
        while need > 0:
            pts = self.lo + self.rng.random((max(2 * need, 16), 2)) * (self.hi - self.lo)
            if free_only:
                pts = pts[self.free(pts)]
            out.append(pts[:need])
            need -= len(out[-1])
        return np.concatenate(out)[:n]

    def informed(
        self, n: int, start: np.ndarray, goal: np.ndarray, c_best: float, max_tries: int = 50
    ) -> np.ndarray:
        if not math.isfinite(c_best):
            return self.uniform(n)
        c_min = float(np.linalg.norm(goal - start))
        centre = 0.5 * (start + goal)
        a = 0.5 * c_best
        b = 0.5 * math.sqrt(max(c_best**2 - c_min**2, 1e-12))
        ang = math.atan2(goal[1] - start[1], goal[0] - start[0])
        rot = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
        out, need, tries = [], n, 0
        while need > 0 and tries < max_tries:
            tries += 1
            m = max(2 * need, 16)
            r = np.sqrt(self.rng.random(m))
            t = self.rng.random(m) * 2 * math.pi
            unit = np.column_stack([r * np.cos(t), r * np.sin(t)])
            pts = centre + (unit * [a, b]) @ rot.T
            inside = np.all((pts >= self.lo) & (pts <= self.hi), axis=1)
            pts = pts[inside]
            pts = pts[self.free(pts)]
            out.append(pts[:need])
            need -= len(out[-1])
        return np.concatenate(out)[:n] if out else np.empty((0, 2))

    def informed_measure(self, start: np.ndarray, goal: np.ndarray, c_best: float) -> float:
        """Lebesgue measure of the informed set (bounded by the free area)."""
        if not math.isfinite(c_best):
            return self.free_area
        c_min = float(np.linalg.norm(goal - start))
        ellipse = math.pi * 0.25 * c_best * math.sqrt(max(c_best**2 - c_min**2, 0.0))
        return min(self.free_area, ellipse)


def rgg_radius(n_vertices: int, measure: float, eta: float = 1.1, d: int = 2) -> float:
    """Connection radius of a random geometric graph that keeps asymptotic optimality
    (Karaman & Frazzoli 2011, as used by BIT* and AIT*)."""
    n = max(n_vertices, 2)
    zeta = math.pi  # area of the unit disc
    return (
        eta * 2.0 * (1.0 + 1.0 / d) ** (1.0 / d) * (measure / zeta * math.log(n) / n) ** (1.0 / d)
    )


def extract_path(points: np.ndarray, parent: np.ndarray, goal: int) -> np.ndarray:
    idx, out = goal, []
    while idx >= 0:
        out.append(points[idx])
        idx = int(parent[idx])
    return np.array(out[::-1])


def tree_edges(
    points: np.ndarray, parent: np.ndarray, mask: np.ndarray | None = None
) -> np.ndarray:
    idx = np.nonzero(parent >= 0)[0]
    if mask is not None:
        idx = idx[mask[idx]]
    return np.stack([points[parent[idx]], points[idx]], axis=1)


class SearchGraph:
    """Vertices of a batch-sampled random geometric graph plus a forward search tree on it.

    Used by BIT* and AIT*. Index 0 is the start, index 1 the goal. Samples are added in
    batches; neighbours (r-disc) are recomputed after every batch. Collision checks of edges
    are cached, so an edge is never checked twice.
    """

    def __init__(
        self,
        costmap: Costmap,
        start: np.ndarray,
        goal: np.ndarray,
        checker: ClearanceChecker | None = None,
    ) -> None:
        self.cm = costmap
        self.checker = checker or ClearanceChecker(costmap, 0.0, start, goal)
        self.start_xy, self.goal_xy = np.asarray(start, float), np.asarray(goal, float)
        self.n = 0
        cap = 1024
        self.pts = np.empty((cap, 2))
        self.g_hat = np.empty(cap)  # straight-line distance from the start (admissible)
        self.h_hat = np.empty(cap)  # straight-line distance to the goal (admissible)
        self.alive = np.zeros(cap, dtype=bool)
        self.in_tree = np.zeros(cap, dtype=bool)
        self.g = np.full(cap, np.inf)  # cost-to-come through the forward tree
        self.parent = np.full(cap, -1, dtype=np.int64)
        self.children: list[set[int]] = []
        self.nbrs: list[np.ndarray] = []
        self.edge_ok: dict[tuple[int, int], bool] = {}
        self.collision_checks = 0
        self.add(np.vstack([self.start_xy, self.goal_xy]))
        self.start, self.goal = 0, 1
        self.g[0] = 0.0
        self.in_tree[0] = True

    def _grow(self, need: int) -> None:
        cap = len(self.pts)
        if self.n + need <= cap:
            return
        new = max(2 * cap, self.n + need)
        for name, fill in (
            ("pts", None),
            ("g_hat", None),
            ("h_hat", None),
            ("alive", False),
            ("in_tree", False),
            ("g", np.inf),
            ("parent", -1),
        ):
            old = getattr(self, name)
            shape = (new,) + old.shape[1:]
            arr = (
                np.empty(shape, dtype=old.dtype)
                if fill is None
                else np.full(shape, fill, old.dtype)
            )
            arr[:cap] = old
            setattr(self, name, arr)

    def add(self, pts: np.ndarray) -> np.ndarray:
        k = len(pts)
        self._grow(k)
        idx = np.arange(self.n, self.n + k)
        self.pts[idx] = pts
        self.g_hat[idx] = np.linalg.norm(pts - self.start_xy, axis=1)
        self.h_hat[idx] = np.linalg.norm(pts - self.goal_xy, axis=1)
        self.alive[idx] = True
        self.children.extend(set() for _ in range(k))
        self.n += k
        return idx

    @property
    def c_best(self) -> float:
        return float(self.g[self.goal])

    def prune_samples(self) -> None:
        """Drop unconnected samples that cannot lie on a better path (outside the ellipse)."""
        n = self.n
        bad = self.alive[:n] & ~self.in_tree[:n] & (self.g_hat[:n] + self.h_hat[:n] >= self.c_best)
        bad[self.goal] = False
        self.alive[:n][bad] = False

    def build_neighbours(self, radius: float) -> None:
        from scipy.spatial import cKDTree

        idx = np.nonzero(self.alive[: self.n])[0]
        kd = cKDTree(self.pts[idx])
        lists = kd.query_ball_point(self.pts[idx], radius)
        self.nbrs = [np.empty(0, dtype=np.int64)] * self.n
        for i, lst in zip(idx, lists, strict=True):
            arr = idx[np.asarray(lst, dtype=np.int64)]
            self.nbrs[i] = arr[arr != i]
        # edges already known to be in collision are not part of the graph any more
        for (u, v), ok in self.edge_ok.items():
            if not ok and self.alive[u] and self.alive[v]:
                self.nbrs[u] = self.nbrs[u][self.nbrs[u] != v]
                self.nbrs[v] = self.nbrs[v][self.nbrs[v] != u]
        self.radius = radius

    def remove_edge(self, u: int, v: int) -> None:
        self.nbrs[u] = self.nbrs[u][self.nbrs[u] != v]
        self.nbrs[v] = self.nbrs[v][self.nbrs[v] != u]

    def dist(self, u: int, v: int | np.ndarray) -> float | np.ndarray:
        d = self.pts[v] - self.pts[u]
        return np.hypot(d[..., 0], d[..., 1])

    def edge_valid(self, u: int, v: int) -> bool:
        key = (u, v) if u < v else (v, u)
        ok = self.edge_ok.get(key)
        if ok is None:
            ok = self.checker.segment_free(self.pts[u], self.pts[v])
            self.edge_ok[key] = ok
            self.collision_checks += 1
        return ok

    def known_invalid(self, u: int, v: int) -> bool:
        key = (u, v) if u < v else (v, u)
        return self.edge_ok.get(key) is False

    def set_parent(self, v: int, u: int, cost: float) -> None:
        """Make u the parent of v (adds v to the tree or rewires it) and update the costs of
        v's whole subtree."""
        if self.in_tree[v] and self.parent[v] >= 0:
            self.children[self.parent[v]].discard(v)
        self.in_tree[v] = True
        self.parent[v] = u
        self.children[u].add(v)
        self.g[v] = self.g[u] + cost
        stack = [v]
        while stack:
            p = stack.pop()
            for c in self.children[p]:
                self.g[c] = self.g[p] + self.dist(p, c)
                stack.append(c)

    def path(self) -> np.ndarray:
        return extract_path(self.pts, self.parent, self.goal)

    def tree_edges(self) -> np.ndarray:
        return tree_edges(self.pts[: self.n], self.parent[: self.n], self.in_tree[: self.n])
