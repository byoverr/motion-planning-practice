"""A* on the costmap grid (8-connected), the default global planner.

Edge cost = step length * (1 + cost_weight * inflation cost of the target cell), so the path
prefers the middle of aisles instead of hugging the shelves (Nav2's NavFn / Smac 2D do the
same). With ``heuristic_weight > 1`` it becomes weighted A* (faster, no longer optimal);
with ``heuristic_weight = 0`` it is Dijkstra.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

from mpp.mapping.costmap import Costmap
from mpp.planning.global_planners.base import GlobalPlanner, PlanResult, smooth_path

SQRT2 = math.sqrt(2.0)
NEIGHBOURS = [
    (1, 0, 1.0),
    (-1, 0, 1.0),
    (0, 1, 1.0),
    (0, -1, 1.0),
    (1, 1, SQRT2),
    (1, -1, SQRT2),
    (-1, 1, SQRT2),
    (-1, -1, SQRT2),
]


def octile(dx: int, dy: int) -> float:
    dx, dy = abs(dx), abs(dy)
    return (dx + dy) + (SQRT2 - 2.0) * min(dx, dy)


class AStar(GlobalPlanner):
    name = "astar"

    def __init__(
        self,
        cost_weight: float = 2.0,
        heuristic_weight: float = 1.0,
        smooth: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed)
        self.cost_weight = cost_weight
        self.heuristic_weight = heuristic_weight
        self.smooth = smooth

    def _plan(self, costmap: Costmap, start: np.ndarray, goal: np.ndarray) -> PlanResult:
        nx, ny = costmap.nx, costmap.ny
        free = costmap.clearance >= costmap.inscribed
        cost = costmap.cost
        sx, sy = (int(v) for v in costmap.world_to_cell(start))
        gx, gy = (int(v) for v in costmap.world_to_cell(goal))
        free = free.copy()
        free[sy, sx] = free[gy, gx] = True  # the exact points were validated by the caller

        g = np.full((ny, nx), np.inf)
        parent = np.full((ny, nx), -1, dtype=np.int64)
        closed = np.zeros((ny, nx), dtype=bool)
        hw = self.heuristic_weight
        g[sy, sx] = 0.0
        heap = [(hw * octile(gx - sx, gy - sy), 0.0, sx, sy)]
        expanded = 0
        while heap:
            _, gc, x, y = heapq.heappop(heap)
            if closed[y, x]:
                continue
            closed[y, x] = True
            expanded += 1
            if x == gx and y == gy:
                break
            for dx, dy, step in NEIGHBOURS:
                xn, yn = x + dx, y + dy
                if xn < 0 or yn < 0 or xn >= nx or yn >= ny:
                    continue
                if closed[yn, xn] or not free[yn, xn]:
                    continue
                if dx and dy and not (free[y, xn] and free[yn, x]):
                    continue  # do not cut diagonally between two blocked cells
                ng = gc + step * (1.0 + self.cost_weight * cost[yn, xn])
                if ng < g[yn, xn]:
                    g[yn, xn] = ng
                    parent[yn, xn] = y * nx + x
                    heapq.heappush(heap, (ng + hw * octile(gx - xn, gy - yn), ng, xn, yn))

        ys, xs = np.nonzero(closed)
        explored = costmap.cell_to_world(xs, ys)
        if not closed[gy, gx]:
            return PlanResult(None, self.name, iterations=expanded, explored=explored)

        cells = []
        idx = gy * nx + gx
        while idx >= 0:
            cells.append((idx % nx, idx // nx))
            idx = parent[idx // nx, idx % nx]
        cells.reverse()
        c = np.array(cells)
        path = costmap.cell_to_world(c[:, 0], c[:, 1])
        path[0], path[-1] = start, goal
        if self.smooth:
            path = smooth_path(path, costmap)
        return PlanResult(
            path,
            self.name,
            iterations=expanded,
            first_solution_time=None,
            explored=explored,
            info={"grid_cost": float(g[gy, gx]) * costmap.res},
        )
