"""A* must find exactly the same optimal cost as SciPy's Dijkstra on the same graph."""

import math

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from mpp.mapping.costmap import Costmap
from mpp.planning.global_planners import make_global_planner
from mpp.world.layout import warehouse


def reference_cost(cm: Costmap, start, goal, cost_weight: float) -> float:
    """Build the 8-connected grid graph A* searches (same edge weights, no corner cutting)
    and solve it with scipy.sparse.csgraph.dijkstra."""
    free = cm.clearance >= cm.inscribed
    sx, sy = (int(v) for v in cm.world_to_cell(start))
    gx, gy = (int(v) for v in cm.world_to_cell(goal))
    free = free.copy()
    free[sy, sx] = free[gy, gx] = True
    ny, nx = free.shape
    idx = np.arange(nx * ny).reshape(ny, nx)
    rows, cols, w = [], [], []
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
        ys, xs = np.mgrid[max(0, -dy) : ny - max(0, dy), max(0, -dx) : nx - max(0, dx)]
        ok = free[ys, xs] & free[ys + dy, xs + dx]
        if dx and dy:
            ok &= free[ys, xs + dx] & free[ys + dy, xs]
        step = math.sqrt(2) if dx and dy else 1.0
        rows.append(idx[ys, xs][ok])
        cols.append(idx[ys + dy, xs + dx][ok])
        w.append(step * (1.0 + cost_weight * cm.cost[ys + dy, xs + dx][ok]))
    g = coo_matrix(
        (np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(nx * ny, nx * ny)
    ).tocsr()
    d = dijkstra(g, directed=True, indices=idx[sy, sx])
    return float(d[idx[gy, gx]]) * cm.res


@pytest.mark.parametrize(("s", "g"), [("S3", "G1"), ("S2", "G2"), ("S1", "G3"), ("S1", "G4")])
@pytest.mark.parametrize("cost_weight", [0.0, 2.0])
def test_astar_is_optimal(s, g, cost_weight):
    lay = warehouse()
    cm = Costmap.from_layout(lay)
    a, b = lay.pose(s).xy, lay.pose(g).xy
    res = make_global_planner("astar", cost_weight=cost_weight, smooth=False).plan(cm, a, b)
    assert res.info["grid_cost"] == pytest.approx(reference_cost(cm, a, b, cost_weight), rel=1e-9)
