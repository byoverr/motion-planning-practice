"""BIT*, Batch Informed Trees (Gammell, Srinivasa & Barfoot, 2015).

BIT* sees a batch of samples as an implicit random geometric graph and searches it in order
of *potential* solution cost, like A* on a graph whose edges are not collision-checked yet:

* the edge queue is ordered by  g_T(v) + c_hat(v, x) + h_hat(x)  (straight-line estimates);
* an edge is collision-checked only when it is the most promising one left;
* when a solution exists, new samples are drawn only inside the informed ellipse
  |x - start| + |x - goal| < c_best, so later batches refine the path.

This is a compact educational version: r-disc neighbourhoods, lazy priority queues and
pruning of unconnected samples only (tree vertices are kept).
"""

from __future__ import annotations

import heapq
import math
import time

import numpy as np

from mpp.geometry import resample_path
from mpp.mapping.costmap import Costmap
from mpp.planning.global_planners.base import ClearanceChecker, GlobalPlanner, PlanResult
from mpp.planning.global_planners.sampling import Sampler, SearchGraph, rgg_radius


class BITStar(GlobalPlanner):
    name = "bit_star"

    def __init__(
        self,
        batch_size: int = 150,
        time_limit: float = 1.0,
        eta: float = 1.1,
        stop_on_first: bool = False,
        clearance_margin: float = 0.04,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed)
        self.margin = clearance_margin
        self.batch_size = batch_size
        self.time_limit = time_limit
        self.eta = eta
        self.stop_on_first = stop_on_first

    def _plan(self, costmap: Costmap, start: np.ndarray, goal: np.ndarray) -> PlanResult:
        t0 = time.perf_counter()
        checker = ClearanceChecker(costmap, self.margin, start, goal)
        sampler = Sampler(costmap, self.rng, checker)
        G = SearchGraph(costmap, start, goal, checker)
        qv: list[tuple[float, int]] = []  # vertex queue: g_T(v) + h_hat(v)
        qe: list[tuple[float, int, int]] = []  # edge queue: g_T(v) + c_hat(v,x) + h_hat(x)
        expanded_in_old_batch = np.zeros(0, dtype=bool)
        first_time = None
        batches = edges_processed = 0

        def best_qv() -> float:
            return qv[0][0] if qv else math.inf

        def best_qe() -> float:
            return qe[0][0] if qe else math.inf

        def expand(v: int) -> None:
            c_best = G.c_best
            near = G.nbrs[v]
            if len(near) == 0:
                return
            c_hat = G.dist(v, near)
            samples = near[~G.in_tree[near]]
            cs = c_hat[~G.in_tree[near]]
            ok = G.g_hat[v] + cs + G.h_hat[samples] < c_best
            for x, c in zip(samples[ok], cs[ok], strict=True):
                heapq.heappush(qe, (G.g[v] + c + G.h_hat[x], v, int(x)))
            if v < len(expanded_in_old_batch) and expanded_in_old_batch[v]:
                return  # rewiring edges of old vertices were already considered
            tree = near[G.in_tree[near]]
            ct = c_hat[G.in_tree[near]]
            ok = (
                (G.g_hat[v] + ct + G.h_hat[tree] < c_best)
                & (G.g[v] + ct < G.g[tree])
                & (G.parent[tree] != v)
                & (tree != G.parent[v])
            )
            for w, c in zip(tree[ok], ct[ok], strict=True):
                heapq.heappush(qe, (G.g[v] + c + G.h_hat[w], v, int(w)))

        while time.perf_counter() - t0 < self.time_limit:
            if not qe and not qv:
                if self.stop_on_first and math.isfinite(G.c_best):
                    break
                # ---- new batch
                if math.isfinite(G.c_best):
                    G.prune_samples()
                G.add(sampler.informed(self.batch_size, start, goal, G.c_best))
                batches += 1
                n_alive = int(G.alive[: G.n].sum())
                measure = sampler.informed_measure(start, goal, G.c_best)
                G.build_neighbours(rgg_radius(n_alive, measure, self.eta))
                expanded_in_old_batch = G.in_tree[: G.n].copy()
                c_best = G.c_best
                for v in np.nonzero(G.in_tree[: G.n])[0]:
                    if G.g_hat[v] + G.h_hat[v] < c_best:
                        heapq.heappush(qv, (G.g[v] + G.h_hat[v], int(v)))

            # expand vertices while they might give better edges than the best queued edge
            while qv and best_qv() <= best_qe():
                _, v = heapq.heappop(qv)
                expand(v)
            if not qe:
                qv.clear()
                continue

            _, v, x = heapq.heappop(qe)
            edges_processed += 1
            c_best = G.c_best
            c_hat = float(G.dist(v, x))
            if G.g[v] + c_hat + G.h_hat[x] >= c_best:
                qe.clear()  # nothing left in this batch can improve the solution
                qv.clear()
                continue
            if G.g[v] + c_hat >= G.g[x]:
                continue  # stale entry: x is already reached more cheaply
            if G.g_hat[v] + c_hat + G.h_hat[x] >= c_best:
                continue
            if not G.edge_valid(v, x):
                continue
            # the edge is collision-free and improves x: add or rewire
            was_in_tree = bool(G.in_tree[x])
            G.set_parent(x, v, c_hat)
            if not was_in_tree:
                heapq.heappush(qv, (G.g[x] + G.h_hat[x], int(x)))
            if x == G.goal and first_time is None:
                first_time = time.perf_counter() - t0

        info = {
            "batches": batches,
            "vertices": int(G.in_tree[: G.n].sum()),
            "samples": int(G.alive[: G.n].sum()),
            "collision_checks": G.collision_checks,
        }
        if not math.isfinite(G.c_best):
            return PlanResult(
                None, self.name, iterations=edges_processed, tree=G.tree_edges(), info=info
            )
        path = G.path()
        info["raw_length"] = G.c_best
        return PlanResult(
            resample_path(path, 0.05),
            self.name,
            iterations=edges_processed,
            first_solution_time=first_time,
            tree=G.tree_edges(),
            info=info,
        )
