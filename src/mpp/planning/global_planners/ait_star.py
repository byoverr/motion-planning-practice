"""AIT*, Adaptively Informed Trees (Strub & Gammell, ICRA 2020).

BIT* guides its search with the straight-line distance to the goal, which is a poor guess in
a warehouse full of shelves. AIT* runs *two* searches on the same batch of samples:

* a **reverse search** (LPA*, from the goal) over the graph with straight-line edge costs and
  no collision checks. It yields a much better cost-to-go heuristic h(x) that already knows
  about the graph's connectivity;
* a **forward search** (like BIT*) from the start ordered by  g(u) + c_hat(u, v) + h(v),
  which collision-checks edges lazily.

When the forward search finds that an edge is in collision, the edge is removed from the graph
and, if the reverse search tree used it, the reverse search is repaired incrementally (that is
what LPA* is for) and the forward queue is re-sorted with the updated heuristic.

Educational version: r-disc graph, one reverse search per batch, unconnected samples pruned.
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

INF = math.inf


class _ReverseSearch:
    """Lifelong Planning A* from the goal towards the start on the sample graph."""

    def __init__(self, G: SearchGraph) -> None:
        self.G = G
        n = G.n
        self.h = np.full(n, INF)  # cost-to-go estimate ("g" of LPA*)
        self.rhs = np.full(n, INF)
        self.rparent = np.full(n, -1, dtype=np.int64)
        self.heap: list[tuple[float, float, int]] = []
        self.qkey: dict[int, tuple[float, float]] = {}
        self.expansions = 0
        self.rhs[G.goal] = 0.0
        self._push(G.goal)

    def key(self, v: int) -> tuple[float, float]:
        m = min(self.h[v], self.rhs[v])
        return (m + self.G.g_hat[v], m)  # focus the reverse search towards the start

    def _push(self, v: int) -> None:
        k = self.key(v)
        self.qkey[v] = k
        heapq.heappush(self.heap, (k[0], k[1], v))

    def _top(self) -> tuple[float, float]:
        while self.heap:
            k0, k1, v = self.heap[0]
            if self.qkey.get(v) == (k0, k1):
                return (k0, k1)
            heapq.heappop(self.heap)  # stale entry
        return (INF, INF)

    def update_vertex(self, v: int) -> None:
        G = self.G
        if v != G.goal:
            nb = G.nbrs[v]
            if len(nb):
                costs = self.h[nb] + G.dist(v, nb)
                j = int(np.argmin(costs))
                self.rhs[v], self.rparent[v] = costs[j], (nb[j] if math.isfinite(costs[j]) else -1)
            else:
                self.rhs[v], self.rparent[v] = INF, -1
        self.qkey.pop(v, None)
        if self.h[v] != self.rhs[v]:
            self._push(v)

    def compute(self) -> None:
        G = self.G
        s = G.start
        while True:
            top = self._top()
            if top == (INF, INF) or (top >= self.key(s) and self.rhs[s] == self.h[s]):
                return
            _, _, u = heapq.heappop(self.heap)
            del self.qkey[u]
            self.expansions += 1
            nb = G.nbrs[u]
            if self.h[u] > self.rhs[u]:  # over-consistent: lower h and relax the neighbours
                self.h[u] = self.rhs[u]
                if len(nb) == 0:
                    continue
                cand = self.h[u] + G.dist(u, nb)
                better = (cand < self.rhs[nb]) & (nb != G.goal)
                for v, c in zip(nb[better], cand[better], strict=True):
                    self.rhs[v], self.rparent[v] = c, u
                    self.qkey.pop(int(v), None)
                    if self.h[v] != self.rhs[v]:
                        self._push(int(v))
            else:  # under-consistent: the estimate was too optimistic, recompute dependants
                self.h[u] = INF
                self.update_vertex(u)
                for v in nb[self.rparent[nb] == u]:
                    self.update_vertex(int(v))


class AITStar(GlobalPlanner):
    name = "ait_star"

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
        fq: list[tuple[float, int, int]] = []
        rev: _ReverseSearch | None = None
        first_time = None
        batches = edges_processed = repairs = reverse_expansions = 0

        def push_outgoing(u: int) -> None:
            nb = G.nbrs[u]
            if len(nb) == 0:
                return
            c = G.dist(u, nb)
            h = rev.h[nb]
            ok = np.isfinite(h) & (G.g[u] + c < G.g[nb]) & (nb != G.parent[u])
            ok &= G.g[u] + c + h < G.c_best
            for v, cc, hh in zip(nb[ok], c[ok], h[ok], strict=True):
                heapq.heappush(fq, (G.g[u] + cc + hh, u, int(v)))

        def rebuild_forward_queue() -> None:
            fq.clear()
            for u in np.nonzero(G.in_tree[: G.n])[0]:
                push_outgoing(int(u))

        new_batch = True
        while time.perf_counter() - t0 < self.time_limit:
            if new_batch:
                if self.stop_on_first and math.isfinite(G.c_best):
                    break
                if math.isfinite(G.c_best):
                    G.prune_samples()
                G.add(sampler.informed(self.batch_size, start, goal, G.c_best))
                batches += 1
                n_alive = int(G.alive[: G.n].sum())
                measure = sampler.informed_measure(start, goal, G.c_best)
                G.build_neighbours(rgg_radius(n_alive, measure, self.eta))
                if rev is not None:
                    reverse_expansions += rev.expansions
                rev = _ReverseSearch(G)
                rev.compute()
                rebuild_forward_queue()
                new_batch = False

            if not fq or fq[0][0] >= G.c_best:
                new_batch = True
                continue
            _, u, v = heapq.heappop(fq)
            edges_processed += 1
            c = float(G.dist(u, v))
            if G.g[u] + c >= G.g[v] or G.g[u] + c + rev.h[v] >= G.c_best:
                continue
            if G.edge_valid(u, v):
                G.set_parent(v, u, c)
                if v == G.goal and first_time is None:
                    first_time = time.perf_counter() - t0
                push_outgoing(v)
                continue
            # collision: remove the edge, repair the reverse search if it relied on it
            G.remove_edge(u, v)
            if rev.rparent[u] == v or rev.rparent[v] == u:
                repairs += 1
                if rev.rparent[u] == v:
                    rev.update_vertex(u)
                if rev.rparent[v] == u:
                    rev.update_vertex(v)
                rev.compute()
                rebuild_forward_queue()

        if rev is not None:
            reverse_expansions += rev.expansions
        info = {
            "batches": batches,
            "vertices": int(G.in_tree[: G.n].sum()),
            "samples": int(G.alive[: G.n].sum()),
            "collision_checks": G.collision_checks,
            "reverse_repairs": repairs,
            "reverse_expansions": reverse_expansions,
        }
        if not math.isfinite(G.c_best):
            return PlanResult(
                None, self.name, iterations=edges_processed, tree=G.tree_edges(), info=info
            )
        info["raw_length"] = G.c_best
        return PlanResult(
            resample_path(G.path(), 0.05),
            self.name,
            iterations=edges_processed,
            first_solution_time=first_time,
            tree=G.tree_edges(),
            info=info,
        )
