"""Optional reference implementations from OMPL (``uv sync --extra ompl``).

Use them as ``ompl:RRTConnect``, ``ompl:BITstar``, ``ompl:RRTstar``, ``ompl:PRMstar`` ... to
compare our Python versions with the C++ originals on the same costmap. Note that the Python
bindings of OMPL 2.0 do not expose AIT*/EIT* yet, so AIT* has no OMPL counterpart here.
"""

from __future__ import annotations

import numpy as np

from mpp.geometry import resample_path
from mpp.mapping.costmap import Costmap
from mpp.planning.global_planners.base import (
    ClearanceChecker,
    GlobalPlanner,
    PlanResult,
    shortcut_path,
)

_seeded = False  # OMPL accepts a seed only once per process


class OMPLPlanner(GlobalPlanner):
    def __init__(
        self,
        planner: str,
        time_limit: float = 1.0,
        clearance_margin: float = 0.04,
        simplify: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed)
        try:
            import ompl.base as ob
            import ompl.geometric as og
        except ImportError as e:  # pragma: no cover - depends on the optional extra
            raise ImportError("OMPL is not installed: run `uv sync --extra ompl`") from e
        if not hasattr(og, planner):
            known = [n for n in dir(og) if n[0].isupper() and "star" in n or "RRT" in n]
            raise KeyError(f"OMPL has no planner {planner!r}; try one of {sorted(known)}")
        self.ob, self.og = ob, og
        self.planner = planner
        self.name = f"ompl:{planner}"
        self.time_limit = time_limit
        self.margin = clearance_margin
        self.simplify = simplify

    def _plan(self, costmap: Costmap, start: np.ndarray, goal: np.ndarray) -> PlanResult:
        ob, og = self.ob, self.og
        global _seeded
        from ompl import util as ou

        ou.setLogLevel(ou.LogLevel.LOG_WARN)
        if self.seed is not None and not _seeded:
            ou.RNG.setSeed(int(self.seed) + 1)
            _seeded = True
        space = ob.RealVectorStateSpace(2)
        bounds = ob.RealVectorBounds(2)
        x0, x1, y0, y1 = costmap.extent
        bounds.setLow(0, x0)
        bounds.setHigh(0, x1)
        bounds.setLow(1, y0)
        bounds.setHigh(1, y1)
        space.setBounds(bounds)
        ss = og.SimpleSetup(space)
        checker = ClearanceChecker(costmap, self.margin, start, goal)

        def valid(state) -> bool:
            return bool(checker.points_free(np.array([state[0], state[1]]))[0])

        ss.setStateValidityChecker(valid)
        si = ss.getSpaceInformation()
        si.setStateValidityCheckingResolution(0.25 * costmap.res / max(x1 - x0, y1 - y0))
        s, g = space.allocState(), space.allocState()
        s[0], s[1] = float(start[0]), float(start[1])
        g[0], g[1] = float(goal[0]), float(goal[1])
        ss.setStartAndGoalStates(s, g)
        ss.getProblemDefinition().setOptimizationObjective(ob.PathLengthOptimizationObjective(si))
        ss.setPlanner(getattr(og, self.planner)(si))
        status = ss.solve(self.time_limit)
        if not ss.haveExactSolutionPath():
            return PlanResult(None, self.name, info={"status": str(status)})
        sol = ss.getSolutionPath()
        path = np.array([(st[0], st[1]) for st in sol.getStates()])
        raw = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        if self.simplify:
            path = shortcut_path(path, costmap, self.rng, checker=checker)
        return PlanResult(resample_path(path, 0.05), self.name, info={"raw_length": raw})
