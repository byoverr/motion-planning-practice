"""Global planners: find a collision-free path from start to goal on the costmap."""

from __future__ import annotations

from mpp.planning.global_planners.ait_star import AITStar
from mpp.planning.global_planners.astar import AStar
from mpp.planning.global_planners.base import GlobalPlanner, PlanResult
from mpp.planning.global_planners.bit_star import BITStar
from mpp.planning.global_planners.rrt_connect import RRTConnect

GLOBAL_PLANNERS: dict[str, type[GlobalPlanner]] = {
    "astar": AStar,
    "rrt_connect": RRTConnect,
    "bit_star": BITStar,
    "ait_star": AITStar,
}


def make_global_planner(name: str, **kwargs) -> GlobalPlanner:
    """Create a planner by name. Names starting with ``ompl:`` (e.g. ``ompl:BITstar``) use the
    optional OMPL bindings as a reference implementation."""
    if name.startswith("ompl:"):
        from mpp.planning.global_planners.ompl_planner import OMPLPlanner

        return OMPLPlanner(name.split(":", 1)[1], **kwargs)
    if name == "dijkstra":
        return AStar(heuristic_weight=0.0, **kwargs)
    try:
        cls = GLOBAL_PLANNERS[name]
    except KeyError:
        known = ", ".join([*GLOBAL_PLANNERS, "dijkstra", "ompl:<Planner>"])
        raise KeyError(f"unknown global planner {name!r}; known: {known}") from None
    return cls(**kwargs)


__all__ = ["GLOBAL_PLANNERS", "GlobalPlanner", "PlanResult", "make_global_planner"]
