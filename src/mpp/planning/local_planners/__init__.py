"""Local planners (controllers): follow the global path and react to what the lidar sees."""

from __future__ import annotations

from mpp.planning.local_planners.base import LocalPlan, LocalPlanner
from mpp.planning.local_planners.dwa import DWA
from mpp.planning.local_planners.mppi import MPPI
from mpp.planning.local_planners.rpp import RegulatedPurePursuit
from mpp.robot.model import BURGER, RobotParams

LOCAL_PLANNERS: dict[str, type[LocalPlanner]] = {
    "dwa": DWA,
    "mppi": MPPI,
    "rpp": RegulatedPurePursuit,
}


def make_local_planner(name: str, robot: RobotParams = BURGER, **kwargs) -> LocalPlanner:
    try:
        cls = LOCAL_PLANNERS[name]
    except KeyError:
        raise KeyError(
            f"unknown local planner {name!r}; known: {', '.join(LOCAL_PLANNERS)}"
        ) from None
    return cls(robot, **kwargs)


__all__ = ["LOCAL_PLANNERS", "LocalPlan", "LocalPlanner", "make_local_planner"]
