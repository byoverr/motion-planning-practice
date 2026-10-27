"""DWA, the Dynamic Window Approach (Fox, Burgard & Thrun, 1997), scored with DWB-style critics.

Every control cycle:

1. the *dynamic window* is the set of target (v, w) reachable within ``window`` seconds
   given the acceleration limits;
2. for each target the robot is rolled out for ``horizon`` seconds, accelerating towards it;
3. trajectories that hit an obstacle are discarded, the rest are scored by critics
   (stay on the path, reach the look-ahead point, align with the path, keep clear of
   obstacles, go fast);
4. the cheapest trajectory's first command is sent to the robot.
"""

from __future__ import annotations

import numpy as np

from mpp.geometry import rollout_unicycle, wrap_angle
from mpp.mapping.costmap import Costmap
from mpp.planning.local_planners.base import LocalPlan, LocalPlanner
from mpp.robot.model import RobotParams


class DWA(LocalPlanner):
    name = "dwa"
    avoids_obstacles = True

    def __init__(
        self,
        robot: RobotParams,
        dt: float = 0.1,
        horizon: float = 1.7,
        window: float = 1.0,
        nv: int = 7,
        nw: int = 25,
        lookahead: float = 0.9,
        w_path: float = 3.0,
        w_goal: float = 4.0,
        w_heading: float = 0.6,
        w_obstacle: float = 0.5,
        w_speed: float = 0.6,
        safety: float = 0.03,
        blocked_margin: float = 0.4,
    ) -> None:
        super().__init__(robot, dt)
        self.safety = safety
        self.blocked_margin = blocked_margin
        self.steps = int(round(horizon / dt))
        self.window = window
        self.nv, self.nw = nv, nw
        self.lookahead = lookahead
        self.w = dict(
            path=w_path, goal=w_goal, heading=w_heading, obstacle=w_obstacle, speed=w_speed
        )

    def compute(self, pose, vel, costmap: Costmap) -> LocalPlan:
        r = self.robot
        self.update_progress(pose)
        v0, w0 = vel
        # window of target velocities reachable within `window` seconds (DWB samples targets
        # and accelerates towards them during the rollout instead of using one control cycle)
        tw = self.window
        vs = np.linspace(max(0.0, v0 - r.acc_v * tw), min(r.max_v, v0 + r.acc_v * tw), self.nv)
        ws = np.linspace(max(-r.max_w, w0 - r.acc_w * tw), min(r.max_w, w0 + r.acc_w * tw), self.nw)
        Vt, Wt = (a.ravel() for a in np.meshgrid(vs, ws))
        t = np.arange(1, self.steps + 1) * self.dt
        V = v0 + np.clip(Vt[:, None] - v0, -r.acc_v * t, r.acc_v * t)
        W = w0 + np.clip(Wt[:, None] - w0, -r.acc_w * t, r.acc_w * t)
        traj = rollout_unicycle(pose, V, W, self.dt)
        xy = traj[..., :2]

        clear = costmap.clearance_at(xy)
        valid = clear.min(axis=1) > self.collision_threshold(costmap, pose, self.safety)
        if not valid.any():
            return LocalPlan(0.0, 0.0, "no_valid_trajectory", None, xy[::4])

        seg, tangent = self.free_segment(costmap, self.lookahead + 0.6, self.blocked_margin)
        goal = self.local_goal(costmap, self.lookahead)
        end, end_yaw = xy[:, -1], traj[:, -1, 2]

        if len(seg):
            d2 = ((end[:, None, :] - seg[None, :, :]) ** 2).sum(-1)
            j = d2.argmin(axis=1)
            path_cost = np.sqrt(d2[np.arange(len(end)), j])
            heading_cost = np.abs(wrap_angle(tangent[j] - end_yaw)) / np.pi
        else:
            path_cost = np.zeros(len(end))
            to_goal = np.arctan2(goal[1] - end[:, 1], goal[0] - end[:, 0])
            heading_cost = np.abs(wrap_angle(to_goal - end_yaw)) / np.pi
        goal_cost = np.linalg.norm(end - goal, axis=1)
        obstacle_cost = costmap.cost_at(xy).mean(axis=1)
        speed_cost = min(1.0, self.remaining() / 0.6) * (r.max_v - Vt) / r.max_v

        w = self.w
        total = (
            w["path"] * path_cost
            + w["goal"] * goal_cost
            + w["heading"] * heading_cost
            + w["obstacle"] * obstacle_cost
            + w["speed"] * speed_cost
        )
        total[~valid] = np.inf
        k = int(np.argmin(total))
        return LocalPlan(float(V[k, 0]), float(W[k, 0]), "ok", traj[k], xy[valid][::3])
