"""MPPI, Model Predictive Path Integral control (Williams et al., 2016), the default
controller of Nav2 since Iron.

Instead of a grid of constant commands (DWA), MPPI perturbs a whole *sequence* of commands
over the horizon with Gaussian noise, rolls out hundreds of such sequences, scores them and
takes a softmax-weighted average:

    U <- sum_k  exp(-(S_k - S_min) / lambda) * U_k  /  sum_k exp(...)

The result is warm-started on the next cycle (shifted by one step). Because the sequences can
curve, MPPI finds smooth manoeuvres around obstacles that a constant-command DWA misses.
"""

from __future__ import annotations

import numpy as np

from mpp.geometry import rollout_unicycle, wrap_angle
from mpp.mapping.costmap import Costmap
from mpp.planning.local_planners.base import LocalPlan, LocalPlanner, distance_to_points
from mpp.robot.model import RobotParams


class MPPI(LocalPlanner):
    name = "mppi"
    avoids_obstacles = True

    def __init__(
        self,
        robot: RobotParams,
        dt: float = 0.1,
        horizon: float = 2.5,
        samples: int = 400,
        std_v: float = 0.05,
        std_w: float = 0.4,
        temperature: float = 0.1,
        lookahead: float = 1.0,
        w_collision: float = 1000.0,
        w_obstacle: float = 2.0,
        w_path: float = 3.0,
        w_goal: float = 5.0,
        w_heading: float = 0.5,
        w_reverse: float = 4.0,
        w_smooth: float = 0.01,
        w_speed: float = 4.0,
        safety: float = 0.02,
        seed: int | None = 0,
    ) -> None:
        super().__init__(robot, dt)
        self.safety = safety
        self.T = int(round(horizon / dt))
        self.K = samples
        self.std = np.array([std_v, std_w])
        self.lam = temperature
        self.lookahead = lookahead
        self.w = dict(
            collision=w_collision,
            obstacle=w_obstacle,
            path=w_path,
            goal=w_goal,
            heading=w_heading,
            reverse=w_reverse,
            smooth=w_smooth,
            speed=w_speed,
        )
        self.rng = np.random.default_rng(seed)
        self.U = np.zeros((self.T, 2))

    def reset(self) -> None:
        self.U = np.zeros((self.T, 2))

    def compute(self, pose, vel, costmap: Costmap) -> LocalPlan:
        r = self.robot
        self.update_progress(pose)
        # warm start: the previous optimum shifted by one step
        self.U = np.vstack([self.U[1:], self.U[-1:]])
        noise = self.rng.normal(size=(self.K, self.T, 2)) * self.std
        noise[0] = 0.0  # keep the unperturbed previous solution as one of the candidates
        U = self.U[None] + noise
        U[..., 0] = np.clip(U[..., 0], r.min_v, r.max_v)
        U[..., 1] = np.clip(U[..., 1], -r.max_w, r.max_w)
        traj = rollout_unicycle(pose, U[..., 0], U[..., 1], self.dt)
        xy = traj[..., :2]

        clear = costmap.clearance_at(xy)
        collide = clear < self.collision_threshold(costmap, pose, self.safety)
        # only the part of a rollout before its first collision is a valid prediction
        hit = collide.any(axis=1)

        seg_free, _ = self.free_segment(costmap, self.lookahead + 1.0)
        goal = self.local_goal(costmap, self.lookahead)

        w = self.w
        S = w["collision"] * hit
        S = S + w["obstacle"] * costmap.cost_at(xy).mean(axis=1)
        if len(seg_free):
            S = S + w["path"] * distance_to_points(xy[:, ::3], seg_free[::2]).mean(axis=1)
        S = S + w["goal"] * np.linalg.norm(xy[:, -1] - goal, axis=1)
        to_goal = np.arctan2(goal[1] - xy[:, -1, 1], goal[0] - xy[:, -1, 0])
        S = S + w["heading"] * np.abs(wrap_angle(to_goal - traj[:, -1, 2])) / np.pi
        S = S + w["reverse"] * np.clip(-U[..., 0], 0, None).mean(axis=1)
        # "go fast", faded out close to the end of the path so the robot can stop at the goal
        fade = min(1.0, self.remaining() / 0.6)
        S = S + fade * w["speed"] * (r.max_v - U[..., 0]).mean(axis=1) / r.max_v
        dU = np.diff(np.concatenate([np.broadcast_to(np.array(vel), (self.K, 1, 2)), U], 1), axis=1)
        S = S + w["smooth"] * (dU**2).sum(axis=(1, 2))

        weights = np.exp(-(S - S.min()) / self.lam)
        weights /= weights.sum()
        self.U = np.einsum("k,ktc->tc", weights, U)

        best = rollout_unicycle(pose, self.U[None, :, 0], self.U[None, :, 1], self.dt)[0]
        status = "no_valid_trajectory" if hit.all() else "ok"
        if status != "ok":
            self.U[:] = 0.0
            return LocalPlan(0.0, 0.0, status, None, xy[::20])
        v, wz = self.clamp(self.U[0, 0], self.U[0, 1], vel)
        return LocalPlan(v, wz, status, best, xy[np.argsort(S)[:40]])
