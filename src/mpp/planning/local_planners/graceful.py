"""Graceful Controller, the newest stock controller of Nav2 (nav2_graceful_controller).

It is built on the smooth control law of Park & Kuipers (ICRA 2011): the robot drives to a
*target pose* (a point on the path plus the path direction there) along a curvature computed
in egocentric polar coordinates (r, phi, delta):

    kappa = -1/r * ( k_delta * (delta - atan(-k_phi * phi))
                     + (1 + k_phi / (1 + (k_phi*phi)^2)) * sin(delta) )
    v     = v_max / (1 + beta * |kappa|^lambda),   w = kappa * v

The controller tries target poses from `max_lookahead` back towards the robot, forward
simulates the control law to each of them and takes the first one whose simulated
trajectory is collision-free. If the robot still has to turn a lot it first rotates in place.
Like pure pursuit it does not plan around obstacles: if no target is reachable it reports
"blocked" and the navigator replans. Ported from the Nav2 sources (Apache-2.0).
"""

from __future__ import annotations

import math

import numpy as np

from mpp.geometry import wrap_angle
from mpp.mapping.costmap import Costmap
from mpp.planning.local_planners.base import LocalPlan, LocalPlanner
from mpp.robot.model import RobotParams


class GracefulController(LocalPlanner):
    name = "graceful"
    avoids_obstacles = False

    def __init__(
        self,
        robot: RobotParams,
        dt: float = 0.1,
        min_lookahead: float = 0.25,
        max_lookahead: float = 1.0,
        k_phi: float = 2.0,
        k_delta: float = 1.0,
        beta: float = 0.4,
        lam: float = 2.0,
        v_linear_min: float = 0.05,
        v_angular_max: float = 1.0,
        v_angular_min_in_place: float = 0.25,
        slowdown_radius: float = 1.0,
        deceleration_max: float = 2.5,
        initial_rotation_tolerance: float = 0.75,
        rotation_scaling_factor: float = 0.5,
        target_step: float = 0.1,
        safety: float = 0.01,
    ) -> None:
        super().__init__(robot, dt)
        self.min_la, self.max_la = min_lookahead, max_lookahead
        self.k_phi, self.k_delta, self.beta, self.lam = k_phi, k_delta, beta, lam
        self.v_min, self.v_max = v_linear_min, robot.max_v
        self.w_max, self.w_min_in_place = min(v_angular_max, robot.max_w), v_angular_min_in_place
        self.slowdown_radius = slowdown_radius
        self.dec_max = deceleration_max
        self.rot_tol = initial_rotation_tolerance
        self.rot_scale = rotation_scaling_factor
        self.target_step = target_step
        self.safety = safety
        self.do_initial_rotation = True

    def reset(self) -> None:
        self.do_initial_rotation = True

    # ------------------------------------------------------------------ control law

    def curvature(self, r: float, phi: float, delta: float) -> float:
        prop = self.k_delta * (delta - math.atan(-self.k_phi * phi))
        feedback = (1.0 + self.k_phi / (1.0 + (self.k_phi * phi) ** 2)) * math.sin(delta)
        return -1.0 / max(r, 1e-6) * (prop + feedback)

    def control(self, target, current) -> tuple[float, float]:
        """Velocity towards `target` (x, y, yaw) from `current` (x, y, yaw)."""
        dx, dy = target[0] - current[0], target[1] - current[1]
        los = math.atan2(-dy, dx)
        r = math.hypot(dx, dy)
        phi = float(wrap_angle(target[2] + los))
        delta = float(wrap_angle(current[2] + los))
        kappa = self.curvature(r, phi, delta)
        v = self.v_max / (1.0 + self.beta * abs(kappa) ** self.lam)
        v = min(self.v_max * r / self.slowdown_radius, v)
        v = min(math.sqrt(2.0 * r * self.dec_max), v)
        v = float(np.clip(v, self.v_min, self.v_max))
        w = float(np.clip(kappa * v, -self.w_max, self.w_max))
        if kappa != 0.0:
            v = w / kappa
        return v, w

    def rotate(self, angle: float) -> tuple[float, float]:
        w = self.rot_scale * angle * self.w_max
        return 0.0, math.copysign(max(abs(w), self.w_min_in_place), w)

    # ------------------------------------------------------------------ target validation

    def simulate(self, target, pose, costmap: Costmap) -> tuple[float, float] | None:
        """Forward-simulate the control law to `target`; first command if collision-free."""
        res = costmap.res
        dt = res / self.v_max
        x, y, yaw = pose
        angle = math.atan2(target[1] - y, target[0] - x) - yaw
        rotating = self.do_initial_rotation and abs(wrap_angle(angle)) >= self.rot_tol
        max_iter = int(3 * math.hypot(target[0] - x, target[1] - y) / res) + 1
        cmd = None
        threshold = self.robot.radius + self.safety
        for _ in range(max_iter):
            if rotating:
                err = float(wrap_angle(math.atan2(target[1] - y, target[0] - x) - yaw))
                v, w = self.rotate(err)
                if abs(err) < self.rot_tol:
                    rotating = False
            else:
                v, w = self.control(target, (x, y, yaw))
            if cmd is None:
                cmd = (v, w)
            x, y, yaw = x + v * dt * math.cos(yaw), y + v * dt * math.sin(yaw), yaw + w * dt
            if float(costmap.clearance_at(np.array([x, y]))) < threshold:
                return None
            if math.hypot(target[0] - x, target[1] - y) <= res:
                break
        return cmd

    def compute(self, pose, vel, costmap: Costmap) -> LocalPlan:
        self.update_progress(pose)
        remaining = self.remaining()
        x, y, yaw = pose
        angle0 = math.atan2(*(self.point_at(self.max_la) - np.array([x, y]))[::-1]) - yaw
        if abs(wrap_angle(angle0)) < self.rot_tol:
            self.do_initial_rotation = False
        ahead = min(self.max_la, remaining)
        while ahead > 1e-6:
            if remaining >= self.max_la and ahead < self.min_la:
                break
            p = self.point_at(ahead)
            target = (p[0], p[1], self.tangent_at(ahead))
            cmd = self.simulate(target, pose, costmap)
            if cmd is not None:
                v, w = self.clamp(*cmd, vel)
                return LocalPlan(v, w, "ok", None)
            ahead -= self.target_step
        return LocalPlan(0.0, 0.0, "blocked")
