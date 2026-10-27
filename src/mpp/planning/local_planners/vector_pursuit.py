"""Vector Pursuit (Wit, 2000), as in the Nav2 community plugin vector_pursuit_controller
by Black Coffee Robotics (Apache-2.0).

Pure pursuit only uses the *position* of the look-ahead point. Vector pursuit also uses the
path's *orientation* there: with screw theory it combines "rotate onto the circle through
the point" and "arrive with the path heading" into one turning radius

    phi = angle of the screw to the point,     R = | k*phi / ((k-1)*phi + theta) * d^2/(2y) |

where theta is the path heading at the look-ahead point in the robot frame and k weighs
position against orientation. Speed is limited by lateral acceleration (v = sqrt(a_lat*R)),
proximity to obstacles and the approach to the goal. It does not avoid obstacles: a
collision ahead stops it and the navigator replans.
"""

from __future__ import annotations

import math

import numpy as np

from mpp.geometry import to_robot_frame, wrap_angle
from mpp.mapping.costmap import Costmap
from mpp.planning.local_planners.base import LocalPlan, LocalPlanner
from mpp.robot.model import RobotParams


class VectorPursuit(LocalPlanner):
    name = "vector_pursuit"
    avoids_obstacles = False

    def __init__(
        self,
        robot: RobotParams,
        dt: float = 0.1,
        k: float = 8.0,
        min_turning_radius: float = 0.25,
        lookahead_time: float = 1.5,
        min_lookahead: float = 0.3,
        max_lookahead: float = 0.9,
        rotate_to_heading_min_angle: float = 0.785,
        rotate_to_heading_angular_vel: float = 1.8,
        max_lateral_accel: float = 0.5,
        cost_scaling_dist: float = 0.35,  # 0.6 upstream, scaled to the small TurtleBot3
        cost_scaling_gain: float = 1.0,
        approach_velocity_scaling_dist: float = 1.0,
        min_approach_velocity: float = 0.05,
        min_linear_velocity: float = 0.05,
    ) -> None:
        super().__init__(robot, dt)
        self.k = k
        self.r_min = min_turning_radius
        self.la_time, self.min_la, self.max_la = lookahead_time, min_lookahead, max_lookahead
        self.rot_angle = rotate_to_heading_min_angle
        self.rot_w = min(rotate_to_heading_angular_vel, robot.max_w)
        self.a_lat = max_lateral_accel
        self.cost_dist, self.cost_gain = cost_scaling_dist, cost_scaling_gain
        self.approach_dist, self.v_approach_min = (
            approach_velocity_scaling_dist,
            min_approach_velocity,
        )
        self.v_min = min_linear_velocity
        self.last_v = 0.0

    def reset(self) -> None:
        self.last_v = 0.0

    def turning_radius(self, x: float, y: float, theta: float) -> float:
        """Screw-theory turning radius to the point (x, y) with path heading theta (robot frame)."""
        if x < 0.0:
            return self.r_min
        if abs(y) <= 1e-6:
            return math.inf
        d2 = x * x + y * y
        phi_1 = math.atan2(2 * y * y - d2, 2 * x * y)
        term_2 = d2 / (2 * y)
        phi_2 = math.atan2(term_2, 0.0)
        phi = max((phi_1 - phi_2) % (2 * math.pi), 1e-9)
        term_1 = self.k * phi / ((self.k - 1) * phi + theta)
        return max(abs(term_1 * term_2), self.r_min)

    def compute(self, pose, vel, costmap: Costmap) -> LocalPlan:
        r = self.robot
        self.update_progress(pose)
        la = float(np.clip(abs(self.last_v) * self.la_time, self.min_la, self.max_la))
        target = self.point_at(la)
        local = to_robot_frame(target[None], pose)[0]
        theta = float(wrap_angle(self.tangent_at(la) - pose[2]))
        angle = math.atan2(local[1], local[0])

        if abs(angle) > self.rot_angle:
            v, w = 0.0, math.copysign(self.rot_w, angle)
        else:
            radius = self.turning_radius(local[0], local[1], theta)
            v = r.max_v
            clearance = float(costmap.clearance_at(np.array(pose[:2])))
            v_cost = (
                v * self.cost_gain * clearance / self.cost_dist if clearance < self.cost_dist else v
            )
            v_curve = math.sqrt(self.a_lat * radius) if math.isfinite(radius) else v
            v = min(v, v_curve, v_cost, abs(vel[0]) + r.acc_v * self.dt)
            if self.remaining() < self.approach_dist:
                d_goal = float(np.linalg.norm(self.path[-1] - np.array(pose[:2])))
                v = min(v, max(r.max_v * d_goal / self.approach_dist, self.v_approach_min))
            v = max(v, self.v_min)
            w = v / radius if math.isfinite(radius) else 0.0
            if local[1] < 0:
                w = -w
        v, w = self.clamp(v, w, vel)
        if self.arc_blocked(costmap, pose, max(v, 0.15), w, float(np.hypot(*local))):
            self.last_v = 0.0
            return LocalPlan(0.0, 0.0, "blocked")
        self.last_v = v
        return LocalPlan(v, w, "ok")
