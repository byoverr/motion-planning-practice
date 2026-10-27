"""Regulated Pure Pursuit (Macenski et al., 2023), Nav2's path-tracking controller.

Pure pursuit steers along the arc through a look-ahead point on the path:
curvature = 2 * y / L^2, where y is the lateral offset of the point in the robot frame.
The "regulated" part scales the speed down on tight curves and near obstacles, and an
adaptive look-ahead grows with speed.

It does **not** plan around obstacles: if the arc ahead collides, it stops and reports
"blocked", and the navigator has to replan the global path. That is exactly why it is in
the comparison.
"""

from __future__ import annotations

import numpy as np

from mpp.geometry import rollout_unicycle, to_robot_frame
from mpp.mapping.costmap import Costmap
from mpp.planning.local_planners.base import LocalPlan, LocalPlanner
from mpp.robot.model import RobotParams


class RegulatedPurePursuit(LocalPlanner):
    name = "rpp"
    avoids_obstacles = False

    def __init__(
        self,
        robot: RobotParams,
        dt: float = 0.1,
        lookahead_time: float = 1.5,
        min_lookahead: float = 0.25,
        max_lookahead: float = 0.6,
        rotate_to_heading_angle: float = 0.8,
        rotate_speed: float = 1.2,
        regulated_min_radius: float = 0.6,
        proximity_distance: float = 0.35,
        proximity_gain: float = 1.0,
        min_speed: float = 0.04,
        approach_distance: float = 0.4,
        collision_horizon: float = 1.0,
    ) -> None:
        super().__init__(robot, dt)
        self.lookahead_time = lookahead_time
        self.min_la, self.max_la = min_lookahead, max_lookahead
        self.rotate_angle = rotate_to_heading_angle
        self.rotate_speed = rotate_speed
        self.r_min = regulated_min_radius
        self.prox_dist = proximity_distance
        self.prox_gain = proximity_gain
        self.min_speed = min_speed
        self.approach = approach_distance
        self.collision_steps = int(round(collision_horizon / dt))

    def compute(self, pose, vel, costmap: Costmap) -> LocalPlan:
        r = self.robot
        self.update_progress(pose)
        la = float(np.clip(abs(vel[0]) * self.lookahead_time, self.min_la, self.max_la))
        target = self.point_at(la)
        local = to_robot_frame(target[None], pose)[0]
        angle = float(np.arctan2(local[1], local[0]))

        if abs(angle) > self.rotate_angle:
            v, w = 0.0, float(np.sign(angle)) * self.rotate_speed
        else:
            dist2 = max(float(local @ local), 1e-6)
            curvature = 2.0 * local[1] / dist2
            v = r.max_v
            # regulation 1: slow down on tight curves
            radius = 1.0 / max(abs(curvature), 1e-6)
            if radius < self.r_min:
                v *= radius / self.r_min
            # regulation 2: slow down close to obstacles
            clearance = float(costmap.clearance_at(np.array(pose[:2])))
            if clearance < self.prox_dist:
                v *= self.prox_gain * clearance / self.prox_dist
            # regulation 3: slow down when approaching the goal
            rem = self.remaining()
            if rem < self.approach:
                v *= max(rem / self.approach, 0.2)
            v = max(v, self.min_speed)
            w = v * curvature

        v, w = self.clamp(v, w, vel)
        # collision check of the commanded arc (turning in place is always safe for a disc)
        steps = self.collision_steps
        arc = rollout_unicycle(
            pose, np.full((1, steps), max(v, 0.15)), np.full((1, steps), w), self.dt
        )[0]
        if v > 1e-3 and (costmap.clearance_at(arc[:, :2]) < r.radius + 0.01).any():
            return LocalPlan(0.0, 0.0, "blocked", arc)
        return LocalPlan(v, w, "ok", arc)
