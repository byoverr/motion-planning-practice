"""Thin wrapper around MuJoCo: drive the TurtleBot3, read its pose, scan with the lidar, drop boxes.

The navigation stack only talks to this class, so swapping MuJoCo for another simulator
(or a real robot) means re-implementing these few methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from mpp.robot.model import BURGER, D435, DepthCameraParams, RobotParams
from mpp.sim.scene import GROUP_BOX, GROUP_WORLD, build_model
from mpp.world.layout import Layout, Pose2D
from mpp.world.scenarios import Scenario


@dataclass
class Scan:
    origin: np.ndarray  # (2,) sensor position in the world
    angles: np.ndarray  # (N,) world-frame ray directions
    ranges: np.ndarray  # (N,) metres; inf = nothing within range, nan = invalid (too close)
    max_range: float


@dataclass
class ContactStats:
    in_contact: bool = False
    events: int = 0
    time: float = 0.0
    with_: set[str] = field(default_factory=set)


class Simulation:
    def __init__(
        self,
        layout: Layout,
        scenario: Scenario | None,
        start: Pose2D,
        robot: RobotParams = BURGER,
        lidar_noise: float = 0.0,
        seed: int = 0,
        sensor: str = "lidar",
        camera: DepthCameraParams = D435,
    ) -> None:
        if sensor not in ("lidar", "depth"):
            raise ValueError(f"unknown sensor {sensor!r}; use 'lidar' or 'depth'")
        self.layout = layout
        self.scenario = scenario
        self.robot = robot
        self.sensor = sensor
        self.camera = camera
        self.model = build_model(layout, scenario, start, sensor)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.base_id = m.body("base_footprint").id
        self.scan_body_id = m.body("base_scan").id
        self.lidar_site = m.site("lidar").id
        self.act_left = m.actuator("wheel_left").id
        self.act_right = m.actuator("wheel_right").id
        self.base_qadr = m.jnt_qposadr[m.joint("base").id]
        self.base_vadr = m.jnt_dofadr[m.joint("base").id]
        robot_bodies = {self.base_id} | {
            b for b in range(m.nbody) if m.body_rootid[b] == self.base_id
        }
        self.robot_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in robot_bodies}
        self.floor_geom = m.geom("floor").id
        self.boxes = {b.name: b for b in (scenario.boxes if scenario else ())}
        self.box_body = {name: m.body(name).id for name in self.boxes}
        self.dropped: dict[str, float] = {}
        self.contacts = ContactStats()
        self.lidar_noise = lidar_noise
        self.rng = np.random.default_rng(seed)
        n = robot.lidar_rays
        self._ray_angles = np.arange(n) * (2 * np.pi / n)
        self._ray_vec = np.zeros(3 * n)
        self._ray_geom = np.zeros(n, dtype=np.int32)
        self._ray_dist = np.zeros(n)
        self._groups = np.zeros(6, dtype=np.uint8)
        self._groups[GROUP_WORLD] = 1
        self._groups[GROUP_BOX] = 1
        if sensor == "depth":
            self._init_depth_camera()
        self.last_cloud = np.empty((0, 3))  # obstacle points of the last depth frame (world)
        self.last_depth: np.ndarray | None = None  # z-depth image of the last frame (rows, cols)
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------ state

    @property
    def time(self) -> float:
        return float(self.data.time)

    def pose(self) -> tuple[float, float, float]:
        q = self.data.qpos[self.base_qadr : self.base_qadr + 7]
        w, x, y, z = q[3:7]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return float(q[0]), float(q[1]), float(yaw)

    def velocity(self) -> tuple[float, float]:
        """Forward speed and yaw rate of the base."""
        qv = self.data.qvel[self.base_vadr : self.base_vadr + 6]
        yaw = self.pose()[2]
        v = qv[0] * np.cos(yaw) + qv[1] * np.sin(yaw)  # free joint: linear part in world frame
        return float(v), float(qv[5])  # angular part is in the body frame (z stays up)

    def box_positions(self) -> dict[str, np.ndarray]:
        return {n: self.data.xpos[b][:2].copy() for n, b in self.box_body.items()}

    # ------------------------------------------------------------------ control

    def set_cmd(self, v: float, w: float) -> None:
        r = self.robot
        v = float(np.clip(v, r.min_v, r.max_v))
        w = float(np.clip(w, -r.max_w, r.max_w))
        wl, wr = r.wheel_speeds(v, w)
        self.data.ctrl[self.act_left] = wl
        self.data.ctrl[self.act_right] = wr

    def step(self, duration: float) -> None:
        n = max(int(round(duration / self.model.opt.timestep)), 1)
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            self._track_contacts()
        self._update_boxes()

    def _track_contacts(self) -> None:
        d, m = self.data, self.model
        touching = False
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            r1, r2 = g1 in self.robot_geoms, g2 in self.robot_geoms
            if r1 == r2:
                continue
            other = g2 if r1 else g1
            if other == self.floor_geom:
                continue
            touching = True
            self.contacts.with_.add(m.geom(other).name)
        if touching:
            self.contacts.time += m.opt.timestep
            if not self.contacts.in_contact:
                self.contacts.events += 1
        self.contacts.in_contact = touching

    def _update_boxes(self) -> None:
        x, y, _ = self.pose()
        for name, box in self.boxes.items():
            if name in self.dropped:
                continue
            if np.hypot(box.x - x, box.y - y) <= box.trigger_distance:
                self.model.body_gravcomp[self.box_body[name]] = 0.0
                self.dropped[name] = self.time

    # ------------------------------------------------------------------ sensing

    def scan(self) -> Scan:
        """One reading of the robot's range sensor as a planar scan for the costmap."""
        return self._depth_scan() if self.sensor == "depth" else self._lidar_scan()

    def _lidar_scan(self) -> Scan:
        """360-degree planar lidar (LDS-01 style) via MuJoCo ray casting."""
        m, d, r = self.model, self.data, self.robot
        origin = d.site_xpos[self.lidar_site].copy()
        yaw = self.pose()[2]
        ang = yaw + self._ray_angles
        n = len(ang)
        vec = self._ray_vec.reshape(n, 3)
        vec[:, 0] = np.cos(ang)
        vec[:, 1] = np.sin(ang)
        vec[:, 2] = 0.0
        rmin, rmax = r.lidar_range
        mujoco.mj_multiRay(
            m,
            d,
            origin,
            self._ray_vec,
            self._groups,
            1,
            self.scan_body_id,
            self._ray_geom,
            self._ray_dist,
            None,
            n,
            rmax,
        )
        dist = self._ray_dist.copy()
        if self.lidar_noise > 0:
            dist += self.rng.normal(0.0, self.lidar_noise, n)
        ranges = np.where((self._ray_geom < 0) | (dist > rmax), np.inf, dist)
        ranges = np.where(ranges < rmin, np.nan, ranges)
        return Scan(origin=origin[:2], angles=ang, ranges=ranges, max_range=rmax)

    # ------------------------------------------------------------------ depth camera

    def _init_depth_camera(self) -> None:
        c = self.camera
        self.cam_site = self.model.site("depth_camera").id
        # pinhole model: rays through a regular grid on the image plane (x forward, y left, z up)
        u = np.linspace(
            np.tan(np.radians(c.hfov_deg / 2)), -np.tan(np.radians(c.hfov_deg / 2)), c.cols
        )
        v = np.linspace(
            np.tan(np.radians(c.vfov_deg / 2)), -np.tan(np.radians(c.vfov_deg / 2)), c.rows
        )
        uu, vv = np.meshgrid(u, v)  # (rows, cols), row 0 is the top of the image
        d = np.stack([np.ones_like(uu), uu, vv], axis=-1)
        self._cam_dirs = d / np.linalg.norm(d, axis=-1, keepdims=True)  # unit rays, camera frame
        self._cam_col_angle = np.arctan(u)  # horizontal angle of every image column
        n = c.rows * c.cols
        self._cam_vec = np.zeros(3 * n)
        self._cam_geom = np.zeros(n, dtype=np.int32)
        self._cam_dist = np.zeros(n)

    def _depth_scan(self) -> Scan:
        """RGB-D camera turned into a "virtual laser scan", the way TurtleBot 2 and many
        camera-only robots feed a 2D costmap (depthimage_to_laserscan / pointcloud_to_laserscan):

        1. cast one ray per (decimated) depth pixel and add stereo noise to the z-depth;
        2. drop pixels outside the valid range and points outside the obstacle height band
           (the floor, things above the robot);
        3. per image column keep the closest obstacle point -> one beam of the scan.
        Columns that see nothing clear free space up to the camera range; a column with a
        reading closer than the minimum range is unknown and is skipped.
        """
        c, m, d = self.camera, self.model, self.data
        origin = d.site_xpos[self.cam_site].copy()
        rot = d.site_xmat[self.cam_site].reshape(3, 3)
        dirs = self._cam_dirs.reshape(-1, 3)
        world = dirs @ rot.T
        self._cam_vec[:] = world.ravel()
        n = len(dirs)
        mujoco.mj_multiRay(
            m,
            d,
            origin,
            self._cam_vec,
            self._groups,
            1,
            -1,
            self._cam_geom,
            self._cam_dist,
            None,
            n,
            c.max_range / dirs[:, 0].min(),
        )
        hit = self._cam_geom >= 0
        z = np.where(hit, self._cam_dist * dirs[:, 0], np.nan)  # depth along the optical axis
        if c.noise_coeff > 0:
            z = z + self.rng.normal(0.0, 1.0, n) * c.noise_coeff * np.nan_to_num(z) ** 2
        too_close = hit & (z < c.min_range)
        valid = hit & (z >= c.min_range) & (z <= c.max_range)
        dist = z / dirs[:, 0]
        pts = origin + np.nan_to_num(dist)[:, None] * world
        obstacle = valid & (pts[:, 2] > c.min_obstacle_height) & (pts[:, 2] < c.max_obstacle_height)

        rows, cols = c.rows, c.cols
        self.last_depth = np.where(valid, z, np.nan).reshape(rows, cols)
        self.last_cloud = pts[obstacle]
        horiz = np.hypot(pts[:, 0] - origin[0], pts[:, 1] - origin[1])
        horiz = np.where(obstacle, horiz, np.inf).reshape(rows, cols)
        ranges = horiz.min(axis=0)
        unknown = (too_close.reshape(rows, cols)).any(axis=0)
        ranges = np.where(unknown, np.nan, ranges)
        yaw = self.pose()[2]
        return Scan(
            origin=origin[:2],
            angles=yaw + self._cam_col_angle,
            ranges=ranges,
            max_range=c.max_range,
        )

    def fov(self) -> tuple[float, float] | None:
        """Horizontal field of view (half-angle, range) of a limited sensor, None for 360 deg."""
        if self.sensor == "depth":
            return np.radians(self.camera.hfov_deg / 2), self.camera.max_range
        return None
