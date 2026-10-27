"""Build the MuJoCo model: warehouse floor, shelves, falling boxes and the TurtleBot3."""

from __future__ import annotations

import mujoco
import numpy as np

from mpp.robot.model import D435, MJCF_PATH, DepthCameraParams
from mpp.world.layout import Layout, Pose2D
from mpp.world.scenarios import Scenario

# geom groups: 0 = world (walls, shelves, floor), 1 = robot visuals, 2 = boxes, 3 = robot collision
GROUP_WORLD, GROUP_BOX = 0, 2

SHELF_RGBA = (0.36, 0.45, 0.62, 1.0)
WALL_RGBA = (0.78, 0.76, 0.72, 1.0)
BOX_RGBA = (0.71, 0.52, 0.31, 1.0)


def yaw_quat(yaw: float) -> list[float]:
    return [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]


def mount_depth_camera(spec: mujoco.MjSpec, cam: DepthCameraParams = D435) -> None:
    """Replace the LDS lidar by an Intel RealSense D435 (90 x 25 x 25 mm) on the top plate."""
    for g in spec.body("base_scan").geoms:
        g.group = 5  # hide the lidar (group 5 is neither drawn nor ray-cast)
        g.contype = g.conaffinity = 0
    body = spec.body("base_footprint").add_body(name="camera_link", pos=list(cam.mount))
    body.add_geom(
        name="d435_case",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.0125, 0.045, 0.0125],
        rgba=[0.55, 0.56, 0.58, 1.0],
        group=1,
        contype=0,
        conaffinity=0,
        density=0,
    )
    for y in (-0.025, 0.025, 0.0):
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.0045 if y else 0.003, 0.001],
            pos=[0.0126, y, 0.0],
            quat=[0.70710678, 0.0, 0.70710678, 0.0],
            rgba=[0.05, 0.05, 0.05, 1.0],
            group=1,
            contype=0,
            conaffinity=0,
            density=0,
        )
    body.add_site(name="depth_camera", pos=[0.0125, 0.0, 0.0], size=[0.005, 0.005, 0.005])
    # an actual MuJoCo camera with the same optics, for rendering RGB / depth images
    body.add_camera(
        name="rgbd", pos=[0.0125, 0.0, 0.0], xyaxes=[0, -1, 0, 0, 0, 1], fovy=cam.vfov_deg
    )


def build_spec(
    layout: Layout, scenario: Scenario | None, start: Pose2D, sensor: str = "lidar"
) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(MJCF_PATH))
    spec.modelname = "warehouse"
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    spec.visual.headlight.ambient = [0.25, 0.25, 0.25]
    spec.visual.headlight.diffuse = [0.3, 0.3, 0.3]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    spec.visual.quality.shadowsize = 8192
    spec.stat.center = [layout.width / 2, layout.height / 2, 0.0]
    spec.stat.extent = max(layout.width, layout.height) * 0.6

    spec.add_texture(
        name="floor_tex",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        width=256,
        height=256,
        rgb1=[0.86, 0.86, 0.84],
        rgb2=[0.78, 0.78, 0.76],
    )
    floor_mat = spec.add_material(name="floor_mat", texrepeat=[layout.width, layout.height])
    floor_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor_tex"

    wb = spec.worldbody
    wb.add_light(
        name="sun",
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        pos=[layout.width / 2, layout.height / 2, 6.0],
        dir=[0.3, 0.4, -1.0],
        diffuse=[0.45, 0.45, 0.45],
        specular=[0.05, 0.05, 0.05],
        castshadow=True,
    )
    wb.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=[layout.width / 2, layout.height / 2, 0.0],
        size=[layout.width / 2 + 0.3, layout.height / 2 + 0.3, 0.05],
        material="floor_mat",
        group=GROUP_WORLD,
    )

    h = layout.wall_height
    for r in layout.all_rects():
        (cx, cy), (hx, hy) = r.center, r.half_size
        wb.add_geom(
            name=r.name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[cx, cy, h / 2],
            size=[hx, hy, h / 2],
            rgba=WALL_RGBA if r.kind == "wall" else SHELF_RGBA,
            group=GROUP_WORLD,
        )

    for b in scenario.boxes if scenario else ():
        sx, sy, sz = b.size
        body = wb.add_body(name=b.name, pos=[b.x, b.y, b.drop_height + sz / 2], gravcomp=1.0)
        body.add_freejoint(name=f"{b.name}_joint")
        body.add_geom(
            name=f"{b.name}_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[sx / 2, sy / 2, sz / 2],
            mass=b.mass,
            rgba=BOX_RGBA,
            group=GROUP_BOX,
            friction=[0.8, 0.005, 0.0001],
        )

    wb.add_camera(
        name="overview",
        pos=[layout.width / 2, 1.0, 9.5],
        xyaxes=[1, 0, 0, 0, 0.967, 0.254],
        fovy=45,
    )

    if sensor == "depth":
        mount_depth_camera(spec)
    robot = spec.body("base_footprint")
    robot.pos = [start.x, start.y, 0.0]
    robot.quat = yaw_quat(start.yaw)
    robot.add_camera(name="chase", pos=[-0.9, 0.0, 0.7], xyaxes=[0, -1, 0, 0.55, 0, 0.83], fovy=60)
    return spec


def build_model(
    layout: Layout, scenario: Scenario | None, start: Pose2D, sensor: str = "lidar"
) -> mujoco.MjModel:
    return build_spec(layout, scenario, start, sensor).compile()
