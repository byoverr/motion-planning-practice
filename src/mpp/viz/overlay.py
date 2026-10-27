"""Draw plans and trajectories into a MuJoCo scene (interactive viewer or offscreen renderer)."""

from __future__ import annotations

import mujoco
import numpy as np

from mpp.navigation.navigator import Navigator

Z = 0.02  # draw slightly above the floor

GLOBAL_RGBA = np.array([0.16, 0.47, 0.84, 0.9])
LOCAL_RGBA = np.array([0.92, 0.41, 0.20, 1.0])
CANDIDATE_RGBA = np.array([0.6, 0.6, 0.6, 0.25])
TRACE_RGBA = np.array([0.11, 0.69, 0.48, 0.9])
GOAL_RGBA = np.array([0.89, 0.29, 0.28, 0.8])
OBSTACLE_RGBA = np.array([0.93, 0.63, 0.0, 0.9])


def _segment(scn: mujoco.MjvScene, a, b, width: float, rgba) -> bool:
    if scn.ngeom >= scn.maxgeom:
        return False
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), rgba
    )
    mujoco.mjv_connector(
        g,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        width,
        np.array([a[0], a[1], Z]),
        np.array([b[0], b[1], Z]),
    )
    scn.ngeom += 1
    return True


def _polyline(scn, pts: np.ndarray, width: float, rgba, max_segments: int = 150) -> None:
    if pts is None or len(pts) < 2:
        return
    step = max(1, len(pts) // max_segments)
    pts = np.vstack([pts[::step], pts[-1:]])
    for a, b in zip(pts[:-1], pts[1:], strict=True):
        if not _segment(scn, a, b, width, rgba):
            return


def _sphere(scn, xy, radius: float, rgba, z: float = Z) -> None:
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0, 0]),
        np.array([xy[0], xy[1], z]),
        np.eye(3).ravel(),
        rgba,
    )
    scn.ngeom += 1


def draw_navigation(
    scn: mujoco.MjvScene,
    nav: Navigator,
    trace: np.ndarray | None = None,
    candidates: bool = True,
) -> None:
    """Append the global path, the local plan, the executed trace, the goal and the obstacle
    cells the lidar has added to the costmap."""
    if trace is not None:
        _polyline(scn, trace, 0.025, TRACE_RGBA)
    if nav.lp.path is not None:
        _polyline(scn, nav.lp.path[nav.lp.idx :], 0.02, GLOBAL_RGBA)
    lp = nav.last_local
    if lp is not None:
        if candidates and lp.candidates is not None:
            for c in lp.candidates[:20]:
                _polyline(scn, c, 0.003, CANDIDATE_RGBA, max_segments=6)
        if lp.best is not None:
            _polyline(scn, lp.best[:, :2], 0.025, LOCAL_RGBA, max_segments=20)
    _sphere(scn, nav.goal, 0.09, GOAL_RGBA, z=0.09)
    cm = nav.cm
    iy, ix = np.nonzero(cm.obstacles & ~cm.static)
    for xy in cm.cell_to_world(ix, iy)[:300]:
        _sphere(scn, xy, 0.025, OBSTACLE_RGBA, z=0.2)
