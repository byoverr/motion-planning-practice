"""Picture of what the RGB-D camera sees: RGB image, depth image and the virtual laser scan.

    MUJOCO_GL=egl uv run python scripts/make_camera_figure.py   # (osmesa/egl only on headless Linux)

Writes docs/img/rgbd_view.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Wedge  # noqa: E402

from mpp.mapping.costmap import Costmap  # noqa: E402
from mpp.sim.sim import Simulation  # noqa: E402
from mpp.viz.plot2d import draw_costmap, draw_layout  # noqa: E402
from mpp.world.layout import Pose2D, warehouse  # noqa: E402
from mpp.world.scenarios import get_scenario  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    lay = warehouse()
    sc = get_scenario("low_box")
    sim = Simulation(lay, sc, Pose2D("view", 4.0, 5.95, 0.12), sensor="depth")
    sim.step(1.0)  # let the box settle
    scan = sim.scan()
    cm = Costmap.from_layout(lay)
    cm.integrate_scan(scan.origin, scan.angles, scan.ranges, scan.max_range)

    r = mujoco.Renderer(sim.model, 360, 540)
    r.update_scene(sim.data, "rgbd")
    rgb = r.render()
    r.enable_depth_rendering()
    r.update_scene(sim.data, "rgbd")
    depth = r.render()
    r.disable_depth_rendering()
    depth = np.where(depth > sim.camera.max_range, np.nan, depth)

    cam = mujoco.MjvCamera()
    cam.lookat[:] = [4.05, 5.95, 0.12]
    cam.distance, cam.azimuth, cam.elevation = 0.75, 200, -18
    r.update_scene(sim.data, cam)
    robot = r.render()

    fig, axes = plt.subplots(1, 4, figsize=(17, 3.9), gridspec_kw={"width_ratios": [1, 1, 1, 1]})
    axes[0].imshow(robot)
    axes[0].set_title("TurtleBot3 Burger + RealSense D435", fontsize=10)
    axes[1].imshow(rgb)
    axes[1].set_title("RGB с камеры", fontsize=10)
    im = axes[2].imshow(depth, cmap="viridis", vmin=0.3, vmax=sim.camera.max_range)
    axes[2].set_title("глубина, м (0.3–3 м)", fontsize=10)
    fig.colorbar(im, ax=axes[2], fraction=0.035)
    for ax in axes[:3]:
        ax.axis("off")

    ax = axes[3]
    draw_costmap(ax, cm)
    draw_layout(ax, lay, labels=False)
    half, rng = sim.fov()
    x, y, yaw = sim.pose()
    ax.add_patch(
        Wedge(
            (x, y),
            rng,
            np.degrees(yaw - half),
            np.degrees(yaw + half),
            color="#2a78d6",
            alpha=0.12,
            zorder=4,
        )
    )
    ok = np.isfinite(scan.ranges)
    hits = scan.origin + scan.ranges[ok, None] * np.column_stack(
        [np.cos(scan.angles[ok]), np.sin(scan.angles[ok])]
    )
    ax.plot(*hits.T, ".", color="#eb6834", ms=4, zorder=6, label="виртуальный скан")
    ax.plot(x, y, "o", color="#1baf7a", ms=8, zorder=7)
    ax.set_xlim(3.5, 7.2)
    ax.set_ylim(4.6, 7.1)
    ax.set_aspect("equal")
    ax.set_title("costmap: точки ниже 40 см -> «скан»", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = ROOT / "docs" / "img" / "rgbd_view.png"
    fig.savefig(out, dpi=120)
    print(out)


if __name__ == "__main__":
    main()
