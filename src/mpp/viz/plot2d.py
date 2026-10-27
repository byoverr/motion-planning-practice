"""Top-down matplotlib pictures of the map, plans and executed trajectories."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

from mpp.mapping.costmap import Costmap  # noqa: E402
from mpp.world.layout import Layout  # noqa: E402
from mpp.world.scenarios import Scenario  # noqa: E402

# categorical slots of the reference palette (validated for colour-blind separation)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SHELF = "#8d99ae"
WALL = "#2b2d42"
BOX = "#b5651d"


def draw_layout(ax, layout: Layout, labels: bool = True) -> None:
    for r in layout.all_rects():
        ax.add_patch(
            Rectangle(
                (r.x0, r.y0),
                r.x1 - r.x0,
                r.y1 - r.y0,
                facecolor=WALL if r.kind == "wall" else SHELF,
                edgecolor="none",
                zorder=2,
            )
        )
    if not labels:
        return
    for p in layout.starts.values():
        ax.plot(p.x, p.y, marker="s", color="#2a9d8f", ms=9, zorder=6)
        ax.annotate(p.name, (p.x, p.y), xytext=(6, 6), textcoords="offset points", zorder=6)
    for p in layout.goals.values():
        ax.plot(p.x, p.y, marker="*", color="#e63946", ms=14, zorder=6)
        ax.annotate(p.name, (p.x, p.y), xytext=(6, 6), textcoords="offset points", zorder=6)


def draw_costmap(ax, costmap: Costmap, alpha: float = 0.55) -> None:
    img = np.where(costmap.lethal, np.nan, costmap.cost)
    ax.imshow(
        img,
        origin="lower",
        extent=costmap.extent,
        cmap="Purples",
        vmin=0.0,
        vmax=1.4,
        alpha=alpha,
        interpolation="nearest",
        zorder=1,
    )
    obs = np.where(costmap.obstacles & ~costmap.static, 1.0, np.nan)
    if np.isfinite(obs).any():
        ax.imshow(
            obs,
            origin="lower",
            extent=costmap.extent,
            cmap="autumn",
            vmin=0,
            vmax=1,
            interpolation="nearest",
            zorder=3,
        )


def draw_boxes(ax, scenario: Scenario | None, final_xy: dict[str, np.ndarray] | None = None):
    if scenario is None:
        return
    for b in scenario.boxes:
        x, y = (final_xy or {}).get(b.name, (b.x, b.y))
        sx, sy = b.size[0], b.size[1]
        ax.add_patch(
            Rectangle(
                (x - sx / 2, y - sy / 2),
                sx,
                sy,
                facecolor="none",
                edgecolor=BOX,
                lw=1.8,
                ls="--",
                zorder=7,
            )
        )


def draw_tree(ax, edges: np.ndarray | None, color: str = "#adb5bd") -> None:
    """edges: (E, 2, 2) array of segments."""
    if edges is None or len(edges) == 0:
        return
    from matplotlib.collections import LineCollection

    ax.add_collection(LineCollection(edges, colors=color, linewidths=0.4, alpha=0.7, zorder=4))


def new_axes(layout: Layout, title: str = "", figsize=(11, 7.9)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-0.2, layout.width + 0.2)
    ax.set_ylim(-0.2, layout.height + 0.2)
    ax.set_aspect("equal")
    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")
    ax.grid(True, lw=0.3, alpha=0.4)
    if title:
        ax.set_title(title)
    return fig, ax


def plot_map(layout: Layout, costmap: Costmap | None, out: Path, title: str = "") -> None:
    fig, ax = new_axes(layout, title)
    if costmap is not None:
        draw_costmap(ax, costmap)
    draw_layout(ax, layout)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_plans(
    layout: Layout,
    costmap: Costmap,
    plans: dict[str, np.ndarray | None],
    out: Path,
    title: str = "",
    trees: dict[str, np.ndarray | None] | None = None,
) -> None:
    """Several global plans for the same start/goal on one picture."""
    fig, ax = new_axes(layout, title)
    draw_costmap(ax, costmap)
    draw_layout(ax, layout)
    for k, (name, path) in enumerate(plans.items()):
        color = SERIES[k % len(SERIES)]
        if trees and trees.get(name) is not None:
            draw_tree(ax, trees[name], color=color)
        if path is None:
            continue
        ax.plot(path[:, 0], path[:, 1], lw=2.2, color=color, label=name, zorder=8)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_episode(
    layout: Layout,
    costmap: Costmap,
    scenario: Scenario | None,
    trajectory: np.ndarray,
    plans: Iterable[tuple[float, np.ndarray]],
    events: Iterable,
    out: Path,
    title: str = "",
    box_xy: dict[str, np.ndarray] | None = None,
    robot_radius: float = 0.105,
) -> None:
    """Executed trajectory, every global plan that was produced, and the navigation events."""
    fig, ax = new_axes(layout, title)
    draw_costmap(ax, costmap)
    draw_layout(ax, layout)
    draw_boxes(ax, scenario, box_xy)
    plans = list(plans)
    for k, (_t, path) in enumerate(plans):
        if path is None:
            continue
        first = k == 0
        ax.plot(
            path[:, 0],
            path[:, 1],
            lw=1.6 if first else 1.3,
            ls="--",
            color="#1d3557" if first else "#f4a261",
            alpha=0.9,
            label="первый глобальный план" if first else ("перепланирование" if k == 1 else None),
            zorder=8,
        )
    if len(trajectory):
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            lw=2.4,
            color="#2a9d8f",
            label="фактическая траектория",
            zorder=9,
        )
        ax.add_patch(Circle(trajectory[-1, :2], robot_radius, color="#2a9d8f", alpha=0.6, zorder=9))
    marks = {
        "box": ("v", BOX, "упала коробка"),
        "bypass": ("o", "#e9c46a", "объезд локально"),
        "replan": ("X", "#e76f51", "перепланирование"),
        "recovery": ("P", "#9b5de5", "восстановление"),
        "fail": ("x", "#d00000", "отказ"),
    }
    seen = set()
    for e in events:
        if e.kind not in marks or e.xy is None:
            continue
        m, c, label = marks[e.kind]
        ax.plot(
            *e.xy,
            marker=m,
            color=c,
            ms=11,
            mec="k",
            zorder=10,
            label=None if e.kind in seen else label,
        )
        seen.add(e.kind)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
