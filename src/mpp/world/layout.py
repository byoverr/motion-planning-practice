"""Static world: a small warehouse with shelves, aisles, a narrow corridor and a dead end.

All coordinates are metres in the world frame, origin in the bottom-left corner of the
floor, x to the right, y up. Shelves are plain boxes ("walls") the robot must drive around.

    y=7 +--------------------------------------------------------------+
        | S3        |L3|        |  D4  |                 |U|           |
        |           |  |        +------+ top aisle 0.9   |U|  G4    G1 |
        |           +--+        |      |         UUUUUUUUU|           |
        |   [====L2====]        |  D3  |      [=====R2=====]          |
        | S2 ------------------ narrow corridor 0.45 m ------------ G2 |
        |   [====L1====]        |  D2  |      [=====R1=====]          |
        |            C1         |      |                 C2           |
        | S1 ------------------ bottom aisle 0.9 m ---------------- G3 |
        |                       |  D1  |                              |
    y=0 +--------------------------------------------------------------+
       x=0                     x=4    x=6                           x=10

    Run ``uv run mpp map`` for an exact picture.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Rect:
    """Axis-aligned box on the floor, [x0, x1] x [y0, y1]."""

    name: str
    x0: float
    y0: float
    x1: float
    y1: float
    kind: str = "shelf"  # "shelf" or "wall", only changes the colour

    @property
    def center(self) -> tuple[float, float]:
        return (0.5 * (self.x0 + self.x1), 0.5 * (self.y0 + self.y1))

    @property
    def half_size(self) -> tuple[float, float]:
        return (0.5 * (self.x1 - self.x0), 0.5 * (self.y1 - self.y0))


@dataclass(frozen=True)
class Pose2D:
    name: str
    x: float
    y: float
    yaw: float = 0.0

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y])


@dataclass
class Layout:
    width: float
    height: float
    rects: list[Rect]
    starts: dict[str, Pose2D] = field(default_factory=dict)
    goals: dict[str, Pose2D] = field(default_factory=dict)
    wall_height: float = 1.0
    wall_thickness: float = 0.1

    def outer_walls(self) -> list[Rect]:
        t, w, h = self.wall_thickness, self.width, self.height
        return [
            Rect("wall_south", -t, -t, w + t, 0.0, "wall"),
            Rect("wall_north", -t, h, w + t, h + t, "wall"),
            Rect("wall_west", -t, 0.0, 0.0, h, "wall"),
            Rect("wall_east", w, 0.0, w + t, h, "wall"),
        ]

    def all_rects(self) -> list[Rect]:
        return self.outer_walls() + list(self.rects)

    def pose(self, name: str) -> Pose2D:
        if name in self.starts:
            return self.starts[name]
        if name in self.goals:
            return self.goals[name]
        raise KeyError(f"unknown pose {name!r}; known: {sorted(self.starts) + sorted(self.goals)}")


def warehouse() -> Layout:
    """The default practice map (10 x 7 m)."""
    rects = [
        # central block of shelves, 2 m deep, with three ways through it
        Rect("D1", 4.0, 0.0, 6.0, 0.9),
        Rect("D2", 4.0, 1.8, 6.0, 3.2),  # bottom aisle: y 0.9 .. 1.8 (0.9 m)
        Rect("D3", 4.0, 3.65, 6.0, 5.6),  # narrow corridor: y 3.2 .. 3.65 (0.45 m)
        Rect("D4", 4.0, 6.5, 6.0, 7.0),  # top aisle: y 5.6 .. 6.5 (0.9 m)
        # left zone
        Rect("L1", 1.2, 2.3, 3.2, 2.7),
        Rect("L2", 1.2, 4.3, 3.2, 4.7),
        Rect("L3", 2.2, 5.3, 2.6, 7.0),
        # right zone
        Rect("R1", 6.8, 2.4, 9.0, 2.8),
        Rect("R2", 6.8, 4.2, 9.0, 4.6),
        # U-shaped dead end that opens towards the top aisle: the straight line to G1 runs into it
        Rect("U_bottom", 7.0, 5.2, 8.6, 5.4),
        Rect("U_side", 8.4, 5.4, 8.6, 7.0),
        # building columns
        Rect("C1", 2.35, 1.15, 2.65, 1.45),
        Rect("C2", 7.6, 1.25, 7.9, 1.55),
    ]
    starts = {
        "S1": Pose2D("S1", 0.5, 1.35, 0.0),
        "S2": Pose2D("S2", 0.5, 3.425, 0.0),
        "S3": Pose2D("S3", 0.5, 6.0, 0.0),
    }
    goals = {
        "G1": Pose2D("G1", 9.4, 6.3, np.pi / 2),
        "G2": Pose2D("G2", 9.4, 3.425, 0.0),
        "G3": Pose2D("G3", 9.4, 1.0, 0.0),
        "G4": Pose2D("G4", 7.7, 6.1, 0.0),
    }
    return Layout(width=10.0, height=7.0, rects=rects, starts=starts, goals=goals)
