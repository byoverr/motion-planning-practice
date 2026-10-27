"""2D costmap: a static layer from the known map plus an obstacle layer from the lidar.

Like the Nav2 costmap, but in one small class:

* ``static``    occupied cells from the known shelves and walls;
* ``obstacles`` cells marked by lidar hits and cleared by ray tracing (the fallen box ends up here);
* ``clearance`` distance in metres from every cell to the nearest occupied cell;
* ``cost``      0..1 "inflation" cost that decays with clearance, 1 inside the inscribed radius.

The robot is a disc, so collision checking in configuration space is just
``clearance(p) >= inscribed`` where ``inscribed = robot radius + safety margin``.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from mpp.world.layout import Layout


class Costmap:
    def __init__(
        self,
        width: float,
        height: float,
        resolution: float = 0.05,
        origin: tuple[float, float] = (0.0, 0.0),
        robot_radius: float = 0.105,
        safety_margin: float = 0.03,
        inflation_radius: float = 0.5,
        cost_scaling: float = 8.0,
    ) -> None:
        self.res = float(resolution)
        self.origin = np.array(origin, dtype=float)
        self.nx = int(np.ceil(width / self.res - 1e-9))
        self.ny = int(np.ceil(height / self.res - 1e-9))
        self.robot_radius = robot_radius
        self.inscribed = robot_radius + safety_margin
        self.inflation_radius = inflation_radius
        self.cost_scaling = cost_scaling
        self.static = np.zeros((self.ny, self.nx), dtype=bool)
        self.obstacles = np.zeros((self.ny, self.nx), dtype=bool)
        self.version = 0
        self._static_near = self.static
        self.update()

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_layout(cls, layout: Layout, resolution: float = 0.05, **kwargs) -> Costmap:
        pad = layout.wall_thickness
        cm = cls(
            layout.width + 2 * pad,
            layout.height + 2 * pad,
            resolution=resolution,
            origin=(-pad, -pad),
            **kwargs,
        )
        for r in layout.all_rects():
            cm.fill_rect(r.x0, r.y0, r.x1, r.y1)
        cm.update()
        return cm

    def fill_rect(self, x0: float, y0: float, x1: float, y1: float, layer: str = "static") -> None:
        """Mark every cell that overlaps the rectangle (conservative rasterisation)."""
        eps = 1e-6
        ix0 = int(np.floor((x0 - self.origin[0]) / self.res + eps))
        ix1 = int(np.ceil((x1 - self.origin[0]) / self.res - eps))
        iy0 = int(np.floor((y0 - self.origin[1]) / self.res + eps))
        iy1 = int(np.ceil((y1 - self.origin[1]) / self.res - eps))
        ix0, iy0 = max(ix0, 0), max(iy0, 0)
        ix1, iy1 = min(ix1, self.nx), min(iy1, self.ny)
        getattr(self, layer)[iy0:iy1, ix0:ix1] = True

    def copy(self) -> Costmap:
        other = object.__new__(Costmap)
        other.__dict__.update(self.__dict__)
        for name in ("static", "obstacles", "lethal", "clearance", "cost", "_static_near"):
            setattr(other, name, getattr(self, name).copy())
        other.origin = self.origin.copy()
        return other

    # ------------------------------------------------------------------ derived layers

    def update(self) -> None:
        """Recompute clearance and cost after the layers changed."""
        self.lethal = self.static | self.obstacles
        self._static_near = ndimage.binary_dilation(self.static, iterations=1)
        if self.lethal.any():
            edt = ndimage.distance_transform_edt(~self.lethal)
        else:
            edt = np.full(self.lethal.shape, 1e3)
        # distance from a cell centre to the boundary of the nearest occupied cell
        self.clearance = np.where(self.lethal, 0.0, edt * self.res - 0.5 * self.res)
        c = np.exp(-self.cost_scaling * (self.clearance - self.inscribed))
        c = np.where(self.clearance < self.inscribed, 1.0, c * 0.95)
        c[self.clearance > self.inflation_radius] = 0.0
        self.cost = c
        self.version += 1

    # ------------------------------------------------------------------ coordinates

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """(xmin, xmax, ymin, ymax) for matplotlib imshow."""
        x0, y0 = self.origin
        return (x0, x0 + self.nx * self.res, y0, y0 + self.ny * self.res)

    def world_to_cell(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xy = np.asarray(xy, dtype=float)
        ix = np.floor((xy[..., 0] - self.origin[0]) / self.res).astype(int)
        iy = np.floor((xy[..., 1] - self.origin[1]) / self.res).astype(int)
        return ix, iy

    def cell_to_world(self, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
        x = self.origin[0] + (np.asarray(ix) + 0.5) * self.res
        y = self.origin[1] + (np.asarray(iy) + 0.5) * self.res
        return np.stack([x, y], axis=-1)

    def in_bounds(self, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
        return (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)

    # ------------------------------------------------------------------ queries

    def _sample(self, field: np.ndarray, xy: np.ndarray, outside: float) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        shape = xy.shape[:-1]
        pts = xy.reshape(-1, 2)
        # fractional cell coordinates relative to cell centres
        fx = (pts[:, 0] - self.origin[0]) / self.res - 0.5
        fy = (pts[:, 1] - self.origin[1]) / self.res - 0.5
        vals = ndimage.map_coordinates(field, [fy, fx], order=1, mode="nearest")
        inside = (fx >= -0.5) & (fx <= self.nx - 0.5) & (fy >= -0.5) & (fy <= self.ny - 0.5)
        vals = np.where(inside, vals, outside)
        return vals.reshape(shape)

    def clearance_at(self, xy: np.ndarray) -> np.ndarray:
        """Distance (m) to the nearest occupied cell, bilinearly interpolated."""
        return self._sample(self.clearance, xy, outside=0.0)

    def cost_at(self, xy: np.ndarray) -> np.ndarray:
        return self._sample(self.cost, xy, outside=1.0)

    def is_free(self, xy: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """True where a disc robot centred at xy does not collide (configuration space)."""
        return self.clearance_at(xy) >= self.inscribed + margin

    def segment_free(
        self, a: np.ndarray, b: np.ndarray, step: float | None = None, margin: float = 0.0
    ) -> bool:
        a, b = np.asarray(a, float), np.asarray(b, float)
        step = step or 0.25 * self.res
        n = max(int(np.ceil(np.linalg.norm(b - a) / step)), 1)
        t = np.linspace(0.0, 1.0, n + 1)[:, None]
        return bool(np.all(self.is_free(a + t * (b - a), margin)))

    def nearest_free(self, xy: np.ndarray, max_dist: float = 0.6) -> np.ndarray | None:
        """Closest free cell centre to xy (used when the robot starts inside the inflation)."""
        xy = np.asarray(xy, float)
        if self.is_free(xy):
            return xy
        ix, iy = self.world_to_cell(xy)
        r = int(np.ceil(max_dist / self.res))
        xs = np.arange(max(ix - r, 0), min(ix + r + 1, self.nx))
        ys = np.arange(max(iy - r, 0), min(iy + r + 1, self.ny))
        gx, gy = np.meshgrid(xs, ys)
        centres = self.cell_to_world(gx, gy).reshape(-1, 2)
        ok = self.clearance[gy, gx].ravel() >= self.inscribed + 0.25 * self.res
        if not ok.any():
            return None
        centres = centres[ok]
        d = np.linalg.norm(centres - xy, axis=1)
        k = int(np.argmin(d))
        return centres[k] if d[k] <= max_dist else None

    # ------------------------------------------------------------------ sensing

    def integrate_scan(
        self,
        origin: np.ndarray,
        angles: np.ndarray,
        ranges: np.ndarray,
        max_range: float,
    ) -> bool:
        """Clear cells along every ray and mark the hit cells. Returns True if anything changed.

        angles are world-frame ray directions; a range >= max_range (or inf) means "no hit",
        NaN means an invalid reading and the ray is skipped.
        Hits right next to a known shelf are ignored so the static walls do not grow.
        """
        origin = np.asarray(origin, float)[:2]
        valid = ~np.isnan(ranges)
        angles, ranges = np.asarray(angles)[valid], np.asarray(ranges)[valid]
        dirs = np.column_stack([np.cos(angles), np.sin(angles)])
        hit = ranges < max_range - 1e-6
        r = np.minimum(ranges, max_range)
        before = self.obstacles.copy()

        # clearing: free space between the sensor and (just before) the hit
        steps = np.arange(0.0, max_range, 0.5 * self.res)
        pts = origin + steps[None, :, None] * dirs[:, None, :]
        keep = steps[None, :] < (r[:, None] - self.res)
        ix, iy = self.world_to_cell(pts[keep])
        ok = self.in_bounds(ix, iy)
        self.obstacles[iy[ok], ix[ok]] = False

        # marking
        hp = origin + r[hit, None] * dirs[hit]
        ix, iy = self.world_to_cell(hp)
        ok = self.in_bounds(ix, iy)
        ix, iy = ix[ok], iy[ok]
        new = ~self._static_near[iy, ix]
        self.obstacles[iy[new], ix[new]] = True

        changed = bool((before != self.obstacles).any())
        if changed:
            self.update()
        return changed
