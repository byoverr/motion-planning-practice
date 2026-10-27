"""Motion memory: reuse what worked on an earlier run of the same route (research prototype).

Warehouse robots drive the same aisles over and over, yet a local planner re-solves every
local problem from scratch. A motion memory stores the executed trajectory of a successful
run (poses and the commands that produced them). On the next run, wherever the robot is close
to a remembered pose with a similar heading, the memory proposes the command sequence that
followed it. MPPI draws part of its samples around this proposal (as in Biased-MPPI) and
keeps the rest around its own warm start, so a stale or wrong memory costs nothing but a few
wasted samples, while a good one lets the planner work with far fewer samples.

The record is tiny (5 floats per 0.1 s, ~1 kB per 10 m of route; much less if compressed to
splines), so it could be shared across a fleet over a weak network: see
docs/research/fleet_local_planning.md.
"""

from __future__ import annotations

import numpy as np

from mpp.geometry import wrap_angle


class MotionMemory:
    def __init__(
        self,
        trajectory: np.ndarray,
        max_distance: float = 0.25,
        max_heading: float = 0.6,
    ) -> None:
        """trajectory: (N, >=5) rows of (x, y, yaw, v, w) sampled at the control period,
        where (v, w) is the command applied *after* reaching (x, y, yaw)."""
        traj = np.asarray(trajectory, float)
        self.xy = traj[:, :2]
        self.yaw = traj[:, 2]
        self.u = traj[:, 3:5]
        self.max_d2 = max_distance**2
        self.max_heading = max_heading

    @classmethod
    def from_episode(cls, trajectory: np.ndarray, **kwargs) -> MotionMemory:
        """From EpisodeResult.trajectory rows (x, y, yaw, v, w, t), where (v, w) is the
        command that led *to* the row; shift it so commands follow their pose."""
        t = np.asarray(trajectory, float)
        rows = np.column_stack([t[:-1, :3], t[1:, 3:5]])
        return cls(rows, **kwargs)

    def nbytes(self) -> int:
        return int(len(self.xy) * 5 * 4)  # as float32

    def lookup(self, pose, steps: int) -> np.ndarray | None:
        """Command sequence (steps, 2) that followed the closest remembered matching pose."""
        d2 = ((self.xy - np.asarray(pose[:2])) ** 2).sum(axis=1)
        ok = (d2 <= self.max_d2) & (np.abs(wrap_angle(self.yaw - pose[2])) <= self.max_heading)
        if not ok.any():
            return None
        i = int(np.argmin(np.where(ok, d2, np.inf)))
        seq = self.u[i : i + steps]
        if len(seq) < steps:
            pad = np.zeros((steps - len(seq), 2))  # the remembered run stopped at the goal
            seq = np.vstack([seq, pad])
        return seq
