"""Navigator: glues the global and the local planner together (a tiny Nav2 behaviour tree).

Each control cycle it

1. checks whether the goal is reached;
2. when the costmap changed, validates the rest of the global path. If an obstacle now sits on
   the path it decides between
   * **bypass**: the obstacle can be passed within a small window around the path and the
     controller can avoid obstacles on its own -> keep the path, the local planner drives around;
   * **replan**: the corridor is blocked (no way around inside the window) or the controller
     only follows paths (pure pursuit) -> ask the global planner for a new route;
3. runs the local planner; if it has no valid command for a while or the robot makes no
   progress, it replans and, if needed, backs up a little (recovery).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from mpp.mapping.costmap import Costmap
from mpp.planning.global_planners.base import GlobalPlanner, PlanResult
from mpp.planning.local_planners.base import LocalPlan, LocalPlanner


@dataclass
class Event:
    t: float
    kind: str  # plan | replan | bypass | box | recovery | goal | fail
    message: str
    xy: tuple[float, float] | None = None


@dataclass
class NavConfig:
    goal_tolerance: float = 0.12
    check_period: float = 0.3  # s between path validations (only when the costmap changed)
    replan_policy: str = "smart"  # smart | always | never  (what to do when the path is blocked)
    bypass_margin: float = 0.6  # m around the blocked part where a detour is searched
    bypass_lead: float = 0.5  # m of path before/after the blocked part the detour must connect
    periodic_replan: float | None = None  # s; Nav2's default tree replans at 1 Hz
    no_traj_timeout: float = 1.5  # s without a valid local command -> replan
    stuck_window: float = 8.0  # s ...
    stuck_distance: float = 0.12  # ... without moving this far -> replan + recovery
    backup_distance: float = 0.12
    backup_speed: float = 0.08
    min_replan_interval: float = 1.0  # s; the lidar keeps revealing more of a new obstacle
    max_replans: int = 40
    max_plan_failures: int = 5


@dataclass
class NavStats:
    plans: list[tuple[float, np.ndarray]] = field(default_factory=list)
    plan_results: list[PlanResult] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    replans: int = 0
    bypasses: int = 0
    recoveries: int = 0
    plan_failures: int = 0


class Navigator:
    def __init__(
        self,
        global_planner: GlobalPlanner,
        local_planner: LocalPlanner,
        costmap: Costmap,
        goal: np.ndarray,
        config: NavConfig | None = None,
    ) -> None:
        self.gp = global_planner
        self.lp = local_planner
        self.cm = costmap
        self.goal = np.asarray(goal, float)
        self.cfg = config or NavConfig()
        self.stats = NavStats()
        self.state = "idle"  # idle | following | backing_up | done | failed
        self.path: np.ndarray | None = None
        self._checked_version = -1
        self._last_check = -1e9
        self._last_plan_t = 0.0
        self._no_traj_since: float | None = None
        self._stuck_ref: tuple[float, np.ndarray] | None = None
        self._backup_from: np.ndarray | None = None
        self._backup_t0 = 0.0
        self._bypassed: list[np.ndarray] = []  # obstacles we already decided to drive around
        self._recheck = False
        self.last_local: LocalPlan | None = None

    # ------------------------------------------------------------------ events

    def _event(self, t: float, kind: str, msg: str, xy=None) -> None:
        self.stats.events.append(Event(t, kind, msg, None if xy is None else tuple(map(float, xy))))

    # ------------------------------------------------------------------ planning

    def start(self, pose: tuple[float, float, float], t: float) -> bool:
        ok = self._plan(pose, t, "plan", "первый план")
        self.state = "following" if ok else "failed"
        return ok

    def _plan(self, pose, t: float, kind: str, reason: str) -> bool:
        xy = np.array(pose[:2])
        start = self.cm.nearest_free(xy)
        if start is None:
            start = xy
        res = self.gp.plan(self.cm, start, self.goal)
        self.stats.plan_results.append(res)
        self._last_plan_t = t
        if kind == "replan":
            self.stats.replans += 1
        if not res.success:
            self.stats.plan_failures += 1
            self._event(t, "fail", f"{self.gp.name}: путь не найден ({reason})", xy)
            return False
        path = res.path
        if np.linalg.norm(start - xy) > 1e-6:
            path = np.vstack([xy, path])
        self.path = path
        self.stats.plans.append((t, path))
        self.lp.set_path(path)
        self._bypassed.clear()
        self._event(t, kind, f"{reason}: {res.length:.2f} м за {res.time * 1000:.0f} мс", xy)
        return True

    def _replan(self, pose, t: float, reason: str) -> None:
        if self.stats.replans >= self.cfg.max_replans:
            self.state = "failed"
            self._event(t, "fail", "слишком много перепланирований", pose[:2])
            return
        ok = self._plan(pose, t, "replan", reason)
        if not ok and self.stats.plan_failures >= self.cfg.max_plan_failures:
            self.state = "failed"

    # ------------------------------------------------------------------ path validation

    def _blocked_run(self) -> tuple[int, int] | None:
        """Indices (on the tracked path) of the first blocked stretch ahead of the robot."""
        ahead = self.lp.path[self.lp.idx :]
        free = self.cm.is_free(ahead)
        if free.all():
            return None
        i0 = int(np.argmin(free))
        after = np.nonzero(free[i0:])[0]
        i1 = i0 + int(after[0]) - 1 if after.size else len(ahead) - 1
        return self.lp.idx + i0, self.lp.idx + i1

    def detour_exists(self, a: np.ndarray, b: np.ndarray, corridor: np.ndarray) -> bool:
        """Is there a collision-free connection between a and b inside a window around the
        path piece `corridor`? Connected-component labelling of free configuration space."""
        cm, m = self.cm, self.cfg.bypass_margin
        pts = np.vstack([corridor, a, b])
        ix0, iy0 = (int(v) for v in cm.world_to_cell(pts.min(axis=0) - m))
        ix1, iy1 = (int(v) for v in cm.world_to_cell(pts.max(axis=0) + m))
        ix0, iy0 = max(ix0, 0), max(iy0, 0)
        ix1, iy1 = min(ix1, cm.nx - 1), min(iy1, cm.ny - 1)
        free = cm.clearance[iy0 : iy1 + 1, ix0 : ix1 + 1] >= cm.inscribed
        labels, _ = ndimage.label(free, structure=np.ones((3, 3)))

        def label_at(p: np.ndarray) -> set[int]:
            # the point is free but its cell centre may sit just inside the inflation
            x, y = (int(v) for v in cm.world_to_cell(p))
            patch = labels[max(y - iy0 - 1, 0) : y - iy0 + 2, max(x - ix0 - 1, 0) : x - ix0 + 2]
            return set(patch[patch > 0].tolist())

        return bool(label_at(a) & label_at(b))

    def _check_path(self, pose, t: float) -> None:
        run = self._blocked_run()
        if run is None:
            return
        i0, i1 = run
        path = self.lp.path
        xy_block = path[(i0 + i1) // 2]
        if i1 == len(path) - 1 and not self.cm.is_free(self.goal):
            self.state = "failed"
            self._event(t, "fail", "цель занята препятствием", self.goal)
            return
        policy = self.cfg.replan_policy
        if policy == "never":
            return
        if t - self._last_plan_t < self.cfg.min_replan_interval:
            self._recheck = True  # look again once the interval has passed
            return
        if policy == "always":
            self._replan(pose, t, "путь перекрыт")
            return
        if not self.lp.avoids_obstacles:
            self._replan(pose, t, f"путь перекрыт, {self.lp.name} не умеет объезжать")
            return
        lead = int(self.cfg.bypass_lead / 0.05)
        ia = max(i0 - lead, self.lp.idx)
        ib = min(i1 + lead, len(path) - 1)
        a = path[ia]
        if not self.cm.is_free(a):
            a = self.cm.nearest_free(np.array(pose[:2]))
        b = path[ib]
        if a is not None and self.cm.is_free(b) and self.detour_exists(a, b, path[ia : ib + 1]):
            if all(np.linalg.norm(xy_block - p) > 0.6 for p in self._bypassed):
                self._bypassed.append(xy_block)
                self.stats.bypasses += 1
                self._event(
                    t, "bypass", "препятствие на пути, проход есть: объезд локально", xy_block
                )
            return
        self._replan(pose, t, "проход перекрыт, объехать нельзя")

    # ------------------------------------------------------------------ control loop

    def tick(
        self, pose: tuple[float, float, float], vel: tuple[float, float], t: float
    ) -> tuple[float, float]:
        if self.state in ("done", "failed", "idle"):
            return 0.0, 0.0
        xy = np.array(pose[:2])
        if np.linalg.norm(xy - self.goal) <= self.cfg.goal_tolerance:
            self.state = "done"
            self._event(t, "goal", "цель достигнута", xy)
            return 0.0, 0.0

        if self.state == "backing_up":
            moved = np.linalg.norm(xy - self._backup_from)
            if moved >= self.cfg.backup_distance or t - self._backup_t0 > 3.0:
                self.state = "following"
                self._replan(pose, t, "после отъезда назад")
            return -self.cfg.backup_speed, 0.0

        changed = self.cm.version != self._checked_version or self._recheck
        if changed and t - self._last_check >= self.cfg.check_period:
            self._checked_version = self.cm.version
            self._last_check = t
            self._recheck = False
            self._check_path(pose, t)
            if self.state != "following":
                return 0.0, 0.0
        if self.cfg.periodic_replan and t - self._last_plan_t >= self.cfg.periodic_replan:
            self._plan(pose, t, "replan", "периодическое перепланирование")

        lp = self.lp.compute(pose, vel, self.cm)
        self.last_local = lp
        if lp.status != "ok":
            if self._no_traj_since is None:
                self._no_traj_since = t
            elif t - self._no_traj_since > self.cfg.no_traj_timeout:
                self._no_traj_since = None
                self._recover(pose, t, f"{self.lp.name}: {lp.status}")
        else:
            self._no_traj_since = None

        # stuck detection: the robot has to move a little within every window
        if self._stuck_ref is None:
            self._stuck_ref = (t, xy)
        elif t - self._stuck_ref[0] > self.cfg.stuck_window:
            if np.linalg.norm(xy - self._stuck_ref[1]) < self.cfg.stuck_distance:
                self._recover(pose, t, "нет прогресса")
            self._stuck_ref = (t, xy)
        return lp.v, lp.w

    def _recover(self, pose, t: float, reason: str) -> None:
        """Replan first; if the robot sits inside the inflation, back up before replanning."""
        self.stats.recoveries += 1
        xy = np.array(pose[:2])
        self._event(t, "recovery", reason, xy)
        if self.cm.is_free(xy) or not self._can_back_up(pose):
            self._replan(pose, t, reason)
        else:
            self.state = "backing_up"
            self._backup_from = xy
            self._backup_t0 = t

    def _can_back_up(self, pose) -> bool:
        """Backing up is blind for the lidar-less rear in many robots; here we at least check
        the costmap so the recovery does not push the robot into a shelf."""
        x, y, yaw = pose
        d = np.linspace(0.0, self.cfg.backup_distance + 0.03, 8)
        pts = np.column_stack([x - d * np.cos(yaw), y - d * np.sin(yaw)])
        clear = self.cm.clearance_at(pts)
        return bool(clear[-1] > clear[0] and (clear > self.lp.robot.radius + 0.01).all())
