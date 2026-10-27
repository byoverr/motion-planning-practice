"""Run one navigation episode: simulator + costmap + navigator, and collect metrics."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np

from mpp.geometry import path_length
from mpp.mapping.costmap import Costmap
from mpp.navigation.navigator import NavConfig, Navigator
from mpp.planning.global_planners import make_global_planner
from mpp.planning.local_planners import make_local_planner
from mpp.robot.model import BURGER
from mpp.sim.sim import Simulation
from mpp.world.layout import Layout, warehouse
from mpp.world.scenarios import Scenario, get_scenario


@dataclass
class EpisodeConfig:
    scenario: str = "open"
    global_planner: str = "astar"
    local_planner: str = "dwa"
    start: str | None = None  # override the scenario's start / goal names
    goal: str | None = None
    seed: int = 0
    control_dt: float = 0.1
    scan_period: float = 0.2  # LDS-01 spins at 5 Hz
    max_time: float = 150.0  # simulated seconds
    global_kwargs: dict = field(default_factory=dict)
    local_kwargs: dict = field(default_factory=dict)
    nav: NavConfig = field(default_factory=NavConfig)
    lidar_noise: float = 0.0


@dataclass
class EpisodeResult:
    scenario: str
    global_planner: str
    local_planner: str
    start: str
    goal: str
    seed: int
    success: bool
    status: str
    sim_time: float
    distance: float
    first_plan_length: float
    first_plan_time: float
    total_plan_time: float
    replans: int
    bypasses: int
    recoveries: int
    collisions: int
    contact_time: float
    min_gap: float  # closest distance between the robot body and an obstacle, m
    mean_speed: float
    wall_time: float
    trajectory: np.ndarray = field(repr=False, default=None)
    plans: list = field(repr=False, default_factory=list)
    plan_results: list = field(repr=False, default_factory=list)
    events: list = field(repr=False, default_factory=list)
    costmap: Costmap | None = field(repr=False, default=None)
    box_positions: dict = field(repr=False, default_factory=dict)

    def row(self) -> dict:
        skip = {"trajectory", "plans", "plan_results", "events", "costmap", "box_positions"}
        return {k: v for k, v in asdict(self).items() if k not in skip}


def make_episode(cfg: EpisodeConfig, layout: Layout | None = None):
    layout = layout or warehouse()
    scenario: Scenario = get_scenario(cfg.scenario)
    start = layout.pose(cfg.start or scenario.start)
    goal = layout.pose(cfg.goal or scenario.goal)
    sim = Simulation(layout, scenario, start, lidar_noise=cfg.lidar_noise, seed=cfg.seed)
    costmap = Costmap.from_layout(layout, robot_radius=BURGER.radius)
    gp = make_global_planner(cfg.global_planner, seed=cfg.seed, **cfg.global_kwargs)
    local_kwargs = dict(cfg.local_kwargs)
    if cfg.local_planner == "mppi":
        local_kwargs.setdefault("seed", cfg.seed)  # MPPI samples random control sequences
    lp = make_local_planner(cfg.local_planner, BURGER, dt=cfg.control_dt, **local_kwargs)
    nav = Navigator(gp, lp, costmap, goal.xy, cfg.nav)
    return layout, scenario, start, goal, sim, costmap, nav


def run_episode(
    cfg: EpisodeConfig,
    on_step: Callable[[Simulation, Navigator], bool | None] | None = None,
) -> EpisodeResult:
    """Simulate until the goal is reached, the navigator gives up or time runs out.

    `on_step(sim, nav)` is called after every control step (used by the viewer); returning
    False stops the episode.
    """
    wall0 = time.perf_counter()
    layout, scenario, start, goal, sim, costmap, nav = make_episode(cfg)

    def scan() -> None:
        s = sim.scan()
        costmap.integrate_scan(s.origin, s.angles, s.ranges, s.max_range)

    scan()
    nav.start(sim.pose(), sim.time)
    traj = [(*sim.pose(), 0.0, 0.0, sim.time)]
    next_scan = cfg.scan_period
    known_drops: set[str] = set()
    status = "timeout"
    while sim.time < cfg.max_time:
        if sim.time + 1e-9 >= next_scan:
            scan()
            next_scan += cfg.scan_period
        pose, vel = sim.pose(), sim.velocity()
        v, w = nav.tick(pose, vel, sim.time)
        if nav.state in ("done", "failed"):
            status = "goal" if nav.state == "done" else "failed"
            break
        sim.set_cmd(v, w)
        sim.step(cfg.control_dt)
        for name in sim.dropped.keys() - known_drops:
            known_drops.add(name)
            box = sim.boxes[name]
            nav._event(sim.time, "box", f"упала коробка {name}", (box.x, box.y))
        x, y, yaw = sim.pose()
        traj.append((x, y, yaw, v, w, sim.time))
        if on_step is not None and on_step(sim, nav) is False:
            status = "stopped"
            break
    sim.set_cmd(0.0, 0.0)
    traj = np.array(traj)

    # ground truth clearance: known map + boxes where they actually landed
    truth = Costmap.from_layout(layout, robot_radius=BURGER.radius)
    boxes = sim.box_positions()
    for name, xy in boxes.items():
        b = sim.boxes[name]
        if name in sim.dropped:
            truth.fill_rect(
                xy[0] - b.size[0] / 2,
                xy[1] - b.size[1] / 2,
                xy[0] + b.size[0] / 2,
                xy[1] + b.size[1] / 2,
            )
    truth.update()
    gap = float(truth.clearance_at(traj[:, :2]).min() - BURGER.radius)

    st = nav.stats
    first = st.plan_results[0] if st.plan_results else None
    dist = path_length(traj[:, :2])
    return EpisodeResult(
        scenario=cfg.scenario,
        global_planner=cfg.global_planner,
        local_planner=cfg.local_planner,
        start=start.name,
        goal=goal.name,
        seed=cfg.seed,
        success=status == "goal",
        status=status,
        sim_time=round(sim.time, 2),
        distance=round(dist, 3),
        first_plan_length=round(first.length, 3) if first and first.success else float("nan"),
        first_plan_time=round(first.time, 4) if first else float("nan"),
        total_plan_time=round(sum(r.time for r in st.plan_results), 4),
        replans=st.replans,
        bypasses=st.bypasses,
        recoveries=st.recoveries,
        collisions=sim.contacts.events,
        contact_time=round(sim.contacts.time, 3),
        min_gap=round(gap, 3),
        mean_speed=round(dist / max(sim.time, 1e-6), 3),
        wall_time=round(time.perf_counter() - wall0, 2),
        trajectory=traj,
        plans=st.plans,
        plan_results=st.plan_results,
        events=st.events,
        costmap=costmap,
        box_positions=boxes,
    )
