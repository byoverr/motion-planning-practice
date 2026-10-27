"""Command line: ``uv run mpp <command>`` (or ``uv run python -m mpp``)."""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from datetime import datetime
from pathlib import Path

from mpp.navigation.navigator import NavConfig
from mpp.planning.global_planners import GLOBAL_PLANNERS
from mpp.planning.local_planners import LOCAL_PLANNERS
from mpp.world.scenarios import SCENARIOS


def _csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _out_dir(arg: str | None, kind: str) -> Path:
    if arg:
        return Path(arg)
    return Path("results") / f"{kind}_{datetime.now():%Y%m%d_%H%M%S}"


def _nav_config(a: argparse.Namespace) -> NavConfig:
    return NavConfig(replan_policy=a.replan_policy, periodic_replan=a.replan_period)


def _add_nav_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--replan-policy",
        default="smart",
        choices=["smart", "always", "never"],
        help="что делать, когда на пути появилось препятствие: smart = объезд, если он есть, "
        "иначе перепланирование; always = всегда перепланировать; never = только при застревании",
    )
    p.add_argument(
        "--replan-period",
        type=float,
        default=None,
        help="перепланировать каждые N секунд (Nav2: 1)",
    )


def cmd_run(a: argparse.Namespace) -> int:
    if a.camera == "rgbd" and a.sensor != "depth":
        sys.exit("--camera rgbd показывает вид с камеры D435, добавьте --sensor depth")
    from mpp.runner import EpisodeConfig, run_episode
    from mpp.viz.plot2d import plot_episode
    from mpp.world.layout import warehouse
    from mpp.world.scenarios import get_scenario

    cfg = EpisodeConfig(
        scenario=a.scenario,
        global_planner=a.global_planner,
        local_planner=a.local_planner,
        start=a.start,
        goal=a.goal,
        seed=a.seed,
        max_time=a.max_time,
        nav=_nav_config(a),
        sensor=a.sensor,
    )
    out = _out_dir(a.out, f"run_{a.scenario}_{a.global_planner}_{a.local_planner}")
    out.mkdir(parents=True, exist_ok=True)
    if a.viewer:
        from mpp.viz.viewer import run_with_viewer

        r = run_with_viewer(cfg, speed=a.speed)
    elif a.gif:
        from mpp.viz.viewer import run_with_recording

        r = run_with_recording(cfg, out / "episode.gif", camera=a.camera)
        print(f"GIF: {out / 'episode.gif'}")
    else:
        r = run_episode(cfg)

    for e in r.events:
        print(f"  {e.t:6.1f} c  {e.kind:9s} {e.message}")
    print()
    for k, v in r.row().items():
        print(f"  {k:18s} {v}")
    title = (
        f"{a.scenario}: {a.global_planner} + {a.local_planner} — "
        f"{'успех' if r.success else r.status}, {r.sim_time:.0f} с"
    )
    plot_episode(
        warehouse(),
        r.costmap,
        get_scenario(a.scenario),
        r.trajectory[:, :2],
        r.plans,
        r.events,
        out / "episode.png",
        title,
        r.box_positions,
    )
    print(f"\nкартинка: {out / 'episode.png'}")
    return 0 if r.success else 1


def cmd_bench(a: argparse.Namespace) -> int:
    from mpp.bench import run_matrix

    out = _out_dir(a.out, "bench")
    rows = run_matrix(
        _csv(a.scenarios),
        _csv(a.global_planners),
        _csv(a.local_planners),
        list(range(a.seeds)),
        out,
        jobs=a.jobs,
        nav=_nav_config(a),
        sensors=_csv(a.sensors),
        plots=not a.no_plots,
    )
    ok = sum(r["success"] for r in rows)
    print(f"\n{ok}/{len(rows)} успешных")
    print(f"таблицы: {out / 'summary.md'}, график: {out / 'time_to_goal.png'}")
    return 0


def cmd_plan(a: argparse.Namespace) -> int:
    from mpp.bench import plan_benchmark
    from mpp.world.layout import warehouse

    layout = warehouse()
    if a.all_pairs:
        pairs = list(itertools.product(sorted(layout.starts), sorted(layout.goals)))
    else:
        pairs = [(a.start, a.goal)]
    kwargs = {n: {"time_limit": a.time_limit} for n in ("rrt_connect", "bit_star", "ait_star")}
    plan_benchmark(
        _csv(a.planners), pairs, list(range(a.seeds)), _out_dir(a.out, "plan"), a.blocked, kwargs
    )
    return 0


def cmd_map(a: argparse.Namespace) -> int:
    from mpp.mapping.costmap import Costmap
    from mpp.viz.plot2d import plot_map
    from mpp.world.layout import warehouse

    layout = warehouse()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_map(layout, Costmap.from_layout(layout), out, "Склад: стеллажи, старты S*, цели G*")
    print(out)
    return 0


def cmd_scene(a: argparse.Namespace) -> int:
    """Export the MuJoCo scene as MJCF, or open it in the viewer."""
    from mpp.sim.scene import build_spec
    from mpp.world.layout import warehouse
    from mpp.world.scenarios import get_scenario

    layout = warehouse()
    sc = get_scenario(a.scenario)
    spec = build_spec(layout, sc, layout.pose(sc.start))
    if a.export:
        from mpp.robot.model import MJCF_PATH

        spec.meshdir = str(MJCF_PATH.parent / "meshes")  # absolute, so the file works anywhere
        spec.compile()
        Path(a.export).write_text(spec.to_xml(), encoding="utf-8")
        print(a.export)
        return 0
    import mujoco.viewer

    mujoco.viewer.launch(spec.compile())
    return 0


def cmd_robot(a: argparse.Namespace) -> int:
    import mujoco
    import mujoco.viewer

    from mpp.robot.model import MJCF_PATH

    spec = mujoco.MjSpec.from_file(str(MJCF_PATH))
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1])
    spec.worldbody.add_light(pos=[0, 0, 2], dir=[0, 0, -1])
    mujoco.viewer.launch(spec.compile())
    return 0


def cmd_list(a: argparse.Namespace) -> int:
    from mpp.world.layout import warehouse

    layout = warehouse()
    print("Сценарии:")
    for s in SCENARIOS.values():
        print(f"  {s.name:8s} {s.start}->{s.goal}  {s.title}")
    print("Глобальные планировщики:", ", ".join([*GLOBAL_PLANNERS, "dijkstra", "ompl:<Planner>"]))
    print("Локальные планировщики: ", ", ".join(LOCAL_PLANNERS))
    print("Старты:", ", ".join(f"{p.name}({p.x}, {p.y})" for p in layout.starts.values()))
    print("Цели:  ", ", ".join(f"{p.name}({p.x}, {p.y})" for p in layout.goals.values()))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mpp", description="Практика планирования движения: TurtleBot3 в MuJoCo"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="один заезд")
    r.add_argument("--scenario", "-s", default="open", choices=list(SCENARIOS))
    r.add_argument("--global", "-g", dest="global_planner", default="astar")
    r.add_argument(
        "--local", "-l", dest="local_planner", default="dwa", choices=list(LOCAL_PLANNERS)
    )
    r.add_argument("--start", help="переопределить старт (S1..S3)")
    r.add_argument("--goal", help="переопределить цель (G1..G4)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--max-time", type=float, default=150.0, help="лимит времени симуляции, с")
    r.add_argument(
        "--viewer", action="store_true", help="окно MuJoCo (на macOS: uv run mjpython -m mpp ...)"
    )
    r.add_argument("--speed", type=float, default=1.0, help="скорость воспроизведения во вьюере")
    r.add_argument("--gif", action="store_true", help="записать анимацию episode.gif (без окна)")
    r.add_argument("--camera", default="overview", choices=["overview", "chase", "rgbd"])
    r.add_argument("--out", help="папка для результатов")
    r.add_argument(
        "--sensor",
        default="lidar",
        choices=["lidar", "depth"],
        help="lidar = LDS-01 360°, depth = RGB-D камера RealSense D435 вместо лидара",
    )
    _add_nav_args(r)
    r.set_defaults(func=cmd_run)

    b = sub.add_parser("bench", help="все сочетания планировщиков на всех сценариях")
    b.add_argument("--scenarios", default=",".join(SCENARIOS))
    b.add_argument("--global", dest="global_planners", default=",".join(GLOBAL_PLANNERS))
    b.add_argument("--local", dest="local_planners", default=",".join(LOCAL_PLANNERS))
    b.add_argument("--seeds", type=int, default=1, help="число повторов с разными seed")
    b.add_argument("--jobs", "-j", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    b.add_argument("--no-plots", action="store_true")
    b.add_argument("--out")
    b.add_argument("--sensors", default="lidar", help="lidar,depth — сравнить датчики")
    _add_nav_args(b)
    b.set_defaults(func=cmd_bench)

    pl = sub.add_parser("plan", help="сравнить только глобальные планировщики (без симуляции)")
    pl.add_argument("--planners", default=",".join(GLOBAL_PLANNERS))
    pl.add_argument("--start", default="S3")
    pl.add_argument("--goal", default="G1")
    pl.add_argument("--all-pairs", action="store_true", help="все пары старт/цель")
    pl.add_argument("--blocked", action="store_true", help="с коробками на карте")
    pl.add_argument("--seeds", type=int, default=5)
    pl.add_argument(
        "--time-limit", type=float, default=1.0, help="бюджет сэмплирующих планировщиков, с"
    )
    pl.add_argument("--out")
    pl.set_defaults(func=cmd_plan)

    m = sub.add_parser("map", help="нарисовать карту")
    m.add_argument("--out", default="results/map.png")
    m.set_defaults(func=cmd_map)

    sc = sub.add_parser("scene", help="открыть сцену в MuJoCo или выгрузить MJCF")
    sc.add_argument("--scenario", "-s", default="blocked", choices=list(SCENARIOS))
    sc.add_argument("--export", help="путь для scene.xml")
    sc.set_defaults(func=cmd_scene)

    sr = sub.add_parser("show-robot", help="открыть модель TurtleBot3 в MuJoCo")
    sr.set_defaults(func=cmd_robot)

    ls = sub.add_parser("list", help="сценарии, планировщики, старты и цели")
    ls.set_defaults(func=cmd_list)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
