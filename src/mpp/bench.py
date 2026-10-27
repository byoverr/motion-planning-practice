"""Benchmarks: every global x local planner combination on every scenario, plus a
planning-only comparison of the global planners."""

from __future__ import annotations

import csv
import itertools
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np

from mpp.mapping.costmap import Costmap
from mpp.navigation.navigator import NavConfig
from mpp.planning.global_planners import make_global_planner
from mpp.runner import EpisodeConfig, run_episode
from mpp.viz.plot2d import SERIES  # noqa: E402
from mpp.world.layout import warehouse
from mpp.world.scenarios import get_scenario

INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _episode_job(args: tuple) -> dict:
    cfg, out_dir = args
    from mpp.viz.plot2d import plot_episode

    r = run_episode(cfg)
    if out_dir is not None:
        name = (
            f"{cfg.scenario}_{cfg.global_planner}_{cfg.local_planner}_{cfg.sensor}_s{cfg.seed}.png"
        )
        title = (
            f"{cfg.scenario}: {cfg.global_planner} + {cfg.local_planner} ({cfg.sensor}) — "
            f"{'успех' if r.success else r.status}, {r.sim_time:.0f} с, "
            f"перепланирований {r.replans}, объездов {r.bypasses}"
        )
        plot_episode(
            warehouse(),
            r.costmap,
            get_scenario(cfg.scenario),
            r.trajectory[:, :2],
            r.plans,
            r.events,
            Path(out_dir) / name,
            title,
            r.box_positions,
        )
    return r.row()


def run_matrix(
    scenarios: list[str],
    global_planners: list[str],
    local_planners: list[str],
    seeds: list[int],
    out: Path,
    jobs: int = 1,
    nav: NavConfig | None = None,
    plots: bool = True,
    sensors: list[str] | None = None,
) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    runs_dir = out / "runs" if plots else None
    if runs_dir:
        runs_dir.mkdir(exist_ok=True)
    cfgs = [
        EpisodeConfig(
            scenario=s,
            global_planner=g,
            local_planner=lp,
            seed=seed,
            nav=nav or NavConfig(),
            sensor=sensor,
        )
        for sensor, s, g, lp, seed in itertools.product(
            sensors or ["lidar"], scenarios, global_planners, local_planners, seeds
        )
    ]
    rows: list[dict] = []
    args = [(c, runs_dir) for c in cfgs]
    if jobs > 1:
        with mp.get_context("spawn").Pool(jobs) as pool:
            for i, row in enumerate(pool.imap(_episode_job, args), 1):
                rows.append(row)
                _progress(i, len(cfgs), row)
    else:
        for i, a in enumerate(args, 1):
            row = _episode_job(a)
            rows.append(row)
            _progress(i, len(cfgs), row)
    write_csv(rows, out / "results.csv")
    (out / "summary.md").write_text(summarize(rows), encoding="utf-8")
    sensors_ = list(dict.fromkeys(r["sensor"] for r in rows))
    for sensor in sensors_:
        name = "time_to_goal.png" if len(sensors_) == 1 else f"time_to_goal_{sensor}.png"
        plot_matrix([r for r in rows if r["sensor"] == sensor], out / name)
    if len(sensors_) > 1:
        plot_sensor_comparison(rows, out / "sensor_comparison.png")
    return rows


def _progress(i: int, n: int, row: dict) -> None:
    mark = "ok " if row["success"] else "FAIL"
    print(
        f"[{i:3d}/{n}] {mark} {row['sensor']:5s} {row['scenario']:8s} "
        f"{row['global_planner']:12s} {row['local_planner']:14s} seed={row['seed']} "
        f"t={row['sim_time']:6.1f}s "
        f"replans={row['replans']} bypasses={row['bypasses']} collisions={row['collisions']}",
        flush=True,
    )


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _group(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    return groups


def summarize(rows: list[dict]) -> str:
    """Markdown tables: one per scenario, rows = global planner, columns = local planner."""
    lines = ["# Результаты бенчмарка", ""]
    lines.append(
        "Ячейка: успехи/запуски · среднее время до цели (с, только успешные) · "
        "перепланирования · объезды · столкновения."
    )
    lines.append("")
    by_scn = _group(rows, ("sensor", "scenario"))
    locals_ = list(dict.fromkeys(r["local_planner"] for r in rows))
    globals_ = list(dict.fromkeys(r["global_planner"] for r in rows))
    sensor_names = {"lidar": "лидар LDS-01", "depth": "RGB-D камера D435"}
    for (sensor, scn), srows in by_scn.items():
        sc = get_scenario(scn)
        lines += [f"## {scn}: {sc.title} · {sensor_names.get(sensor, sensor)}", ""]
        lines += [sc.description, ""]
        lines.append("| глобальный \\ локальный | " + " | ".join(locals_) + " |")
        lines.append("|---" * (len(locals_) + 1) + "|")
        cells = _group(srows, ("global_planner", "local_planner"))
        for g in globals_:
            out = []
            for lp in locals_:
                rs = cells.get((g, lp), [])
                if not rs:
                    out.append("—")
                    continue
                ok = [r for r in rs if r["success"]]
                t = np.mean([r["sim_time"] for r in ok]) if ok else math.nan
                out.append(
                    f"{len(ok)}/{len(rs)} · {t:.1f} с · "
                    f"{np.mean([r['replans'] for r in rs]):.1f} · "
                    f"{np.mean([r['bypasses'] for r in rs]):.1f} · "
                    f"{sum(r['collisions'] for r in rs)}"
                )
            lines.append(f"| {g} | " + " | ".join(out) + " |")
        lines.append("")
    return "\n".join(lines)


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def plot_matrix(rows: list[dict], out: Path, metric: str = "sim_time") -> None:
    """Small multiples (one panel per scenario) of grouped bars: global planner on x,
    one bar per local planner. Failed combinations get a cross instead of a bar."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenarios = list(dict.fromkeys(r["scenario"] for r in rows))
    globals_ = list(dict.fromkeys(r["global_planner"] for r in rows))
    locals_ = list(dict.fromkeys(r["local_planner"] for r in rows))
    cells = _group(rows, ("scenario", "global_planner", "local_planner"))
    ncols = min(2, len(scenarios))
    nrows = math.ceil(len(scenarios) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6.4 * ncols, 3.9 * nrows), sharey=True, squeeze=False
    )
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.ravel()[len(scenarios) :]:
        ax.set_visible(False)
    width = 0.8 / max(len(locals_), 1)
    for ax, scn in zip(axes.ravel(), scenarios, strict=False):
        _style(ax)
        for j, lp in enumerate(locals_):
            xs, ys, fails = [], [], []
            for i, g in enumerate(globals_):
                rs = cells.get((scn, g, lp), [])
                ok = [r[metric] for r in rs if r["success"]]
                x = i + (j - (len(locals_) - 1) / 2) * width
                if ok:
                    xs.append(x)
                    ys.append(float(np.mean(ok)))
                elif rs:
                    fails.append(x)
            color = SERIES[j % len(SERIES)]
            ax.bar(xs, ys, width * 0.92, color=color, label=lp, edgecolor=SURFACE, linewidth=1)
            for x in fails:
                ax.text(x, 1, "×", ha="center", va="bottom", color=INK, fontsize=12)
        ax.set_xticks(range(len(globals_)), globals_, rotation=0)
        ax.set_xlim(-0.6, len(globals_) - 0.4)
        ax.set_title(f"{scn}: {get_scenario(scn).title}", fontsize=10, color=INK, loc="left")
    for row in axes:
        row[0].set_ylabel("время до цели, с (× — не доехал)", color=INK_2)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="локальный планировщик",
        loc="upper center",
        ncol=len(labels),
        frameon=False,
        fontsize=9,
        title_fontsize=9,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.12 / nrows))
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_sensor_comparison(rows: list[dict], out: Path) -> None:
    """Success rate and mean time to goal per local planner, lidar vs depth camera
    (all scenarios and global planners pooled)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    locals_ = list(dict.fromkeys(r["local_planner"] for r in rows))
    sensors_ = list(dict.fromkeys(r["sensor"] for r in rows))
    names = {"lidar": "лидар LDS-01 (360°)", "depth": "RGB-D D435 (87°)"}
    cells = _group(rows, ("local_planner", "sensor"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    fig.patch.set_facecolor(SURFACE)
    width = 0.8 / len(sensors_)
    for ax, metric in zip(axes, ("success", "sim_time"), strict=True):
        _style(ax)
        for j, sensor in enumerate(sensors_):
            ys = []
            for lp in locals_:
                rs = cells.get((lp, sensor), [])
                if metric == "success":
                    ys.append(100.0 * np.mean([r["success"] for r in rs]) if rs else np.nan)
                else:
                    ok = [r["sim_time"] for r in rs if r["success"]]
                    ys.append(float(np.mean(ok)) if ok else np.nan)
            xs = np.arange(len(locals_)) + (j - (len(sensors_) - 1) / 2) * width
            ax.bar(
                xs,
                ys,
                width * 0.92,
                color=SERIES[j],
                label=names.get(sensor, sensor),
                edgecolor=SURFACE,
                linewidth=1,
            )
        ax.set_xticks(range(len(locals_)), locals_, fontsize=8)
        ax.set_ylabel("доехали, %" if metric == "success" else "время до цели, с", color=INK_2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------- planning only


def plan_benchmark(
    planners: list[str],
    pairs: list[tuple[str, str]],
    seeds: list[int],
    out: Path,
    blocked: bool = False,
    planner_kwargs: dict | None = None,
) -> list[dict]:
    """Run the global planners alone on the static map (no simulation)."""
    from mpp.viz.plot2d import plot_plans

    out.mkdir(parents=True, exist_ok=True)
    layout = warehouse()
    cm = Costmap.from_layout(layout)
    if blocked:
        for scn in ("bypass", "blocked"):
            for b in get_scenario(scn).boxes:
                cm.fill_rect(
                    b.x - b.size[0] / 2,
                    b.y - b.size[1] / 2,
                    b.x + b.size[0] / 2,
                    b.y + b.size[1] / 2,
                    layer="obstacles",
                )
        cm.update()
    rows = []
    for s, g in pairs:
        plans, trees = {}, {}
        for name in planners:
            for seed in seeds:
                p = make_global_planner(name, seed=seed, **(planner_kwargs or {}).get(name, {}))
                r = p.plan(cm, layout.pose(s).xy, layout.pose(g).xy)
                rows.append(
                    {
                        "start": s,
                        "goal": g,
                        "planner": name,
                        "seed": seed,
                        "success": r.success,
                        "length": round(r.length, 3),
                        "time": round(r.time, 4),
                        "first_solution_time": r.first_solution_time,
                        "iterations": r.iterations,
                        "collision_checks": r.info.get("collision_checks", ""),
                    }
                )
                if seed == seeds[0]:
                    plans[name] = r.path
                    trees[name] = r.tree
                print(
                    f"{s}->{g} {name:12s} seed={seed} ok={r.success} "
                    f"len={r.length:6.2f} t={r.time * 1000:7.1f} ms",
                    flush=True,
                )
        plot_plans(layout, cm, plans, out / f"plans_{s}_{g}.png", f"{s} → {g}", trees)
    write_csv(rows, out / "plans.csv")
    lines = [
        "| старт→цель | планировщик | успех | длина, м (ср.) | время, мс (ср.) |",
        "|---|---|---|---|---|",
    ]
    for (s, g, name), rs in _group(rows, ("start", "goal", "planner")).items():
        ok = [r for r in rs if r["success"]]
        ln = np.mean([r["length"] for r in ok]) if ok else math.nan
        tm = np.mean([r["time"] for r in rs]) * 1000
        lines.append(f"| {s}→{g} | {name} | {len(ok)}/{len(rs)} | {ln:.2f} | {tm:.0f} |")
    (out / "plans.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return rows
