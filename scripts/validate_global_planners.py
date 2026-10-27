"""Compare our RRT-Connect, BIT* and AIT* with the original C++ implementations in OMPL.

Same map, same validity check (clearance >= inscribed + margin), same time budgets.
OMPL 1.7 runs in a separate, isolated environment (see ompl_reference_worker.py).

    uv run python scripts/validate_global_planners.py            # ~5 minutes
    uv run python scripts/validate_global_planners.py --quick    # ~1 minute

Writes docs/validation.md and docs/img/validation_lengths.png.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from mpp.mapping.costmap import Costmap
from mpp.planning.global_planners import make_global_planner
from mpp.world.layout import warehouse
from mpp.world.scenarios import get_scenario

ROOT = Path(__file__).resolve().parents[1]
OURS = {"RRTConnect": "rrt_connect", "BITstar": "bit_star", "AITstar": "ait_star"}
MARGIN = 0.04


def problems() -> dict[str, tuple[Costmap, np.ndarray, np.ndarray]]:
    lay = warehouse()
    open_cm = Costmap.from_layout(lay)
    blocked = Costmap.from_layout(lay)
    for b in get_scenario("blocked").boxes:
        blocked.fill_rect(
            b.x - b.size[0] / 2,
            b.y - b.size[1] / 2,
            b.x + b.size[0] / 2,
            b.y + b.size[1] / 2,
            layer="obstacles",
        )
    blocked.update()
    p = lambda n: lay.pose(n).xy  # noqa: E731
    return {
        "S3→G1": (open_cm, p("S3"), p("G1")),
        "S1→G4": (open_cm, p("S1"), p("G4")),
        "S2→G2 (коридор закрыт)": (blocked, p("S2"), p("G2")),
    }


def run_ours(job: dict) -> dict:
    cm, start, goal = problems()[job["problem"]]
    kw = {"seed": job["seed"]}
    if job["planner"] != "RRTConnect":
        kw["time_limit"] = job["time"]
    else:
        kw["simplify"] = False
    r = make_global_planner(OURS[job["planner"]], **kw).plan(cm, start, goal)
    length = r.info.get("raw_length", float("inf")) if r.success else float("inf")
    return {
        **job,
        "impl": "mpp",
        "success": r.success,
        "length": length,
        "wall": r.time,
        "checks": r.info.get("collision_checks"),
    }


def run_ompl(jobs: list[dict], n_proc: int) -> list[dict]:
    """Split jobs per problem and per process; each worker gets its own costmap file."""
    out: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        procs = []
        chunks = [jobs[i::n_proc] for i in range(n_proc)]
        for k, chunk in enumerate(chunks):
            by_problem: dict[str, list[dict]] = {}
            for j in chunk:
                by_problem.setdefault(j["problem"], []).append(j)
            for pi, (name, pjobs) in enumerate(by_problem.items()):
                cm, start, goal = problems()[name]
                npz = Path(tmp) / f"p{k}_{pi}.npz"
                np.savez(
                    npz,
                    clearance=cm.clearance,
                    res=cm.res,
                    origin=cm.origin,
                    inscribed=cm.inscribed,
                    margin=MARGIN,
                )
                for j in pjobs:
                    j["start"], j["goal"] = start.tolist(), goal.tolist()
                jf, of = Path(tmp) / f"j{k}_{pi}.json", Path(tmp) / f"o{k}_{pi}.json"
                jf.write_text(json.dumps(pjobs))
                cmd = [
                    "uv",
                    "run",
                    "--isolated",
                    "--no-project",
                    "--python",
                    "3.12",
                    "--with",
                    "ompl==1.7.0",
                    "--with",
                    "numpy",
                    "python",
                    str(ROOT / "scripts" / "ompl_reference_worker.py"),
                    str(npz),
                    str(jf),
                    str(of),
                ]
                procs.append((subprocess.Popen(cmd, stdout=subprocess.DEVNULL), of))
        for p, of in procs:
            p.wait()
            for r in json.loads(of.read_text()):
                r.pop("start"), r.pop("goal")
                out.append({**r, "impl": "OMPL"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    seeds = range(4 if a.quick else 10)
    budgets = [0.5, 2.0] if a.quick else [0.5, 1.0, 3.0]
    jobs = []
    for name in problems():
        for s in seeds:
            jobs.append({"problem": name, "planner": "RRTConnect", "time": 2.0, "seed": s})
            for planner in ("BITstar", "AITstar"):
                for t in budgets:
                    jobs.append({"problem": name, "planner": planner, "time": t, "seed": s})
    # best known cost per problem: OMPL BIT* with a long budget
    long_budget = 10.0 if a.quick else 30.0
    refs = [
        {"problem": n, "planner": pl, "time": long_budget, "seed": 0}
        for n in problems()
        for pl in ("BITstar", "AITstar")
    ]

    print(f"OMPL: {len(jobs) + len(refs)} runs ...", flush=True)
    ompl = run_ompl([dict(j) for j in jobs] + refs, a.jobs)
    print(f"mpp: {len(jobs)} runs ...", flush=True)
    with mp.get_context("spawn").Pool(a.jobs) as pool:
        ours = pool.map(run_ours, jobs)

    best = {}
    for r in ompl + ours:
        if r["success"]:
            best[r["problem"]] = min(best.get(r["problem"], np.inf), r["length"])
    rows = [r for r in ompl if r["time"] < 10] + ours
    write_report(rows, best, budgets, [r for r in ompl if r["time"] >= 10])


def write_report(rows: list[dict], best: dict, budgets: list[float], long_runs: list[dict]) -> None:
    import matplotlib
    import matplotlib.ticker

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mpp.viz.plot2d import SERIES

    def cell(rs: list[dict], key: str = "length") -> str:
        ok = [r[key] for r in rs if r["success"]]
        if not ok:
            return f"0/{len(rs)}"
        med = float(np.median(ok))
        return f"{len(ok)}/{len(rs)} · {med:.2f}"

    lines = [
        "# Проверка глобальных планировщиков против OMPL",
        "",
        "Сгенерировано `scripts/validate_global_planners.py`. Одинаковая карта, одинаковая проверка "
        f"столкновений (клиренс ≥ радиус + 0.03 + запас {MARGIN} м), одинаковый бюджет времени. "
        "OMPL 1.7 (C++, Python-обёртка) — последняя версия, где из Python доступен AIT\\*.",
        "",
        "Ячейка: успехи/запуски · медиана длины пути, м. В скобках — отношение к лучшему "
        "известному пути (OMPL BIT\\* с бюджетом 30 с и все остальные запуски).",
        "",
    ]
    problems_ = list(dict.fromkeys(r["problem"] for r in rows))
    header = "| задача | планировщик | бюджет, с | наш | OMPL |"
    lines += [header, "|---|---|---|---|---|"]
    for prob in problems_:
        for planner in ("RRTConnect", "BITstar", "AITstar"):
            ts = [2.0] if planner == "RRTConnect" else budgets
            for t in ts:
                sel = [
                    r
                    for r in rows
                    if r["problem"] == prob and r["planner"] == planner and r["time"] == t
                ]
                ours = [r for r in sel if r["impl"] == "mpp"]
                ref = [r for r in sel if r["impl"] == "OMPL"]

                def fmt(rs):
                    ok = [r["length"] for r in rs if r["success"]]
                    return cell(rs) + (f" ({np.median(ok) / best[prob]:.3f})" if ok else "")

                label = "первое решение" if planner == "RRTConnect" else f"{t:g}"
                lines.append(f"| {prob} | {planner} | {label} | {fmt(ours)} | {fmt(ref)} |")
    lines += ["", "Длинный бюджет, OMPL (один запуск):", ""]
    for r in long_runs:
        ratio = r["length"] / best[r["problem"]] if r["success"] else float("nan")
        lines.append(
            f"* {r['problem']}, {r['planner']}, {r['time']:g} с: {r['length']:.3f} м ({ratio:.3f})"
        )
    lines += [
        "",
        "Лучшие известные длины: " + ", ".join(f"{k}: {v:.3f} м" for k, v in best.items()),
        "",
    ]
    out = ROOT / "docs" / "validation.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    fig, axes = plt.subplots(1, len(problems_), figsize=(4.6 * len(problems_), 3.8), sharey=True)
    for ax, prob in zip(np.atleast_1d(axes), problems_, strict=True):
        for k, (planner, impl) in enumerate(
            [("BITstar", "mpp"), ("BITstar", "OMPL"), ("AITstar", "mpp"), ("AITstar", "OMPL")]
        ):
            med = []
            for t in budgets:
                ok = [
                    r["length"]
                    for r in rows
                    if r["problem"] == prob
                    and r["planner"] == planner
                    and r["impl"] == impl
                    and r["time"] == t
                    and r["success"]
                ]
                med.append(np.median(ok) / best[prob] if ok else np.nan)
            color = SERIES[0] if planner == "BITstar" else SERIES[1]
            ax.plot(
                budgets,
                med,
                marker="o",
                ms=8,
                lw=2,
                color=color,
                ls="-" if impl == "mpp" else "--",
                label=f"{planner} ({impl})",
            )
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xticks(budgets, [f"{t:g}" for t in budgets])
        ax.set_title(prob, fontsize=10, loc="left")
        ax.set_xlabel("бюджет времени, с")
        ax.grid(True, color="#e4e3df")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("медиана длины / лучшая известная")
    np.atleast_1d(axes)[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "img" / "validation_lengths.png", dpi=130)


if __name__ == "__main__":
    sys.exit(main())
