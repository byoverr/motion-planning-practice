"""Does the motion memory survive a changing warehouse ("cache misses")?

The memory is recorded once on the original `bypass` layout. Then the box falls somewhere
else on every run: shifted along the aisle and against either shelf. MPPI with few samples
drives with and without that stale memory.

    uv run python scripts/experiment_memory_miss.py

Writes docs/research/memory_miss_results.md.
"""

from __future__ import annotations

import dataclasses
import statistics as st
from pathlib import Path

from mpp.planning.local_planners.memory import MotionMemory
from mpp.runner import EpisodeConfig, run_episode
from mpp.world.scenarios import SCENARIOS, get_scenario

ROOT = Path(__file__).resolve().parents[1]
# (x along the aisle, y: 1.12 = against the lower shelf like the original, 1.58 = upper shelf)
LAYOUTS = [(5.0, 1.12), (4.6, 1.12), (5.4, 1.12), (5.0, 1.58), (4.6, 1.58), (5.4, 1.58)]
VARIANTS = [
    ("MPPI, 400 образцов", 400, False),
    ("MPPI, 16 образцов", 16, False),
    ("MPPI + устаревшая память, 16", 16, True),
    ("MPPI + подсказка pure pursuit, 16 (без памяти)", 16, "pursuit"),
]
SEEDS = [1, 2, 3]


def main() -> None:
    base = get_scenario("bypass")
    teacher = run_episode(EpisodeConfig(scenario="bypass", local_planner="mppi", seed=0))
    memory = MotionMemory.from_episode(teacher.trajectory)
    rows = []
    for x, y in LAYOUTS:
        name = f"bypass_{x}_{y}"
        box = dataclasses.replace(base.boxes[0], x=x, y=y)
        SCENARIOS[name] = dataclasses.replace(base, name=name, boxes=(box,))
        same = (x, y) == (5.0, 1.12)
        for label, samples, use_mem in VARIANTS:
            for seed in SEEDS:
                kw = {"samples": samples}
                if use_mem == "pursuit":
                    kw["ancillary"] = "pursuit"
                elif use_mem:
                    kw["memory"] = memory
                holder = {}

                def grab(sim, nav, holder=holder):
                    holder["lp"] = nav.lp

                r = run_episode(
                    EpisodeConfig(scenario=name, local_planner="mppi", seed=seed, local_kwargs=kw),
                    on_step=grab,
                )
                lp = holder["lp"]
                hit = (
                    lp.memory_hits / max(r.trajectory.shape[0], 1)
                    if use_mem is True
                    else float("nan")
                )
                rows.append(dict(layout=(x, y), same=same, variant=label, r=r, hit=hit))
                print(
                    f"box=({x},{y}) {label:30s} seed={seed} ok={r.success} t={r.sim_time:5.1f} "
                    f"gap={r.min_gap:.3f} coll={r.collisions} replans={r.replans} hit={hit:.2f}"
                )

    def summary(sel):
        rs = [x["r"] for x in sel]
        ok = [r for r in rs if r.success]
        t = st.mean(r.sim_time for r in ok) if ok else float("nan")
        hits = [x["hit"] for x in sel if x["hit"] == x["hit"]]
        hit = f" · память в {100 * st.mean(hits):.0f} % циклов" if hits else ""
        return (
            f"{len(ok)}/{len(rs)} · {t:.1f} с · зазор ≥ {100 * min(r.min_gap for r in rs):.1f} см"
            f" · касаний {sum(r.collisions for r in rs)}{hit}"
        )

    lines = [
        "# Эксперимент: память при изменившейся обстановке",
        "",
        "Сгенерировано `scripts/experiment_memory_miss.py`. Память записана один раз на исходной "
        "раскладке `bypass` (коробка у нижнего стеллажа, x = 5.0). Затем коробка падает в другие "
        f"места: {len(LAYOUTS)} раскладок × seed {SEEDS[0]}–{SEEDS[-1]}.",
        "",
        "| вариант | та же раскладка | коробка сдвинута вдоль прохода | коробка у другого стеллажа |",
        "|---|---|---|---|",
    ]
    for label, _, _ in VARIANTS:
        sel = [x for x in rows if x["variant"] == label]
        same = [x for x in sel if x["same"]]
        shifted = [x for x in sel if not x["same"] and x["layout"][1] == 1.12]
        other = [x for x in sel if x["layout"][1] == 1.58]
        lines.append(f"| {label} | {summary(same)} | {summary(shifted)} | {summary(other)} |")
    lines.append("")
    out = ROOT / "docs" / "research" / "memory_miss_results.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
