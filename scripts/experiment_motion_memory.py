"""Does a motion memory from an earlier run let MPPI work with far fewer samples?

For every scenario: one "teacher" run with the default 400 samples is recorded; then MPPI
with few samples drives the same mission with and without that memory, over several seeds.

    uv run python scripts/experiment_motion_memory.py

Writes docs/research/motion_memory_results.md.
"""

from __future__ import annotations

import statistics as st
import time
from pathlib import Path

from mpp.planning.local_planners.memory import MotionMemory
from mpp.runner import EpisodeConfig, run_episode

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ["open", "bypass", "blocked"]
SEEDS = [1, 2, 3, 4, 5]
VARIANTS = [
    ("MPPI, 400 образцов", 400, False),
    ("MPPI, 32 образца", 32, False),
    ("MPPI, 16 образцов", 16, False),
    ("MPPI + память, 32 образца", 32, True),
    ("MPPI + память, 16 образцов", 16, True),
]


def main() -> None:
    rows = []
    t0 = time.time()
    for scn in SCENARIOS:
        teacher = run_episode(EpisodeConfig(scenario=scn, local_planner="mppi", seed=0))
        assert teacher.success, f"teacher run failed on {scn}"
        memory = MotionMemory.from_episode(teacher.trajectory)
        print(f"{scn}: teacher {teacher.sim_time:.1f} s, memory {memory.nbytes()} bytes")
        for name, samples, use_memory in VARIANTS:
            for seed in SEEDS:
                kw = {"samples": samples}
                if use_memory:
                    kw["memory"] = memory
                r = run_episode(
                    EpisodeConfig(scenario=scn, local_planner="mppi", seed=seed, local_kwargs=kw)
                )
                rows.append(
                    {
                        "scenario": scn,
                        "variant": name,
                        "seed": seed,
                        "r": r,
                        "memory_bytes": memory.nbytes() if use_memory else 0,
                    }
                )
                print(
                    f"  {name:28s} seed={seed} ok={r.success} t={r.sim_time:5.1f}s "
                    f"gap={r.min_gap:.3f} local={r.local_ms:.2f} ms coll={r.collisions}"
                )
    lines = [
        "# Эксперимент: память движения для MPPI",
        "",
        "Сгенерировано `scripts/experiment_motion_memory.py`. Для каждого сценария один "
        "«учительский» проезд MPPI с 400 образцами записывается в память, затем тот же маршрут "
        f"проезжается с разными настройками, seed {SEEDS[0]}–{SEEDS[-1]} (другой шум MPPI). "
        "Глобальный планировщик A\\*, лидар.",
        "",
        "Ячейка: доехали · среднее время до цели · минимальный зазор · время одного вызова "
        "MPPI (Python/NumPy, одно ядро) · касания.",
        "",
        "| вариант | " + " | ".join(SCENARIOS) + " |",
        "|---" * (len(SCENARIOS) + 1) + "|",
    ]
    for name, _, _ in VARIANTS:
        cells = []
        for scn in SCENARIOS:
            rs = [x["r"] for x in rows if x["scenario"] == scn and x["variant"] == name]
            ok = [r for r in rs if r.success]
            t = st.mean(r.sim_time for r in ok) if ok else float("nan")
            gap = min(r.min_gap for r in rs)
            ms = st.mean(r.local_ms for r in rs)
            coll = sum(r.collisions for r in rs)
            cells.append(
                f"{len(ok)}/{len(rs)} · {t:.1f} с · {100 * gap:.1f} см · {ms:.2f} мс · {coll}"
            )
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    sizes = sorted({x["memory_bytes"] for x in rows if x["memory_bytes"]})
    lines += [
        "",
        f"Размер памяти на маршрут: {sizes[0]}–{sizes[-1]} байт (float32).",
        f"Время эксперимента: {time.time() - t0:.0f} с.",
        "",
    ]
    out = ROOT / "docs" / "research" / "motion_memory_results.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
