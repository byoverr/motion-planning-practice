"""Run OMPL's C++ planners on an exported costmap (used by validate_global_planners.py).

Runs in its own environment with OMPL 1.7 (the last release whose Python bindings include
AIT*):  uv run --isolated --no-project --python 3.12 --with ompl==1.7.0 --with numpy \\
            python scripts/ompl_reference_worker.py problem.npz jobs.json out.json

The validity check reproduces mpp's: bilinear interpolation of the clearance field,
clearance >= inscribed + margin, with no margin near the start and goal.
"""

import json
import os
import sys
import time

import numpy as np
from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou


def main(problem_path: str, jobs_path: str, out_path: str) -> None:
    p = np.load(problem_path)
    clear, res, origin = p["clearance"], float(p["res"]), p["origin"]
    inscribed, margin = float(p["inscribed"]), float(p["margin"])
    ny, nx = clear.shape
    ou.setLogLevel(ou.LOG_WARN)
    jobs = json.load(open(jobs_path))
    out = []
    seeded = False
    for job in jobs:
        start, goal = np.array(job["start"]), np.array(job["goal"])
        near = 2 * margin + 0.1

        def clearance(x, y):
            fx = min(max((x - origin[0]) / res - 0.5, 0.0), nx - 1.0)
            fy = min(max((y - origin[1]) / res - 0.5, 0.0), ny - 1.0)
            x0, y0 = int(fx), int(fy)
            x1, y1 = min(x0 + 1, nx - 1), min(y0 + 1, ny - 1)
            tx, ty = fx - x0, fy - y0
            top = clear[y0, x0] * (1 - tx) + clear[y0, x1] * tx
            bot = clear[y1, x0] * (1 - tx) + clear[y1, x1] * tx
            return top * (1 - ty) + bot * ty

        def valid(s):
            x, y = s[0], s[1]
            d = min(np.hypot(x - start[0], y - start[1]), np.hypot(x - goal[0], y - goal[1]))
            return bool(clearance(x, y) >= inscribed + (0.0 if d < near else margin))

        space = ob.RealVectorStateSpace(2)
        b = ob.RealVectorBounds(2)
        b.setLow(0, float(origin[0]))
        b.setHigh(0, float(origin[0] + nx * res))
        b.setLow(1, float(origin[1]))
        b.setHigh(1, float(origin[1] + ny * res))
        space.setBounds(b)
        ss = og.SimpleSetup(space)
        ss.setStateValidityChecker(ob.StateValidityCheckerFn(valid))
        si = ss.getSpaceInformation()
        si.setStateValidityCheckingResolution(0.25 * res / max(nx * res, ny * res))
        s, g = ob.State(space), ob.State(space)
        s[0], s[1], g[0], g[1] = *map(float, start), *map(float, goal)
        ss.setStartAndGoalStates(s, g)
        ss.setOptimizationObjective(ob.PathLengthOptimizationObjective(si))
        planner = getattr(og, job["planner"])(si)
        if job["planner"] == "RRTConnect":
            planner.setRange(job.get("range", 0.35))
        ss.setPlanner(planner)
        if not seeded:
            ou.RNG.setSeed(1)
            seeded = True
        t0 = time.perf_counter()
        ss.solve(job["time"])
        dt = time.perf_counter() - t0
        ok = ss.haveExactSolutionPath()
        length = ss.getSolutionPath().length() if ok else float("inf")
        out.append({**job, "success": bool(ok), "length": length, "wall": dt})
    json.dump(out, open(out_path, "w"))
    sys.stdout.flush()
    os._exit(0)  # OMPL 1.7 Linux wheels crash in their atexit handlers


if __name__ == "__main__":
    main(*sys.argv[1:4])
