"""Interactive MuJoCo viewer and offscreen GIF recording for an episode.

On macOS the interactive viewer must run under ``mjpython`` (it ships with the mujoco
package):  ``uv run mjpython -m mpp run --viewer``.
"""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
import numpy as np

from mpp.navigation.navigator import Navigator
from mpp.runner import EpisodeConfig, EpisodeResult, run_episode
from mpp.sim.sim import Simulation
from mpp.viz.overlay import draw_navigation


def _trace(sim: Simulation, store: list) -> np.ndarray:
    x, y, _ = sim.pose()
    if not store or np.hypot(x - store[-1][0], y - store[-1][1]) > 0.03:
        store.append((x, y))
    return np.array(store)


def run_with_viewer(cfg: EpisodeConfig, speed: float = 1.0) -> EpisodeResult:
    import mujoco.viewer

    state: dict = {"trace": []}

    def on_step(sim: Simulation, nav: Navigator):
        v = state.get("viewer")
        if v is None:
            v = mujoco.viewer.launch_passive(sim.model, sim.data)
            with v.lock():
                v.cam.lookat[:] = [sim.layout.width / 2, sim.layout.height / 2, 0]
                v.cam.distance = 10.0
                v.cam.elevation = -65
                v.cam.azimuth = 90
            state["viewer"] = v
            state["t0"] = time.perf_counter() - sim.time / speed
        if not v.is_running():
            return False
        with v.lock():
            v.user_scn.ngeom = 0
            draw_navigation(v.user_scn, nav, _trace(sim, state["trace"]))
        v.sync()
        lag = sim.time / speed - (time.perf_counter() - state["t0"])
        if lag > 0:
            time.sleep(lag)
        return True

    result = run_episode(cfg, on_step)
    v = state.get("viewer")
    if v is not None:
        # keep the window open at the final state until the user closes it
        print(f"эпизод завершён: {result.status}, {result.sim_time:.1f} с; закройте окно MuJoCo")
        while v.is_running():
            v.sync()
            time.sleep(0.05)
    return result


def run_with_recording(
    cfg: EpisodeConfig,
    out: Path,
    camera: str = "overview",
    fps: int = 8,
    speedup: float = 5.0,
    size: tuple[int, int] = (640, 400),
) -> EpisodeResult:
    """Render the episode offscreen to an animated GIF (`speedup` x real time)."""
    from PIL import Image

    state: dict = {"frames": [], "trace": [], "next": 0.0}
    period = speedup / fps

    def on_step(sim: Simulation, nav: Navigator):
        if sim.time + 1e-9 < state["next"]:
            _trace(sim, state["trace"])
            return True
        state["next"] += period
        r = state.get("renderer")
        if r is None:
            sim.model.light_castshadow[:] = 0  # software GL renders shadow maps with acne
            r = state["renderer"] = mujoco.Renderer(sim.model, size[1], size[0])
        r.update_scene(sim.data, camera)
        draw_navigation(r.scene, nav, _trace(sim, state["trace"]), candidates=camera == "chase")
        state["frames"].append(Image.fromarray(r.render()))
        return True

    result = run_episode(cfg, on_step)
    frames = state["frames"]
    if frames:
        frames += [frames[-1]] * fps  # hold the last frame for a second
        out.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            out, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0
        )
    return result
