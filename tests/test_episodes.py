"""End-to-end runs in MuJoCo (a few seconds each)."""

from mpp.runner import EpisodeConfig, run_episode


def test_bypass_is_handled_locally_by_dwa():
    r = run_episode(EpisodeConfig(scenario="bypass", global_planner="astar", local_planner="dwa"))
    assert r.success and r.collisions == 0
    assert r.bypasses >= 1 and r.replans == 0


def test_blocked_corridor_forces_global_replan():
    r = run_episode(EpisodeConfig(scenario="blocked", global_planner="astar", local_planner="mppi"))
    assert r.success and r.collisions == 0
    assert r.replans >= 1 and r.bypasses == 0
    assert r.distance > r.first_plan_length + 1.0  # the detour is longer than the first plan


def test_camera_robot_drives_around_low_box():
    r = run_episode(
        EpisodeConfig(
            scenario="low_box", global_planner="astar", local_planner="dwa", sensor="depth"
        )
    )
    assert r.success and r.collisions == 0
