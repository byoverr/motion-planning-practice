import numpy as np
import pytest
from conftest import path_is_valid

from mpp.geometry import path_length
from mpp.planning.global_planners import GLOBAL_PLANNERS, make_global_planner

START, GOAL = np.array([1.0, 1.0]), np.array([5.0, 1.0])


@pytest.mark.parametrize("name", list(GLOBAL_PLANNERS))
def test_planner_finds_valid_path_through_gap(wall_map, name):
    kw = {} if name == "astar" else {"time_limit": 0.5}
    res = make_global_planner(name, seed=3, **kw).plan(wall_map, START, GOAL)
    assert res.success, res.info
    assert np.allclose(res.path[0], START) and np.allclose(res.path[-1], GOAL)
    assert path_is_valid(wall_map, res.path)
    # the only way is through the gap at the top: at least up to y ~ 3.2 and back
    assert res.path[:, 1].max() > 3.2
    assert path_length(res.path) > 2 * (3.3 - 1.0)


@pytest.mark.parametrize("name", list(GLOBAL_PLANNERS))
def test_planner_fails_when_goal_is_cut_off(wall_map, name):
    cm = wall_map.copy()
    cm.fill_rect(2.9, 3.2, 3.1, 4.0)  # close the gap
    cm.update()
    kw = {} if name == "astar" else {"time_limit": 0.2}
    res = make_global_planner(name, seed=0, **kw).plan(cm, START, GOAL)
    assert not res.success


def test_astar_matches_dijkstra_cost_and_expands_less(wall_map):
    a = make_global_planner("astar", smooth=False).plan(wall_map, START, GOAL)
    d = make_global_planner("dijkstra", smooth=False).plan(wall_map, START, GOAL)
    assert a.info["grid_cost"] == pytest.approx(d.info["grid_cost"], rel=1e-9)
    assert a.iterations < d.iterations


def test_bit_star_and_ait_star_improve_on_rrt_connect(wall_map):
    lengths = {}
    for name in ("rrt_connect", "bit_star", "ait_star"):
        res = make_global_planner(name, seed=1, time_limit=0.8).plan(wall_map, START, GOAL)
        lengths[name] = res.info["raw_length"]
    # asymptotically optimal planners should beat the raw (unsimplified) RRT-Connect path
    assert lengths["bit_star"] < lengths["rrt_connect"]
    assert lengths["ait_star"] < lengths["rrt_connect"]


def test_start_in_collision_is_reported(wall_map):
    res = make_global_planner("astar").plan(wall_map, np.array([3.0, 1.0]), GOAL)
    assert not res.success and res.info["reason"] == "start in collision"
