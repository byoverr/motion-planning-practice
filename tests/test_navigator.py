import numpy as np

from mpp.mapping.costmap import Costmap
from mpp.navigation.navigator import Navigator
from mpp.planning.global_planners import make_global_planner
from mpp.planning.local_planners import make_local_planner


def aisle(width: float) -> Costmap:
    """Corridor along x between two shelves, y in [1, 1 + width]."""
    cm = Costmap(6.0, 3.0)
    cm.fill_rect(0, 0, 6, 1.0)
    cm.fill_rect(0, 1.0 + width, 6, 3.0)
    cm.update()
    return cm


def navigator(cm, local="dwa"):
    goal = np.array([5.5, 1.0 + 0.5 * (cm.ny * cm.res - 1.0 - 1.0) / 2])
    return Navigator(make_global_planner("astar"), make_local_planner(local), cm, goal)


def test_detour_exists_in_wide_aisle_but_not_in_narrow():
    wide = aisle(0.9)
    wide.fill_rect(2.8, 1.0, 3.2, 1.4, layer="obstacles")  # box against the lower shelf
    wide.update()
    nav = navigator(wide)
    corridor = np.column_stack([np.linspace(2.0, 4.0, 41), np.full(41, 1.45)])
    assert nav.detour_exists(np.array([2.0, 1.45]), np.array([4.0, 1.45]), corridor)

    narrow = aisle(0.45)
    narrow.fill_rect(2.8, 1.0, 3.2, 1.45, layer="obstacles")  # box fills the corridor
    narrow.update()
    nav = navigator(narrow)
    corridor = np.column_stack([np.linspace(2.0, 4.0, 41), np.full(41, 1.225)])
    assert not nav.detour_exists(np.array([2.0, 1.225]), np.array([4.0, 1.225]), corridor)


def test_blocked_path_triggers_bypass_or_replan():
    for local, expected in (("dwa", "bypass"), ("rpp", "replan")):
        cm = aisle(0.9)
        nav = Navigator(
            make_global_planner("astar"), make_local_planner(local), cm, np.array([5.5, 1.45])
        )
        assert nav.start((0.5, 1.45, 0.0), 0.0)
        cm.fill_rect(2.8, 1.0, 3.2, 1.4, layer="obstacles")
        cm.update()
        nav.tick((1.0, 1.45, 0.0), (0.0, 0.0), 2.0)
        kinds = [e.kind for e in nav.stats.events]
        assert expected in kinds, (local, kinds)
