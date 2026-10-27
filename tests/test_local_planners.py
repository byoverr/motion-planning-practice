import numpy as np
import pytest

from mpp.mapping.costmap import Costmap
from mpp.planning.local_planners import LOCAL_PLANNERS, make_local_planner


def drive(lp, cm, path, steps=400):
    """Kinematic closed loop: returns the poses visited."""
    lp.set_path(path)
    pose, vel = np.array([path[0, 0], path[0, 1], 0.0]), (0.0, 0.0)
    poses = [pose.copy()]
    for _ in range(steps):
        r = lp.compute(tuple(pose), vel, cm)
        vel = (r.v, r.w)
        pose = pose + np.array(
            [r.v * 0.1 * np.cos(pose[2]), r.v * 0.1 * np.sin(pose[2]), r.w * 0.1]
        )
        poses.append(pose.copy())
        if np.linalg.norm(pose[:2] - path[-1]) < 0.1:
            break
    return np.array(poses)


def empty_room() -> Costmap:
    cm = Costmap(6.0, 3.0)
    cm.fill_rect(0, 0, 6, 0.05)
    cm.fill_rect(0, 2.95, 6, 3)
    cm.update()
    return cm


@pytest.mark.parametrize("name", list(LOCAL_PLANNERS))
def test_follows_straight_path(name):
    cm = empty_room()
    path = np.array([[0.5, 1.5], [5.0, 1.5]])
    poses = drive(make_local_planner(name), cm, path)
    assert np.linalg.norm(poses[-1, :2] - path[-1]) < 0.12
    assert np.abs(poses[:, 1] - 1.5).max() < 0.15
    assert len(poses) < 300  # 4.5 m at <= 0.22 m/s is ~21 s = 210 steps


@pytest.mark.parametrize("name", ["dwa", "mppi"])
def test_avoiding_planners_drive_around_box_on_path(name):
    # the box overlaps the path but leaves room on one side (as in the "bypass" scenario);
    # a box dead in the middle of the path is a local minimum for DWA, see docs/planners.md
    cm = empty_room()
    cm.fill_rect(2.4, 1.05, 2.8, 1.45, layer="obstacles")
    cm.update()
    path = np.array([[0.5, 1.5], [5.0, 1.5]])
    poses = drive(make_local_planner(name), cm, path, steps=500)
    assert np.linalg.norm(poses[-1, :2] - path[-1]) < 0.12
    assert cm.clearance_at(poses[:, :2]).min() > 0.105  # never touched the box


def test_pure_pursuit_stops_in_front_of_box():
    cm = empty_room()
    cm.fill_rect(2.4, 1.3, 2.8, 1.7, layer="obstacles")
    cm.update()
    lp = make_local_planner("rpp")
    lp.set_path(np.array([[0.5, 1.5], [5.0, 1.5]]))
    r = lp.compute((2.15, 1.5, 0.0), (0.2, 0.0), cm)
    assert r.status == "blocked" and r.v == 0.0
    assert not lp.avoids_obstacles
