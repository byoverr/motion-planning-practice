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


@pytest.mark.parametrize("name", ["rpp", "graceful", "vector_pursuit"])
def test_path_trackers_stop_in_front_of_box(name):
    cm = empty_room()
    cm.fill_rect(2.4, 1.3, 2.8, 1.7, layer="obstacles")
    cm.update()
    lp = make_local_planner(name)
    lp.set_path(np.array([[0.5, 1.5], [5.0, 1.5]]))
    r = lp.compute((2.15, 1.5, 0.0), (0.2, 0.0), cm)
    assert r.status == "blocked" and r.v == 0.0
    assert not lp.avoids_obstacles


def test_vector_pursuit_radius_goes_straight_on_straight_path():
    from mpp.planning.local_planners.vector_pursuit import VectorPursuit
    from mpp.robot.model import BURGER

    vp = VectorPursuit(BURGER)
    assert vp.turning_radius(0.6, 0.0, 0.0) == float("inf")
    # point to the left with the path heading also turning left -> finite left turn
    assert 0.25 <= vp.turning_radius(0.5, 0.2, 0.5) < 2.0


def test_graceful_control_law_matches_nav2_formula():
    """kappa = -1/r * (k_delta*(delta - atan(-k_phi*phi))
    + (1 + k_phi/(1+(k_phi*phi)^2))*sin(delta)), see smooth_control_law.cpp."""
    import math

    from mpp.planning.local_planners.graceful import GracefulController
    from mpp.robot.model import BURGER

    gc = GracefulController(BURGER)
    r, phi, delta = 0.8, 0.3, -0.2
    expected = (
        -1
        / r
        * (
            1.0 * (delta - math.atan(-2.0 * phi))
            + (1 + 2.0 / (1 + (2.0 * phi) ** 2)) * math.sin(delta)
        )
    )
    assert gc.curvature(r, phi, delta) == pytest.approx(expected)
    v, w = gc.control((1.0, 0.0, 0.0), (0.0, 0.0, 0.0))  # target straight ahead
    assert w == pytest.approx(0.0, abs=1e-9) and v > 0


def test_motion_memory_proposes_remembered_commands():
    from mpp.planning.local_planners.memory import MotionMemory

    rows = np.array([[0.1 * i, 1.5, 0.0, 0.2, 0.05 * i] for i in range(30)])
    mem = MotionMemory(rows)
    seq = mem.lookup((1.02, 1.52, 0.1), steps=5)
    assert seq is not None and np.allclose(seq[:, 1], 0.05 * np.arange(10, 15))
    assert mem.lookup((1.0, 2.5, 0.0), steps=5) is None  # too far from the remembered run
    assert mem.lookup((1.0, 1.5, 2.0), steps=5) is None  # wrong heading


def test_mppi_with_memory_and_few_samples_follows_path():
    from mpp.planning.local_planners.memory import MotionMemory

    cm = empty_room()
    path = np.array([[0.5, 1.5], [5.0, 1.5]])
    teacher = drive(make_local_planner("mppi"), cm, path)
    rows = np.column_stack([teacher[:-1], np.zeros((len(teacher) - 1, 2))])
    rows[:, 3] = np.hypot(*np.diff(teacher[:, :2], axis=0).T) / 0.1
    rows[:, 4] = np.diff(teacher[:, 2]) / 0.1
    student = make_local_planner("mppi", samples=16, memory=MotionMemory(rows))
    poses = drive(student, cm, path)
    assert np.linalg.norm(poses[-1, :2] - path[-1]) < 0.12
    assert student.memory_hits > 0


def test_mppi_with_pursuit_prior_and_few_samples_follows_path():
    cm = empty_room()
    path = np.array([[0.5, 1.5], [5.0, 1.5]])
    poses = drive(make_local_planner("mppi", samples=16, ancillary="pursuit"), cm, path)
    assert np.linalg.norm(poses[-1, :2] - path[-1]) < 0.12
