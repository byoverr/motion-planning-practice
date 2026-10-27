import numpy as np

from mpp.sim.sim import Simulation
from mpp.world.layout import Pose2D, warehouse
from mpp.world.scenarios import get_scenario


def test_robot_drives_forward_and_turns():
    lay = warehouse()
    sim = Simulation(lay, None, Pose2D("s", 1.0, 3.425, 0.0))
    sim.set_cmd(0.2, 0.0)
    sim.step(2.0)
    x, y, yaw = sim.pose()
    assert 1.3 < x < 1.45 and abs(y - 3.425) < 0.01 and abs(yaw) < 0.02
    assert abs(sim.velocity()[0] - 0.2) < 0.02
    sim.set_cmd(0.0, 1.0)
    sim.step(1.0)
    assert 0.7 < sim.pose()[2] < 1.1  # wheel scrub makes turning slightly slower


def test_lidar_measures_distance_to_shelf():
    lay = warehouse()
    sim = Simulation(lay, None, Pose2D("s", 3.0, 3.425, 0.0))
    scan = sim.scan()
    # ray 0 points along +x through the narrow corridor to the east wall, 7 m away: no hit
    assert np.isinf(scan.ranges[0])
    # ray 90 (up) hits shelf L2 whose lower face is at y = 4.3
    up = scan.ranges[90]
    assert abs(up - (4.3 - 3.425)) < 0.02


def test_box_falls_when_robot_comes_close():
    lay = warehouse()
    sc = get_scenario("blocked")
    box = sc.boxes[0]
    sim = Simulation(lay, sc, Pose2D("s", box.x - box.trigger_distance - 0.3, box.y, 0.0))
    sim.step(0.5)
    assert not sim.dropped
    assert sim.data.xpos[sim.box_body[box.name]][2] > 1.0  # still hanging
    sim.set_cmd(0.22, 0.0)
    for _ in range(20):  # the trigger is checked after every step
        sim.step(0.1)
    sim.set_cmd(0.0, 0.0)
    sim.step(1.0)
    assert box.name in sim.dropped
    z = sim.data.xpos[sim.box_body[box.name]][2]
    assert abs(z - box.size[2] / 2) < 0.01  # resting on the floor


def test_depth_camera_sees_shelf_ahead_and_nothing_behind():
    lay = warehouse()
    sim = Simulation(lay, None, Pose2D("s", 2.2, 3.425, np.pi / 2), sensor="depth")
    scan = sim.scan()
    centre = scan.ranges[len(scan.ranges) // 2]
    # shelf L2 starts at y = 4.3; the camera sits 12.5 mm in front of the robot centre
    assert abs(centre - (4.3 - 3.425 - 0.0125)) < 0.03
    half, _ = sim.fov()
    rel = np.angle(np.exp(1j * (scan.angles - np.pi / 2)))
    assert np.abs(rel).max() <= half + 1e-6  # the camera only looks forward
    assert sim.last_depth.shape == (sim.camera.rows, sim.camera.cols)


def test_low_box_is_invisible_to_lidar_but_seen_by_camera():
    lay = warehouse()
    sc = get_scenario("low_box")
    box = sc.boxes[0]
    pose = Pose2D("s", box.x - 1.0, 6.05, 0.0)
    for sensor, visible in (("lidar", False), ("depth", True)):
        sim = Simulation(lay, sc, pose, sensor=sensor)
        sim.step(0.5)
        scan = sim.scan()
        ok = np.isfinite(scan.ranges)
        hits = scan.origin + scan.ranges[ok, None] * np.column_stack(
            [np.cos(scan.angles[ok]), np.sin(scan.angles[ok])]
        )
        # the box occupies y in [6.1, 6.5]; the shelf face right behind it is at y = 6.5
        half = box.size[1] / 2
        near_box = (np.abs(hits[:, 0] - box.x) < 0.25) & (np.abs(hits[:, 1] - box.y) < half - 0.03)
        assert near_box.any() == visible, sensor
