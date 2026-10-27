import numpy as np

from mpp.mapping.costmap import Costmap
from mpp.world.layout import warehouse


def test_rasterisation_is_conservative_and_exact_on_cell_borders():
    cm = Costmap(2.0, 2.0, resolution=0.05)
    cm.fill_rect(0.5, 0.5, 1.0, 1.0)
    assert cm.static.sum() == 10 * 10  # 0.5 m / 0.05 m = 10 cells, no float spill-over
    cm.fill_rect(1.52, 1.52, 1.53, 1.53)  # tiny box inside one cell still marks it
    assert cm.static.sum() == 101


def test_clearance_is_distance_to_obstacle_surface():
    cm = Costmap(4.0, 4.0, resolution=0.05)
    cm.fill_rect(2.0, 0.0, 2.1, 4.0)
    cm.update()
    assert abs(float(cm.clearance_at(np.array([1.5, 2.0]))) - 0.5) < 0.03
    assert cm.is_free(np.array([1.5, 2.0]))
    assert not cm.is_free(np.array([1.9, 2.0]))  # 0.1 m from the wall < inscribed radius


def test_warehouse_narrow_corridor_is_passable_and_box_blocks_it():
    lay = warehouse()
    cm = Costmap.from_layout(lay)
    centre = np.array([5.0, 3.425])
    assert cm.is_free(centre)
    cm.fill_rect(4.82, 3.2, 5.18, 3.65, layer="obstacles")
    cm.update()
    assert not cm.is_free(centre)


def test_scan_marks_hits_and_clears_free_space():
    cm = Costmap(4.0, 4.0, resolution=0.05)
    cm.obstacles[40, 20] = True  # stale obstacle at (1.025, 2.025) between sensor and wall
    cm.update()
    origin = np.array([0.5, 2.0])
    angles = np.array([0.0, np.pi / 2])
    ranges = np.array([2.0, np.inf])  # hit at (2.5, 2.0); nothing upwards
    changed = cm.integrate_scan(origin, angles, ranges, max_range=3.5)
    assert changed
    ix, iy = cm.world_to_cell(np.array([2.5, 2.0]))
    assert cm.obstacles[iy, ix]
    assert not cm.obstacles[40, 20]  # cleared by the ray


def test_scan_ignores_invalid_readings():
    cm = Costmap(4.0, 4.0, resolution=0.05)
    changed = cm.integrate_scan(
        np.array([2.0, 2.0]), np.array([0.0]), np.array([np.nan]), max_range=3.5
    )
    assert not changed


def test_nearest_free_moves_out_of_inflation():
    cm = Costmap(4.0, 4.0, resolution=0.05)
    cm.fill_rect(2.0, 0.0, 2.1, 4.0)
    cm.update()
    p = cm.nearest_free(np.array([1.95, 2.0]))
    assert p is not None and cm.is_free(p) and p[0] < 1.95
