import numpy as np
import pytest

from mpp.mapping.costmap import Costmap


@pytest.fixture
def wall_map() -> Costmap:
    """6 x 4 m room with a wall across the middle and a 0.6 m gap at the top."""
    cm = Costmap(6.0, 4.0, resolution=0.05)
    # outer walls
    cm.fill_rect(0, 0, 6, 0.05)
    cm.fill_rect(0, 3.95, 6, 4)
    cm.fill_rect(0, 0, 0.05, 4)
    cm.fill_rect(5.95, 0, 6, 4)
    # middle wall, gap y in [3.2, 3.8] (well, up to the outer wall at 3.95)
    cm.fill_rect(2.9, 0, 3.1, 3.2)
    cm.update()
    return cm


def path_is_valid(cm: Costmap, path: np.ndarray) -> bool:
    return all(cm.segment_free(a, b) for a, b in zip(path[:-1], path[1:], strict=True))
