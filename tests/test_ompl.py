import numpy as np
import pytest
from conftest import path_is_valid

from mpp.planning.global_planners import make_global_planner

pytest.importorskip("ompl")


@pytest.mark.parametrize("planner", ["RRTConnect", "BITstar"])
def test_ompl_reference_planner(wall_map, planner):
    res = make_global_planner(f"ompl:{planner}", seed=0, time_limit=0.3).plan(
        wall_map, np.array([1.0, 1.0]), np.array([5.0, 1.0])
    )
    assert res.success and path_is_valid(wall_map, res.path)
