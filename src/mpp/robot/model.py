"""TurtleBot3 Burger parameters and differential-drive conversions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MJCF_PATH = Path(__file__).parent / "turtlebot3_burger" / "turtlebot3_burger.xml"


@dataclass(frozen=True)
class RobotParams:
    """Values follow the ROBOTIS e-Manual and the Nav2 TurtleBot3 configuration."""

    wheel_radius: float = 0.033
    wheel_separation: float = 0.160
    radius: float = 0.105  # circumscribed radius of the footprint, used for collision checks
    max_v: float = 0.22  # m/s
    min_v: float = -0.10  # reversing is allowed but slow
    max_w: float = 2.0  # rad/s (hardware limit is 2.84)
    acc_v: float = 1.0  # m/s^2
    acc_w: float = 3.0  # rad/s^2
    lidar_range: tuple[float, float] = (0.12, 3.5)  # LDS-01
    lidar_rays: int = 360
    lidar_rate: float = 5.0  # Hz

    def wheel_speeds(self, v: float, w: float) -> tuple[float, float]:
        """Body twist -> (left, right) wheel angular velocity in rad/s."""
        half = 0.5 * self.wheel_separation
        return (v - w * half) / self.wheel_radius, (v + w * half) / self.wheel_radius

    def twist(self, w_left: float, w_right: float) -> tuple[float, float]:
        """(left, right) wheel angular velocity -> body twist (v, w)."""
        r = self.wheel_radius
        return 0.5 * r * (w_left + w_right), r * (w_right - w_left) / self.wheel_separation


BURGER = RobotParams()


@dataclass(frozen=True)
class DepthCameraParams:
    """Intel RealSense D435 depth stream (datasheet values), decimated to a coarse pixel grid.

    The depth error of a stereo camera grows with the square of the distance; ~2 cm at 2 m
    matches the D435's < 2 % RMS spec.
    """

    hfov_deg: float = 87.0
    vfov_deg: float = 58.0
    cols: int = 96
    rows: int = 36
    min_range: float = 0.3  # closer than this the stereo matcher gives no depth
    max_range: float = 3.0  # usable range indoors
    noise_coeff: float = 0.005  # sigma_z = noise_coeff * z^2
    # points between these heights are obstacles (Nav2: min/max_obstacle_height)
    min_obstacle_height: float = 0.05
    max_obstacle_height: float = 0.40
    mount: tuple[float, float, float] = (0.0, 0.0, 0.175)  # on the top plate, where the LDS sits


D435 = DepthCameraParams()
