from asset.continuous_mountain_car import Continuous_MountainCarEnv
from asset.pendulum import PendulumEnv
from asset.navigation_2d import Navigation2DEnv, Navigation2DSimpleEnv

from gymnasium.envs.registration import register

register(
    id="MountainCarContinuous-h-v1",
    entry_point="asset:Continuous_MountainCarEnv",
)

register(
    id="Pendulum-h-v1",
    entry_point="asset:PendulumEnv",
)

register(
    id="Navigation2DSimple-v1",
    entry_point="asset:Navigation2DSimpleEnv",
)

register(
    id="Navigation2DObstacle-v1",
    entry_point="asset:Navigation2DEnv",
)
