"""自定义环境注册"""

from asset.navigation_2d import Navigation2DEnv
from gymnasium.envs.registration import register

register(
    id="Navigation2DObstacle-v1",
    entry_point="asset:Navigation2DEnv",
)

__all__ = ["Navigation2DEnv"]
