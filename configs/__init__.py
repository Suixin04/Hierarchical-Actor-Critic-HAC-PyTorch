from configs.base_config import BaseConfig
from configs.mountain_car_config import MountainCarConfig
from configs.pendulum_config import PendulumConfig
from configs.navigation_2d_config import (
    Navigation2DSimpleConfig,
    Navigation2DObstacleConfig,
)

# 环境名称到配置类的映射
CONFIG_REGISTRY = {
    "MountainCarContinuous-h-v1": MountainCarConfig,
    "Pendulum-h-v1": PendulumConfig,
    "Navigation2DSimple-v1": Navigation2DSimpleConfig,      # 无障碍物
    "Navigation2DObstacle-v1": Navigation2DObstacleConfig,  # 带障碍物避障
}

def get_config(env_name: str) -> BaseConfig:
    """根据环境名称获取配置"""
    if env_name not in CONFIG_REGISTRY:
        raise ValueError(f"Unknown environment: {env_name}. Available: {list(CONFIG_REGISTRY.keys())}")
    return CONFIG_REGISTRY[env_name]()

