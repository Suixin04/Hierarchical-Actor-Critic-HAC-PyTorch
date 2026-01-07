from configs.base_config import BaseConfig
from configs.navigation_2d_config import Navigation2DObstacleConfig

# 环境名称到配置类的映射 (只保留主要环境)
CONFIG_REGISTRY = {
    "Navigation2DObstacle-v1": Navigation2DObstacleConfig,
}

def get_config(env_name: str) -> BaseConfig:
    """根据环境名称获取配置"""
    if env_name not in CONFIG_REGISTRY:
        raise ValueError(f"Unknown environment: {env_name}. Available: {list(CONFIG_REGISTRY.keys())}")
    return CONFIG_REGISTRY[env_name]()

