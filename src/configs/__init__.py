"""配置模块"""

from src.configs.base import BaseConfig
from src.configs.navigation import Navigation2DConfig

# 环境名称到配置类的映射
CONFIG_REGISTRY = {
    "Navigation2DObstacle-v1": Navigation2DConfig,
}


def get_config(env_name: str) -> BaseConfig:
    """
    根据环境名称获取配置
    
    Args:
        env_name: 环境名称
        
    Returns:
        配置实例
        
    Raises:
        ValueError: 未知环境
    """
    if env_name not in CONFIG_REGISTRY:
        available = list(CONFIG_REGISTRY.keys())
        raise ValueError(f"Unknown environment: {env_name}. Available: {available}")
    return CONFIG_REGISTRY[env_name]()


__all__ = ["BaseConfig", "Navigation2DConfig", "get_config"]
