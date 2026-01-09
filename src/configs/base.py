"""基础配置类

定义所有环境配置的通用接口和默认值。
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class BaseConfig:
    """环境配置基类"""
    
    # ==================== 环境基本信息 ====================
    env_name: str = ""
    world_size: float = 10.0
    
    # ==================== 状态空间 ====================
    state_dim: int = 0
    state_bounds: np.ndarray = field(default_factory=lambda: np.array([]))
    state_offset: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # ==================== 动作空间 ====================
    action_dim: int = 0
    action_bounds: np.ndarray = field(default_factory=lambda: np.array([]))
    action_offset: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # ==================== 目标空间 ====================
    goal_dim: int = 2  # [x, y]
    goal_state: np.ndarray = field(default_factory=lambda: np.array([]))
    goal_threshold: float = 0.5
    
    # ==================== HAC 参数 ====================
    k_level: int = 2           # 层级数量
    H: int = 20                # 每层最大尝试次数
    lamda: float = 0.3         # 子目标测试概率
    
    # ==================== 学习参数 ====================
    gamma: float = 0.95
    lr: float = 0.001
    n_iter: int = 100
    batch_size: int = 128
    hidden_dim: int = 128
    
    # ==================== SAC 参数 ====================
    sac_alpha: float = 0.2
    sac_auto_entropy: bool = True
    
    # ==================== MPC 参数 ====================
    dt: float = 0.1
    max_v: float = 2.0
    max_omega: float = 2.0
    max_a_v: float = 1.0
    max_a_omega: float = 2.0
    damping_v: float = 0.95
    damping_omega: float = 0.9
    mpc_horizon: int = 10
    
    # ==================== 训练参数 ====================
    max_episodes: int = 1000
    random_seed: int = 42
    
    def __post_init__(self):
        """初始化后处理"""
        self._setup()
    
    def _setup(self):
        """子类需要实现的配置设置"""
        pass
    
    def get_subgoal_bounds(self) -> np.ndarray:
        """获取子目标边界 (世界坐标)"""
        return np.array([self.world_size / 2, self.world_size / 2])
    
    def get_subgoal_offset(self) -> np.ndarray:
        """获取子目标偏移"""
        return np.array([self.world_size / 2, self.world_size / 2])
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  env_name={self.env_name},\n"
            f"  state_dim={self.state_dim}, action_dim={self.action_dim}, goal_dim={self.goal_dim},\n"
            f"  k_level={self.k_level}, H={self.H}, gamma={self.gamma}, lr={self.lr}\n"
            f")"
        )
