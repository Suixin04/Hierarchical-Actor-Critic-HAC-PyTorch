"""基础配置类

定义所有环境配置的通用接口和默认值。

使用 dataclass 提供类型安全和默认值支持。
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class BaseConfig:
    """环境配置基类"""
    
    # ==================== 环境基本信息 ====================
    env_name: str = ""
    
    # ==================== 状态空间 ====================
    state_dim: int = 0
    state_bounds: np.ndarray = field(default_factory=lambda: np.array([]))
    state_offset: np.ndarray = field(default_factory=lambda: np.array([]))
    state_clip_low: np.ndarray = field(default_factory=lambda: np.array([]))
    state_clip_high: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # ==================== 动作空间 ====================
    action_dim: int = 0
    action_bounds: np.ndarray = field(default_factory=lambda: np.array([]))
    action_offset: np.ndarray = field(default_factory=lambda: np.array([]))
    action_clip_low: np.ndarray = field(default_factory=lambda: np.array([]))
    action_clip_high: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # ==================== 目标空间 ====================
    goal_indices: Optional[List[int]] = None
    goal_state: np.ndarray = field(default_factory=lambda: np.array([]))
    goal_threshold: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # ==================== HAC 参数 ====================
    k_level: int = 2
    H: int = 20
    lamda: float = 0.3
    
    # ==================== 学习参数 ====================
    gamma: float = 0.95
    lr: float = 0.001
    n_iter: int = 100
    batch_size: int = 100
    hidden_dim: int = 64
    
    # ==================== SAC 参数 ====================
    sac_alpha: float = 0.2
    sac_auto_entropy: bool = True
    sac_target_entropy: Optional[float] = None
    sac_alpha_lr: Optional[float] = None
    
    # ==================== 探索噪声 ====================
    exploration_action_noise: np.ndarray = field(default_factory=lambda: np.array([]))
    exploration_state_noise: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # ==================== 训练参数 ====================
    max_episodes: int = 1000
    save_episode: int = 10
    random_seed: int = 0
    
    def __post_init__(self):
        """初始化后处理"""
        self._setup()
    
    def _setup(self):
        """子类需要实现的配置设置"""
        pass
    
    @property
    def effective_goal_dim(self) -> int:
        """有效的目标维度"""
        if self.goal_indices is not None:
            return len(self.goal_indices)
        return self.state_dim
    
    @property
    def subgoal_dim(self) -> int:
        """子目标维度"""
        return self.effective_goal_dim
    
    def extract_goal_from_state(self, state: np.ndarray) -> np.ndarray:
        """从状态提取目标部分"""
        if self.goal_indices is not None:
            return state[self.goal_indices]
        return state
    
    def get_subgoal_bounds(self, level: int = None) -> np.ndarray:
        """获取子目标边界"""
        if level == 1 and getattr(self, 'level1_use_polar', False):
            return self.level1_subgoal_bounds
        if self.goal_indices is not None:
            return self.state_bounds[self.goal_indices]
        return self.state_bounds
    
    def get_subgoal_offset(self, level: int = None) -> np.ndarray:
        """获取子目标偏移"""
        if level == 1 and getattr(self, 'level1_use_polar', False):
            return self.level1_subgoal_offset
        if self.goal_indices is not None:
            return self.state_offset[self.goal_indices]
        return self.state_offset
    
    def get_subgoal_clip_low(self, level: int = None) -> np.ndarray:
        """获取子目标裁剪下界"""
        if level == 1 and getattr(self, 'level1_use_polar', False):
            return np.array([self.subgoal_r_min, -self.subgoal_fov / 2])
        if self.goal_indices is not None:
            return self.state_clip_low[self.goal_indices]
        return self.state_clip_low
    
    def get_subgoal_clip_high(self, level: int = None) -> np.ndarray:
        """获取子目标裁剪上界"""
        if level == 1 and getattr(self, 'level1_use_polar', False):
            return np.array([self.subgoal_r_max, self.subgoal_fov / 2])
        if self.goal_indices is not None:
            return self.state_clip_high[self.goal_indices]
        return self.state_clip_high
    
    def get_save_directory(self) -> str:
        """获取保存目录"""
        return f"./preTrained/{self.env_name}/{self.k_level}level/"
    
    def get_filename(self) -> str:
        """获取文件名"""
        return f"HAC_{self.env_name}"
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  env_name={self.env_name},\n"
            f"  state_dim={self.state_dim}, action_dim={self.action_dim},\n"
            f"  goal_dim={self.effective_goal_dim}, goal_indices={self.goal_indices},\n"
            f"  k_level={self.k_level}, H={self.H}, gamma={self.gamma}, lr={self.lr}\n"
            f")"
        )
