"""
基础配置类，定义所有环境配置的通用接口
"""
from dataclasses import dataclass, field
from typing import List, Optional, Literal
import numpy as np


@dataclass
class BaseConfig:
    """环境配置基类"""
    
    # ==================== 环境基本信息 ====================
    env_name: str = ""                    # gymnasium 环境名称
    
    # ==================== 状态空间配置 ====================
    state_dim: int = 0                    # 状态维度
    state_bounds: np.ndarray = None       # 状态边界 (用于归一化)
    state_offset: np.ndarray = None       # 状态偏移
    state_clip_low: np.ndarray = None     # 状态裁剪下界
    state_clip_high: np.ndarray = None    # 状态裁剪上界
    
    # ==================== 动作空间配置 ====================
    action_dim: int = 0                   # 动作维度
    action_bounds: np.ndarray = None      # 动作边界
    action_offset: np.ndarray = None      # 动作偏移
    action_clip_low: np.ndarray = None    # 动作裁剪下界
    action_clip_high: np.ndarray = None   # 动作裁剪上界
    
    # ==================== 目标空间配置 ====================
    # 目标可以是状态的子集，通过索引指定
    goal_indices: List[int] = None        # 目标对应的状态索引，None表示使用全部状态
    goal_dim: int = 0                     # 目标维度 (自动根据goal_indices计算)
    goal_state: np.ndarray = None         # 最终目标状态
    goal_threshold: np.ndarray = None     # 目标达成阈值
    
    # ==================== HAC 算法参数 ====================
    k_level: int = 2                      # 层级数量
    H: int = 20                           # 每层时间范围
    lamda: float = 0.3                    # 子目标测试概率
    algorithm: str = 'ddpg'               # 底层算法 ('ddpg' 或 'sac')
    
    # ==================== DDPG/SAC 算法参数 ====================
    gamma: float = 0.95                   # 折扣因子
    lr: float = 0.001                     # 学习率
    n_iter: int = 100                     # 每次更新的迭代次数
    batch_size: int = 100                 # 批大小
    hidden_dim: int = 64                  # 隐藏层维度
    
    # ==================== SAC 专用参数 ====================
    sac_alpha: float = 0.2                # 初始熵系数 (auto_entropy=False 时使用)
    sac_auto_entropy: bool = True         # 是否自动调节熵系数
    sac_target_entropy: float = None      # 目标熵 (None=自动: -action_dim)
    sac_alpha_lr: float = None            # 熵系数学习率 (None=使用 lr)
    
    # ==================== 探索噪声 ====================
    exploration_action_noise: np.ndarray = None   # 动作探索噪声
    exploration_state_noise: np.ndarray = None    # 状态/子目标探索噪声
    
    # ==================== 训练参数 ====================
    max_episodes: int = 1000              # 最大训练轮数
    save_episode: int = 10                # 保存间隔
    random_seed: int = 0                  # 随机种子
    
    def __post_init__(self):
        """初始化后处理，计算派生属性"""
        self._setup()
    
    def _setup(self):
        """子类需要实现的配置设置方法"""
        pass
    
    @property
    def effective_goal_dim(self) -> int:
        """有效的目标维度"""
        if self.goal_indices is not None:
            return len(self.goal_indices)
        return self.state_dim
    
    @property
    def subgoal_dim(self) -> int:
        """子目标维度 (通常与目标维度相同)"""
        return self.effective_goal_dim
    
    def extract_goal_from_state(self, state: np.ndarray) -> np.ndarray:
        """从完整状态中提取目标相关的部分"""
        if self.goal_indices is not None:
            return state[self.goal_indices]
        return state
    
    def get_subgoal_bounds(self) -> np.ndarray:
        """获取子目标的边界"""
        if self.goal_indices is not None:
            return self.state_bounds[self.goal_indices]
        return self.state_bounds
    
    def get_subgoal_offset(self) -> np.ndarray:
        """获取子目标的偏移"""
        if self.goal_indices is not None:
            return self.state_offset[self.goal_indices]
        return self.state_offset
    
    def get_subgoal_clip_low(self) -> np.ndarray:
        """获取子目标的裁剪下界"""
        if self.goal_indices is not None:
            return self.state_clip_low[self.goal_indices]
        return self.state_clip_low
    
    def get_subgoal_clip_high(self) -> np.ndarray:
        """获取子目标的裁剪上界"""
        if self.goal_indices is not None:
            return self.state_clip_high[self.goal_indices]
        return self.state_clip_high
    
    def get_goal_threshold(self) -> np.ndarray:
        """获取目标阈值"""
        return self.goal_threshold
    
    def get_save_directory(self) -> str:
        """获取模型保存路径"""
        return f"./preTrained/{self.env_name}/{self.k_level}level/"
    
    def get_filename(self) -> str:
        """获取模型文件名"""
        return f"HAC_{self.env_name}"
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  env_name={self.env_name},\n"
            f"  state_dim={self.state_dim}, action_dim={self.action_dim},\n"
            f"  goal_dim={self.effective_goal_dim}, goal_indices={self.goal_indices},\n"
            f"  k_level={self.k_level}, H={self.H}, lamda={self.lamda},\n"
            f"  algorithm={self.algorithm}, gamma={self.gamma}, lr={self.lr}\n"
            f")"
        )
