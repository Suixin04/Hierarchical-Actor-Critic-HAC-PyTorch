"""
Pendulum 环境配置
"""
import numpy as np
from configs.base_config import BaseConfig


class PendulumConfig(BaseConfig):
    """Pendulum 环境配置"""
    
    def _setup(self):
        # ==================== 环境基本信息 ====================
        self.env_name = "Pendulum-h-v1"
        
        # ==================== 状态空间配置 ====================
        # 状态: [角度, 角速度]
        self.state_dim = 2
        self.state_bounds = np.array([np.pi, 8.0])
        self.state_offset = np.array([0.0, 0.0])
        self.state_clip_low = np.array([-np.pi, -8.0])
        self.state_clip_high = np.array([np.pi, 8.0])
        
        # ==================== 动作空间配置 ====================
        # 动作: [力矩]
        self.action_dim = 1
        self.action_bounds = np.array([2.0])
        self.action_offset = np.array([0.0])
        self.action_clip_low = np.array([-2.0])
        self.action_clip_high = np.array([2.0])
        
        # ==================== 目标空间配置 ====================
        # 目标: 使用完整状态 [角度, 角速度]
        # 如果只想用角度作为目标，可以设置 goal_indices = [0]
        self.goal_indices = None
        self.goal_state = np.array([0.0, 0.0])  # 目标是竖直向上，静止
        # 原始仓库参数: threshold = [np.deg2rad(10), 0.05] ≈ [0.175, 0.05]
        self.goal_threshold = np.array([np.deg2rad(10), 0.05])
        
        # ==================== HAC 算法参数 ====================
        self.k_level = 2
        self.H = 20  # 原始仓库: H=20
        self.lamda = 0.3
        
        # ==================== DDPG 算法参数 ====================
        self.gamma = 0.95
        self.lr = 0.001
        self.n_iter = 100  # 原始仓库: n_iter=100
        self.batch_size = 100  # 原始仓库: batch_size=100
        
        # ==================== 探索噪声 ====================
        # 原始仓库参数:
        # exploration_action_noise = [0.1]
        # exploration_state_noise = [np.deg2rad(10), 0.4] ≈ [0.175, 0.4]
        self.exploration_action_noise = np.array([0.1])
        self.exploration_state_noise = np.array([np.deg2rad(10), 0.4])
        
        # ==================== 训练参数 ====================
        self.max_episodes = 1000
        self.save_episode = 10
        self.random_seed = 0


class PendulumAngleOnlyConfig(PendulumConfig):
    """
    Pendulum 只使用角度作为目标的配置示例
    """
    
    def _setup(self):
        super()._setup()
        
        # 只使用角度 (索引0) 作为目标
        self.goal_indices = [0]
        self.goal_state = np.array([0.0])
        self.goal_threshold = np.array([0.1])
        self.exploration_state_noise = np.array([0.1])
