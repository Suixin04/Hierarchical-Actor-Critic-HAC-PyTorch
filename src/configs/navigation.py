"""Navigation2D 环境配置

双轮差速移动机器人导航环境配置。

环境: Navigation2DObstacle-v1
State: [x, y, θ, v, ω, depth_1, ..., depth_16]
Action: [a_v, a_ω]
Goal: [x, y]  (世界坐标)
"""

import numpy as np
from dataclasses import dataclass
from src.configs.base import BaseConfig


@dataclass
class Navigation2DConfig(BaseConfig):
    """带障碍物的避障导航环境配置"""
    
    def _setup(self):
        # ==================== 环境基本信息 ====================
        self.env_name = "Navigation2DObstacle-v1"
        self.world_size = 10.0
        
        # ==================== 机器人参数 ====================
        self.max_v = 2.0
        self.max_omega = 2.0
        self.max_a_v = 1.0
        self.max_a_omega = 2.0
        self.depth_rays = 16
        self.depth_max_range = 5.0
        
        # ==================== 状态空间 ====================
        # State: [x, y, θ, v, ω, depth_1, ..., depth_16]
        self.state_dim = 5 + self.depth_rays
        
        self.state_bounds = np.concatenate([
            [self.world_size / 2, self.world_size / 2],  # x, y
            [np.pi],                                       # θ
            [self.max_v, self.max_omega],                  # v, ω
            [self.depth_max_range / 2] * self.depth_rays,  # depth
        ])
        self.state_offset = np.concatenate([
            [self.world_size / 2, self.world_size / 2],
            [0.0],
            [0.0, 0.0],
            [self.depth_max_range / 2] * self.depth_rays,
        ])
        
        # ==================== 动作空间 ====================
        # Action: [a_v, a_ω]
        self.action_dim = 2
        self.action_bounds = np.array([self.max_a_v, self.max_a_omega])
        self.action_offset = np.array([0.0, 0.0])
        
        # ==================== 目标空间 ====================
        # Goal: [x, y] (世界坐标)
        self.goal_dim = 2
        self.goal_state = np.array([8.5, 8.5])
        self.goal_threshold = 0.5
        
        # ==================== HAC 参数 ====================
        self.k_level = 3       # 3层: Level 2 -> Level 1 -> MPC
        self.H = 8            # 每层最大尝试次数
        self.lamda = 0.3       # 子目标测试概率
        
        # ==================== MPC 参数 ====================
        self.dt = 0.1
        self.damping_v = 0.95
        self.damping_omega = 0.9
        self.mpc_horizon = 8
        
        # ==================== 学习参数 ====================
        self.gamma = 0.99
        self.lr = 0.001
        self.n_iter = 64
        self.batch_size = 128
        self.hidden_dim = 128
        
        # ==================== SAC 参数 ====================
        self.sac_alpha = 0.2
        self.sac_auto_entropy = True
        
        # ==================== 训练参数 ====================
        self.max_episodes = 3000
        self.random_seed = 42
