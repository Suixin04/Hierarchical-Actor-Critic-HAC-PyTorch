"""Navigation2D 环境配置

双轮差速移动机器人导航环境配置。

环境: Navigation2DObstacle-v1
State: [x, y, θ, v, ω, depth_1, ..., depth_16]
Action: [a_v, a_ω]
Goal: [x, y]
"""

import numpy as np
from dataclasses import dataclass
from src.configs.base import BaseConfig


@dataclass
class Navigation2DConfig(BaseConfig):
    """
    带障碍物的避障导航环境配置
    
    关键设计:
    - 使用共享深度编码器
    - Level 1 使用极坐标 + 深度约束
    """
    
    def _setup(self):
        # ==================== 环境基本信息 ====================
        self.env_name = "Navigation2DObstacle-v1"
        
        # 环境参数
        self.world_size = 10.0
        self.max_v = 2.0
        self.max_omega = 2.0
        self.max_a_v = 1.0
        self.max_a_omega = 2.0
        self.depth_rays = 16
        self.depth_max_range = 5.0
        self.num_obstacles = 0
        
        # ==================== 深度编码器 ====================
        self.use_depth_encoder = True
        self.base_state_dim = 5
        self.depth_dim = self.depth_rays
        self.embedding_dim = 8
        self.depth_fov = 2 * np.pi
        
        # ==================== 状态空间 ====================
        self.state_dim = 5 + self.depth_rays
        
        self.state_bounds = np.concatenate([
            [self.world_size / 2, self.world_size / 2],
            [np.pi],
            [self.max_v, self.max_omega],
            [self.depth_max_range / 2] * self.depth_rays,
        ])
        self.state_offset = np.concatenate([
            [self.world_size / 2, self.world_size / 2],
            [0.0],
            [0.0, 0.0],
            [self.depth_max_range / 2] * self.depth_rays,
        ])
        self.state_clip_low = np.concatenate([
            [0, 0, -np.pi, -self.max_v, -self.max_omega],
            [0] * self.depth_rays,
        ])
        self.state_clip_high = np.concatenate([
            [self.world_size, self.world_size, np.pi, self.max_v, self.max_omega],
            [self.depth_max_range] * self.depth_rays,
        ])
        
        # ==================== 动作空间 ====================
        self.action_dim = 2
        self.action_bounds = np.array([self.max_a_v, self.max_a_omega])
        self.action_offset = np.array([0.0, 0.0])
        self.action_clip_low = np.array([-self.max_a_v, -self.max_a_omega])
        self.action_clip_high = np.array([self.max_a_v, self.max_a_omega])
        
        # ==================== 目标空间 ====================
        self.goal_indices = [0, 1]
        self.goal_state = np.array([8.5, 8.5])
        self.goal_threshold = np.array([0.5, 0.5])
        
        # ==================== Level 1 极坐标 ====================
        self.level1_use_polar = True
        self.subgoal_fov = np.pi
        self.subgoal_r_min = 0.3
        self.subgoal_r_max = 2.5  # 略小于 MPC 可达距离 (15*0.1*2=3m)，留有余量
        self.subgoal_safety_margin = 0.3
        
        self.level1_subgoal_bounds = np.array([
            (self.subgoal_r_max - self.subgoal_r_min) / 2,
            self.subgoal_fov / 2
        ])
        self.level1_subgoal_offset = np.array([
            (self.subgoal_r_max + self.subgoal_r_min) / 2,
            0.0
        ])
        
        # ==================== HAC 参数 ====================
        self.k_level = 3
        self.H = 15  # 增加 MPC horizon: 15 * 0.1s = 1.5s, max_v=2 => 可达 3m
        self.lamda = 0.3
        
        # ==================== MPC 参数 ====================
        self.dt = 0.1
        self.damping_v = 0.95
        self.damping_omega = 0.9
        self.mpc_iterations = 10
        self.mpc_lr = 0.5
        self.mpc_Q = [10.0, 10.0]
        self.mpc_R = [0.1, 0.1]
        self.mpc_Qf = [20.0, 20.0]
        self.mpc_reachability_threshold = 0.8
        
        # ==================== 学习参数 ====================
        self.gamma = 0.99
        self.lr = 0.001
        self.n_iter = 100
        self.batch_size = 128
        self.hidden_dim = 64
        
        # ==================== SAC 参数 ====================
        self.sac_alpha = 0.3
        self.sac_auto_entropy = True
        self.sac_target_entropy = None
        self.sac_alpha_lr = None
        
        # ==================== 特权学习 ====================
        self.boundary_margin = 0.1
        self.e2e_obstacle_weight = 10.0
        self.e2e_safe_distance = 0.3
        self.e2e_traj_safe_distance = 0.2
        
        # ==================== 编码器微调 ====================
        self.encoder_finetune_lr = 0.0001
        
        # ==================== 探索噪声 ====================
        self.exploration_action_noise = np.array([0.3, 0.6])
        self.exploration_state_noise = np.array([1.0, 1.0])
        
        # ==================== 训练参数 ====================
        self.max_episodes = 3000
        self.save_episode = 100
        self.random_seed = 0
