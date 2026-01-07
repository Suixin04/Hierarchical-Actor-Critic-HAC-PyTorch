"""
Navigation2D 环境配置 - 双轮差速移动机器人

两种环境：
1. Navigation2DSimple-v1: 无障碍物，用于基础训练
2. Navigation2DObstacle-v1: 带障碍物，用于避障训练
"""
import numpy as np
from configs.base_config import BaseConfig


class Navigation2DSimpleConfig(BaseConfig):
    """
    无障碍物的简单导航环境
    
    State: [x, y, θ, v, ω]
    Action: [a_v, a_ω]
    Goal: [x, y]
    """
    
    def _setup(self):
        # ==================== 环境基本信息 ====================
        self.env_name = "Navigation2DSimple-v1"
        
        # 环境参数
        self.world_size = 10.0
        self.max_v = 2.0
        self.max_omega = 2.0
        self.max_a_v = 1.0
        self.max_a_omega = 2.0
        
        # ==================== 状态空间配置 ====================
        self.state_dim = 5
        
        self.state_bounds = np.array([
            self.world_size / 2, self.world_size / 2,
            np.pi,
            self.max_v, self.max_omega,
        ])
        self.state_offset = np.array([
            self.world_size / 2, self.world_size / 2,
            0.0,
            0.0, 0.0,
        ])
        self.state_clip_low = np.array([
            0, 0, -np.pi, -self.max_v, -self.max_omega,
        ])
        self.state_clip_high = np.array([
            self.world_size, self.world_size, np.pi, self.max_v, self.max_omega,
        ])
        
        # ==================== 动作空间配置 ====================
        self.action_dim = 2
        self.action_bounds = np.array([self.max_a_v, self.max_a_omega])
        self.action_offset = np.array([0.0, 0.0])
        self.action_clip_low = np.array([-self.max_a_v, -self.max_a_omega])
        self.action_clip_high = np.array([self.max_a_v, self.max_a_omega])
        
        # ==================== 目标空间配置 ====================
        self.goal_indices = [0, 1]
        self.goal_state = np.array([8.5, 8.5])
        self.goal_threshold = np.array([0.5, 0.5])  # 子目标达成阈值
        
        # ==================== HAC 算法参数 ====================
        self.k_level = 4                  # 4 level hierarchy
        self.H = 10                       # time horizon per level
        self.lamda = 0.3
        
        # ==================== 学习参数 ====================
        self.gamma = 0.99
        self.lr = 0.001
        self.n_iter = 100
        self.batch_size = 128
        self.hidden_dim = 64
        
        # ==================== SAC 专用参数 ====================
        self.sac_alpha = 0.2               # 初始熵系数
        self.sac_auto_entropy = True       # 自动熵调节
        self.sac_target_entropy = None     # None = -action_dim = -2
        self.sac_alpha_lr = None           # None = 使用 lr
        
        # ==================== 探索噪声 ====================
        self.exploration_action_noise = np.array([0.5, 1.0])
        self.exploration_state_noise = np.array([2.0, 2.0])
        
        # ==================== 训练参数 ====================
        self.max_episodes = 2000
        self.save_episode = 50
        self.random_seed = 0


class Navigation2DObstacleConfig(BaseConfig):
    """
    带障碍物的避障导航环境
    
    关键设计：
    - 使用共享深度编码器 (Shared Depth Encoder)
    - 深度信息被编码为低维嵌入，供各层使用
    
    State: [x, y, θ, v, ω, depth_1, ..., depth_16]
    Action: [a_v, a_ω]
    Goal: [x, y]
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
        self.num_obstacles = 5
        
        # ==================== 深度编码器配置 ====================
        self.use_depth_encoder = True      # 是否使用深度编码器
        self.base_state_dim = 5            # 基础状态维度 [x,y,θ,v,ω]
        self.depth_dim = self.depth_rays   # 深度维度
        self.embedding_dim = 4             # 嵌入维度 (4D 足够表征障碍物方向+距离)
        
        # ==================== 状态空间配置 ====================
        self.state_dim = 5 + self.depth_rays  # 21维
        
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
        
        # ==================== 动作空间配置 ====================
        self.action_dim = 2
        self.action_bounds = np.array([self.max_a_v, self.max_a_omega])
        self.action_offset = np.array([0.0, 0.0])
        self.action_clip_low = np.array([-self.max_a_v, -self.max_a_omega])
        self.action_clip_high = np.array([self.max_a_v, self.max_a_omega])
        
        # ==================== 目标空间配置 ====================
        self.goal_indices = [0, 1]
        self.goal_state = np.array([8.5, 8.5])
        self.goal_threshold = np.array([0.5, 0.5])  # 子目标达成阈值
        
        # ==================== HAC 算法参数 ====================
        self.k_level = 4                  # 4 level hierarchy
        self.H = 10                       # time horizon per level
        self.lamda = 0.3
        
        # ==================== MPC 参数 (用于 HybridHAC) ====================
        self.dt = 0.1                    # 时间步长 (与环境一致)
        
        # ==================== 学习参数 ====================
        self.gamma = 0.99
        self.lr = 0.001
        self.n_iter = 100
        self.batch_size = 128
        self.hidden_dim = 64
        
        # ==================== SAC 专用参数 ====================
        self.sac_alpha = 0.2               # 初始熵系数
        self.sac_auto_entropy = True       # 自动熵调节
        self.sac_target_entropy = None     # None = -action_dim = -2
        self.sac_alpha_lr = None           # None = 使用 lr
        
        # ==================== MPC 参数 ====================
        self.mpc_horizon = 10              # MPC 预测步长
        self.mpc_iterations = 5            # MPC 优化迭代次数
        self.mpc_lr = 0.5                  # MPC 优化学习率
        self.mpc_Q = [10.0, 10.0]          # 位置误差权重
        self.mpc_R = [0.1, 0.1]            # 控制代价权重
        self.mpc_Qf = [20.0, 20.0]         # 终端代价权重
        self.mpc_obstacle_weight = 10.0   # 避障权重
        self.mpc_safe_distance = 0.5      # 安全距离
        
        # ==================== 探索噪声 ====================
        self.exploration_action_noise = np.array([0.3, 0.6])
        self.exploration_state_noise = np.array([1.0, 1.0])  # 简化为仅位置噪声
        
        # ==================== 训练参数 ====================
        self.max_episodes = 3000
        self.save_episode = 100
        self.random_seed = 0
