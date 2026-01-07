"""
Navigation2D 环境配置 - 双轮差速移动机器人

环境：Navigation2DObstacle-v1 - 带障碍物的避障导航
"""
import numpy as np
from configs.base_config import BaseConfig


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
        self.num_obstacles = 6
        
        # ==================== 深度编码器配置 ====================
        self.use_depth_encoder = True      # 是否使用深度编码器
        self.base_state_dim = 5            # 基础状态维度 [x,y,θ,v,ω]
        self.depth_dim = self.depth_rays   # 深度维度
        self.embedding_dim = 8             # 嵌入维度 (增加到8维以更好表征障碍物)
        # 编码器训练模式由 HAC 自动设置:
        # - Level 1: 'e2e' 模式 (encoder 由特权学习更新，学习避障)
        # - Level 2+: 'rl' 模式 (encoder 由 RL 更新，学习高层规划)
        self.depth_fov = 2 * np.pi         # 深度传感器视场角 (360°)
        
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
        self.k_level = 4                  # 2 level hierarchy
        self.H = 8                       # time horizon per level
        self.lamda = 0.3
        
        # ==================== MPC 参数 (用于 HybridHAC) ====================
        self.dt = 0.1                    # 时间步长 (与环境一致)
        
        # 动力学阻尼参数 (必须与环境一致!)
        self.damping_v = 0.95            # 线速度阻尼 (环境: 0.95)
        self.damping_omega = 0.9         # 角速度阻尼 (环境: 0.9)
        
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
        # 注意：MPC 只负责轨迹追踪，避障由 HAC 高层学习实现
        self.mpc_horizon = 8              # MPC 预测步长
        self.mpc_iterations = 20            # MPC 优化迭代次数
        self.mpc_lr = 0.5                  # MPC 优化学习率
        self.mpc_Q = [10.0, 10.0]          # 位置误差权重
        self.mpc_R = [0.1, 0.1]            # 控制代价权重
        self.mpc_Qf = [20.0, 20.0]         # 终端代价权重
        
        # ==================== MPC 可达性预测参数 ====================
        # Level 1 使用 MPC 预测子目标可达性，替代传统 Subgoal Testing
        # 优势：不需要实际执行 H 步就能判断子目标是否可达
        self.mpc_reachability_threshold = 0.8  # 预测距离阈值（比达成阈值宽松一些）
        
        # ==================== 特权学习参数 ====================
        # 训练时使用精确障碍物位置计算梯度，让 HAC 学会输出安全子目标
        # 测试时只依赖深度传感器
        self.e2e_obstacle_weight = 10.0   # E2E 训练中避障损失权重
        self.e2e_safe_distance = 0.3      # 子目标到障碍物的安全距离
        self.e2e_traj_safe_distance = 0.2 # 轨迹点到障碍物的安全距离
        
        # ==================== Encoder 微调参数 ====================
        # RL 阶段 (Phase 2) Level 1 encoder 微调学习率
        # None = 完全冻结 encoder (只用 E2E 训练的表征)
        # 建议值: lr * 0.1 = 0.0001 (小学习率微调)
        self.encoder_finetune_lr = 0.0001
        
        # ==================== 探索噪声 ====================
        self.exploration_action_noise = np.array([0.3, 0.6])
        self.exploration_state_noise = np.array([1.0, 1.0])  # 简化为仅位置噪声
        
        # ==================== 训练参数 ====================
        self.max_episodes = 3000
        self.save_episode = 100
        self.random_seed = 0
