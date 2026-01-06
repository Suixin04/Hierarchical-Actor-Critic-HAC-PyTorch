"""
MountainCarContinuous 环境配置
"""
import numpy as np
from configs.base_config import BaseConfig


class MountainCarConfig(BaseConfig):
    """MountainCar 连续控制环境配置"""
    
    def _setup(self):
        # ==================== 环境基本信息 ====================
        self.env_name = "MountainCarContinuous-h-v1"
        
        # ==================== 状态空间配置 ====================
        # 状态: [位置, 速度]
        self.state_dim = 2
        self.state_bounds = np.array([0.9, 0.07])
        self.state_offset = np.array([-0.3, 0.0])
        self.state_clip_low = np.array([-1.2, -0.07])
        self.state_clip_high = np.array([0.6, 0.07])
        
        # ==================== 动作空间配置 ====================
        # 动作: [力]
        self.action_dim = 1
        self.action_bounds = np.array([1.0])
        self.action_offset = np.array([0.0])
        self.action_clip_low = np.array([-1.0])
        self.action_clip_high = np.array([1.0])
        
        # ==================== 目标空间配置 ====================
        # goal_indices 指定状态的哪些维度作为目标:
        #   - None: 使用全部状态作为目标 (goal_dim = state_dim)
        #   - [0]: 只用第0维 (位置)
        #   - [0, 1]: 用第0和第1维 (位置和速度)
        #   - [0, 2, 5]: 用第0、2、5维 (适用于高维状态空间)
        # 
        # MountainCar 状态: [位置, 速度], 这里只用位置作为目标
        self.goal_indices = [0]
        # 注意: 环境在 position >= 0.45 时 terminate
        # 所以目标位置应该 <= 0.45, 阈值应该足够大使环境和HAC判断一致
        self.goal_state = np.array([0.45])      # 与环境 goal_position 一致
        self.goal_threshold = np.array([0.05])  # 达成阈值 (0.40~0.50 都算成功)
        
        # ==================== HAC 算法参数 ====================
        self.k_level = 2
        self.H = 20
        self.lamda = 0.3
        
        # ==================== DDPG 算法参数 ====================
        self.gamma = 0.95
        self.lr = 0.001
        self.n_iter = 100
        self.batch_size = 100
        
        # ==================== 探索噪声 ====================
        # exploration_action_noise: 底层动作噪声，维度 = action_dim, 范围[-1,1]
        # exploration_state_noise: 子目标噪声，维度 = goal_dim
        # 位置范围 [-1.2, 0.6], 总范围 1.8, 噪声应约为 10% = 0.2
        self.exploration_action_noise = np.array([0.2])
        self.exploration_state_noise = np.array([0.15])  # goal_dim=1 (位置)
        
        # ==================== 训练参数 ====================
        self.max_episodes = 100
        self.save_episode = 10
        self.random_seed = 0


class MountainCarPositionOnlyConfig(MountainCarConfig):
    """
    MountainCar 只使用位置作为目标的配置示例
    这展示了如何使用 goal_indices 来选择状态的子集作为目标
    """
    
    def _setup(self):
        super()._setup()
        
        # 只使用位置 (索引0) 作为目标
        self.goal_indices = [0]
        self.goal_state = np.array([0.45])       # 与环境 goal_position 一致
        self.goal_threshold = np.array([0.05])   # 达成阈值
        self.exploration_state_noise = np.array([0.02])
