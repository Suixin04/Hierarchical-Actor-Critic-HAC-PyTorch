"""
工具函数和类
"""
import numpy as np
from typing import Tuple, List, Optional


class ReplayBuffer:
    """
    高性能经验回放缓冲区
    
    使用预分配的 numpy 数组存储，避免 Python 列表/deque 的开销
    采样时直接索引，无需循环
    """
    
    def __init__(self, max_size: int = 500000):
        """
        Args:
            max_size: 最大容量
        """
        self.max_size = max_size
        self.ptr = 0  # 当前写入位置
        self.size = 0  # 当前存储数量
        
        # 延迟初始化 (第一次 add 时根据数据维度初始化)
        self._initialized = False
        self.states: Optional[np.ndarray] = None
        self.actions: Optional[np.ndarray] = None
        self.rewards: Optional[np.ndarray] = None
        self.next_states: Optional[np.ndarray] = None
        self.goals: Optional[np.ndarray] = None
        self.gammas: Optional[np.ndarray] = None
        self.dones: Optional[np.ndarray] = None
    
    def _init_storage(self, state: np.ndarray, action: np.ndarray, goal: np.ndarray):
        """根据第一条数据的维度初始化存储"""
        state_dim = np.asarray(state).shape[0]
        action_dim = np.asarray(action).shape[0]
        goal_dim = np.asarray(goal).shape[0]
        
        self.states = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.goals = np.zeros((self.max_size, goal_dim), dtype=np.float32)
        self.gammas = np.zeros((self.max_size, 1), dtype=np.float32)
        self.dones = np.zeros((self.max_size, 1), dtype=np.float32)
        
        self._initialized = True
    
    def add(self, transition: Tuple):
        """
        添加一条转换记录
        
        Args:
            transition: (state, action, reward, next_state, goal, gamma, done)
        """
        assert len(transition) == 7, "transition must have length = 7"
        state, action, reward, next_state, goal, gamma, done = transition
        
        # 延迟初始化
        if not self._initialized:
            self._init_storage(state, action, goal)
        
        # 写入数据
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.goals[self.ptr] = goal
        self.gammas[self.ptr] = gamma
        self.dones[self.ptr] = done
        
        # 更新指针 (环形缓冲)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        随机采样一批数据 (向量化操作，无循环)
        
        Args:
            batch_size: 批大小
            
        Returns:
            states, actions, rewards, next_states, goals, gammas, dones
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices].flatten(),
            self.next_states[indices],
            self.goals[indices],
            self.gammas[indices].flatten(),
            self.dones[indices].flatten()
        )
    
    def sample_states(self, batch_size: int) -> np.ndarray:
        """只采样状态 (用于 E2E 训练)"""
        indices = np.random.randint(0, self.size, size=batch_size)
        return self.states[indices]
    
    def __len__(self) -> int:
        return self.size
    
    def clear(self):
        """清空缓冲区"""
        self.ptr = 0
        self.size = 0
