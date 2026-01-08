"""经验回放缓冲区

高性能实现:
- 预分配 numpy 数组，避免 Python 列表/deque 的开销
- 向量化采样，无循环
- 支持延迟初始化（根据第一条数据自动确定维度）
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class Transition:
    """单条经验转换"""
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    goal: np.ndarray
    gamma: float
    done: bool


class ReplayBuffer:
    """
    高性能环形经验回放缓冲区
    
    使用预分配的 numpy 数组存储数据，支持向量化采样。
    
    Attributes:
        max_size: 最大容量
        size: 当前存储数量
    """
    
    def __init__(self, max_size: int = 500_000):
        """
        初始化缓冲区
        
        Args:
            max_size: 最大容量
        """
        self.max_size = max_size
        self._ptr = 0  # 写入指针
        self.size = 0  # 当前大小
        
        # 延迟初始化 (第一次 add 时确定维度)
        self._initialized = False
        self._states: Optional[np.ndarray] = None
        self._actions: Optional[np.ndarray] = None
        self._rewards: Optional[np.ndarray] = None
        self._next_states: Optional[np.ndarray] = None
        self._goals: Optional[np.ndarray] = None
        self._gammas: Optional[np.ndarray] = None
        self._dones: Optional[np.ndarray] = None
    
    def _init_storage(
        self, 
        state: np.ndarray, 
        action: np.ndarray, 
        goal: np.ndarray
    ) -> None:
        """
        根据第一条数据的维度初始化存储
        
        Args:
            state: 状态样本 (用于推断维度)
            action: 动作样本
            goal: 目标样本
        """
        state_dim = np.asarray(state).shape[-1]
        action_dim = np.asarray(action).shape[-1]
        goal_dim = np.asarray(goal).shape[-1]
        
        self._states = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self._actions = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self._rewards = np.zeros(self.max_size, dtype=np.float32)
        self._next_states = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self._goals = np.zeros((self.max_size, goal_dim), dtype=np.float32)
        self._gammas = np.zeros(self.max_size, dtype=np.float32)
        self._dones = np.zeros(self.max_size, dtype=np.float32)
        
        self._initialized = True
    
    def add(self, transition: Tuple) -> None:
        """
        添加一条转换记录
        
        Args:
            transition: (state, action, reward, next_state, goal, gamma, done)
        """
        if len(transition) != 7:
            raise ValueError(f"Transition must have 7 elements, got {len(transition)}")
        
        state, action, reward, next_state, goal, gamma, done = transition
        
        # 延迟初始化
        if not self._initialized:
            self._init_storage(state, action, goal)
        
        # 写入数据
        idx = self._ptr
        self._states[idx] = state
        self._actions[idx] = action
        self._rewards[idx] = reward
        self._next_states[idx] = next_state
        self._goals[idx] = goal
        self._gammas[idx] = gamma
        self._dones[idx] = float(done)
        
        # 更新指针 (环形缓冲)
        self._ptr = (self._ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        随机采样一批数据 (向量化)
        
        Args:
            batch_size: 批大小
            
        Returns:
            (states, actions, rewards, next_states, goals, gammas, dones)
            - states: [batch, state_dim]
            - actions: [batch, action_dim]
            - rewards: [batch]
            - next_states: [batch, state_dim]
            - goals: [batch, goal_dim]
            - gammas: [batch]
            - dones: [batch]
        """
        if self.size < batch_size:
            raise ValueError(f"Buffer has {self.size} samples, need {batch_size}")
        
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return (
            self._states[indices],
            self._actions[indices],
            self._rewards[indices],
            self._next_states[indices],
            self._goals[indices],
            self._gammas[indices],
            self._dones[indices],
        )
    
    def sample_states(self, batch_size: int) -> np.ndarray:
        """
        只采样状态 (用于 E2E 训练)
        
        Args:
            batch_size: 批大小
            
        Returns:
            states: [batch, state_dim]
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        return self._states[indices]
    
    def __len__(self) -> int:
        return self.size
    
    def clear(self) -> None:
        """清空缓冲区"""
        self._ptr = 0
        self.size = 0
