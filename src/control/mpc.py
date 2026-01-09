"""非可微 MPC 控制器

简单的基于优化的 MPC 控制器，用于追踪子目标。
不需要梯度传播，只负责底层控制。
"""

import numpy as np
from typing import Tuple


class MPCController:
    """
    MPC 控制器 (底层)
    
    接收高层提供的子目标 [x, y]，计算控制律 [a_v, a_ω]。
    使用简单的梯度下降优化，不涉及神经网络。
    
    Args:
        horizon: 预测步长
        dt: 时间步长
        max_v: 最大线速度
        max_omega: 最大角速度
        max_a_v: 最大线加速度
        max_a_omega: 最大角加速度
        damping_v: 线速度阻尼
        damping_omega: 角速度阻尼
        num_iterations: 优化迭代次数
        lr: 优化学习率
    """
    
    def __init__(
        self,
        horizon: int = 10,
        dt: float = 0.1,
        max_v: float = 2.0,
        max_omega: float = 2.0,
        max_a_v: float = 1.0,
        max_a_omega: float = 2.0,
        damping_v: float = 0.95,
        damping_omega: float = 0.9,
        num_iterations: int = 5,
        lr: float = 0.5,
    ):
        self.horizon = horizon
        self.dt = dt
        self.max_v = max_v
        self.max_omega = max_omega
        self.max_a_v = max_a_v
        self.max_a_omega = max_a_omega
        self.damping_v = damping_v
        self.damping_omega = damping_omega
        self.num_iterations = num_iterations
        self.lr = lr
        
        # 暖启动缓存
        self._prev_actions = None
    
    def _dynamics_step(
        self, 
        state: np.ndarray, 
        action: np.ndarray
    ) -> np.ndarray:
        """
        单步动力学
        
        State: [x, y, θ, v, ω]
        Action: [a_v, a_ω]
        """
        x, y, theta, v, omega = state[:5]
        a_v, a_omega = action
        
        # 速度更新 (带阻尼)
        v_new = np.clip(
            self.damping_v * v + a_v * self.dt,
            -self.max_v, self.max_v
        )
        omega_new = np.clip(
            self.damping_omega * omega + a_omega * self.dt,
            -self.max_omega, self.max_omega
        )
        
        # 位置更新
        theta_new = theta + omega_new * self.dt
        theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))
        x_new = x + v_new * np.cos(theta_new) * self.dt
        y_new = y + v_new * np.sin(theta_new) * self.dt
        
        return np.array([x_new, y_new, theta_new, v_new, omega_new])
    
    def _rollout(
        self, 
        state: np.ndarray, 
        actions: np.ndarray
    ) -> np.ndarray:
        """
        轨迹展开
        
        Args:
            state: [5] 初始状态
            actions: [horizon, 2] 控制序列
            
        Returns:
            states: [horizon+1, 5] 状态轨迹
        """
        states = [state[:5].copy()]
        current = state[:5].copy()
        
        for t in range(self.horizon):
            current = self._dynamics_step(current, actions[t])
            states.append(current)
        
        return np.array(states)
    
    def _compute_cost(
        self, 
        states: np.ndarray, 
        actions: np.ndarray, 
        goal: np.ndarray
    ) -> float:
        """
        计算轨迹代价
        
        J = Σ ||pos - goal||² + 0.1 * ||action||² + 10 * ||final_pos - goal||²
        """
        cost = 0.0
        
        # 阶段代价
        for t in range(self.horizon):
            pos = states[t, :2]
            cost += np.sum((pos - goal) ** 2)
            cost += 0.1 * np.sum(actions[t] ** 2)
        
        # 终端代价
        final_pos = states[-1, :2]
        cost += 10.0 * np.sum((final_pos - goal) ** 2)
        
        return cost
    
    def _compute_gradient(
        self, 
        state: np.ndarray, 
        actions: np.ndarray, 
        goal: np.ndarray,
        eps: float = 1e-4
    ) -> np.ndarray:
        """
        数值梯度计算
        """
        grad = np.zeros_like(actions)
        base_cost = self._compute_cost(
            self._rollout(state, actions), actions, goal
        )
        
        for t in range(self.horizon):
            for i in range(2):
                actions_plus = actions.copy()
                actions_plus[t, i] += eps
                cost_plus = self._compute_cost(
                    self._rollout(state, actions_plus), actions_plus, goal
                )
                grad[t, i] = (cost_plus - base_cost) / eps
        
        return grad
    
    def select_action(
        self, 
        state: np.ndarray, 
        goal: np.ndarray
    ) -> np.ndarray:
        """
        选择动作
        
        Args:
            state: 当前状态 (可以包含深度，只用前5维)
            goal: 子目标 [x, y]
            
        Returns:
            action: [a_v, a_ω]
        """
        # 初始化动作序列
        if self._prev_actions is not None:
            actions = np.vstack([
                self._prev_actions[1:],
                np.zeros((1, 2))
            ])
        else:
            actions = np.zeros((self.horizon, 2))
        
        # 梯度下降优化
        for _ in range(self.num_iterations):
            grad = self._compute_gradient(state, actions, goal[:2])
            actions = actions - self.lr * grad
            
            # 裁剪动作
            actions[:, 0] = np.clip(actions[:, 0], -self.max_a_v, self.max_a_v)
            actions[:, 1] = np.clip(actions[:, 1], -self.max_a_omega, self.max_a_omega)
        
        # 保存用于暖启动
        self._prev_actions = actions.copy()
        
        return actions[0]
    
    def predict_final_position(
        self, 
        state: np.ndarray, 
        goal: np.ndarray
    ) -> np.ndarray:
        """
        预测 MPC 执行后的最终位置
        
        Args:
            state: 当前状态
            goal: 子目标
            
        Returns:
            final_pos: [x, y]
        """
        # 临时计算，不影响暖启动缓存
        actions = np.zeros((self.horizon, 2))
        
        for _ in range(self.num_iterations):
            grad = self._compute_gradient(state, actions, goal[:2])
            actions = actions - self.lr * grad
            actions[:, 0] = np.clip(actions[:, 0], -self.max_a_v, self.max_a_v)
            actions[:, 1] = np.clip(actions[:, 1], -self.max_a_omega, self.max_a_omega)
        
        states = self._rollout(state, actions)
        return states[-1, :2]
    
    def reset(self) -> None:
        """重置暖启动缓存"""
        self._prev_actions = None
