"""可微模型预测控制器

通过梯度下降优化控制序列，支持:
- 4D 目标: (x, y, v_desired, θ_desired)
- 暖启动 (使用上一次的解作为初始值)
- 早停 (代价变化小于阈值时提前退出)
- 端到端训练 (梯度可回传到目标)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, List

from src.control.dynamics import DifferentiableDynamics
from src.control.cost import MPCCost
from src.utils.common import get_device


class DifferentiableMPC(nn.Module):
    """
    可微 MPC 控制器
    
    通过梯度下降优化控制序列，支持 4D 目标。
    
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
        Q_pos, Q_vel, Q_theta: 阶段代价权重
        R: 控制代价权重
        Qf_pos, Qf_vel, Qf_theta: 终端代价权重
        early_stop_tol: 早停容差
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
        # 4D 代价权重
        Q_pos: Optional[torch.Tensor] = None,
        Q_vel: float = 5.0,
        Q_theta: float = 2.0,
        R: Optional[torch.Tensor] = None,
        Qf_pos: Optional[torch.Tensor] = None,
        Qf_vel: float = 10.0,
        Qf_theta: float = 5.0,
        # 兼容旧接口
        Q: Optional[torch.Tensor] = None,
        Qf: Optional[torch.Tensor] = None,
        early_stop_tol: float = 1e-3,
    ):
        super().__init__()
        
        self.horizon = horizon
        self.max_a_v = max_a_v
        self.max_a_omega = max_a_omega
        self.num_iterations = num_iterations
        self.lr = lr
        self.early_stop_tol = early_stop_tol
        
        # 动力学模型
        self.dynamics = DifferentiableDynamics(
            dt=dt, 
            max_v=max_v, 
            max_omega=max_omega,
            damping_v=damping_v, 
            damping_omega=damping_omega
        )
        
        # 代价函数 (支持 4D 目标)
        self.cost_fn = MPCCost(
            Q_pos=Q_pos if Q_pos is not None else Q,
            Q_vel=Q_vel,
            Q_theta=Q_theta,
            R=R,
            Qf_pos=Qf_pos if Qf_pos is not None else Qf,
            Qf_vel=Qf_vel,
            Qf_theta=Qf_theta,
        )
        
        # 暖启动缓存
        self._prev_actions: Optional[torch.Tensor] = None
    
    def _clip_actions(
        self, 
        actions: torch.Tensor, 
        soft: bool = False
    ) -> torch.Tensor:
        """
        裁剪动作到约束范围
        
        Args:
            actions: [batch, horizon, 2]
            soft: 是否使用软裁剪
            
        Returns:
            clipped: [batch, horizon, 2]
        """
        a_v = actions[:, :, 0]
        a_omega = actions[:, :, 1]
        
        if soft:
            a_v = torch.tanh(a_v / self.max_a_v) * self.max_a_v
            a_omega = torch.tanh(a_omega / self.max_a_omega) * self.max_a_omega
        else:
            a_v = torch.clamp(a_v, -self.max_a_v, self.max_a_v)
            a_omega = torch.clamp(a_omega, -self.max_a_omega, self.max_a_omega)
        
        return torch.stack([a_v, a_omega], dim=2)
    
    def forward(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
        return_trajectory: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        求解 MPC 并返回最优动作
        
        Args:
            state: [batch, state_dim] 当前状态 (只用前5维)
            goal: [batch, 2] 或 [batch, 4] 目标 (位置 或 位置+速度+朝向)
            return_trajectory: 是否返回预测轨迹
            
        Returns:
            action: [batch, 2] 最优动作 (第一步)
            cost: [batch] 总代价
            info: dict
        """
        batch_size = state.shape[0]
        device = state.device
        
        # 只使用前5维
        base_state = state[:, :5].detach()
        goal_detached = goal.detach()
        
        # 初始化控制序列
        actions_data = torch.zeros(batch_size, self.horizon, 2, device=device)
        
        # 暖启动
        if self._prev_actions is not None and self._prev_actions.shape[0] == batch_size:
            actions_data = torch.cat([
                self._prev_actions[:, 1:, :],
                torch.zeros(batch_size, 1, 2, device=device)
            ], dim=1).clone()
        
        # 优化迭代
        prev_cost = float('inf')
        for _ in range(self.num_iterations):
            actions = actions_data.clone().requires_grad_(True)
            
            # 前向传播
            clipped_actions = self._clip_actions(actions)
            states = self.dynamics.rollout(base_state, clipped_actions)
            
            # 计算代价
            cost = self.cost_fn(states, clipped_actions, goal_detached)
            total_cost = cost.sum()
            
            # 早停检查
            if abs(prev_cost - total_cost.item()) < self.early_stop_tol * batch_size:
                break
            prev_cost = total_cost.item()
            
            # 计算梯度并更新
            total_cost.backward()
            
            with torch.no_grad():
                actions_data = actions_data - self.lr * actions.grad
                actions_data[:, :, 0].clamp_(-self.max_a_v, self.max_a_v)
                actions_data[:, :, 1].clamp_(-self.max_a_omega, self.max_a_omega)
        
        # 最终解
        final_actions = actions_data.detach()
        with torch.no_grad():
            final_states = self.dynamics.rollout(base_state, final_actions)
            final_cost = self.cost_fn(final_states, final_actions, goal_detached)
        
        # 保存用于暖启动
        self._prev_actions = final_actions.clone()
        
        info = {
            'predicted_trajectory': final_states if return_trajectory else None,
            'planned_actions': final_actions,
            'cost': final_cost
        }
        
        return final_actions[:, 0, :], final_cost, info
    
    def get_action_with_gradient(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取动作，同时保留对 goal 的梯度 (用于端到端训练)
        
        通过展开优化 (Unrolled Optimization) 实现可微。
        
        Args:
            state: [batch, state_dim]
            goal: [batch, 2] (需要梯度)
            
        Returns:
            action: [batch, 2]
            trajectory: [batch, horizon+1, 5]
            final_position: [batch, 2] (有到 goal 的梯度)
        """
        batch_size = state.shape[0]
        device = state.device
        base_state = state[:, :5].detach()
        
        # 初始化动作序列
        actions = torch.zeros(
            batch_size, self.horizon, 2, 
            device=device,
            requires_grad=True
        )
        
        # 展开优化迭代 (保持计算图)
        for _ in range(self.num_iterations):
            clipped_actions = self._clip_actions(actions, soft=True)
            states = self.dynamics.rollout(base_state, clipped_actions, soft_constraints=True)
            cost = self.cost_fn(states, clipped_actions, goal)
            
            # create_graph=True 保持二阶导数
            grad = torch.autograd.grad(
                cost.sum(), 
                actions, 
                create_graph=True,
                retain_graph=True
            )[0]
            
            actions = actions - self.lr * grad
        
        # 最终轨迹
        final_actions = self._clip_actions(actions, soft=True)
        final_states = self.dynamics.rollout(base_state, final_actions, soft_constraints=True)
        
        return final_actions[:, 0, :], final_states, final_states[:, -1, :2]
    
    def reset(self) -> None:
        """重置暖启动缓存"""
        self._prev_actions = None


class MPCController:
    """
    MPC 控制器包装类
    
    提供与 RL 智能体相同的接口，用于 HAC 底层。
    支持 4D 目标: (x, y, v_desired, θ_desired)
    
    Args:
        state_dim: 状态维度
        action_dim: 动作维度  
        goal_dim: 目标维度 (2 或 4)
        horizon: 预测步长
        **kwargs: 其他 MPC 参数
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        goal_dim: int,
        horizon: int = 10,
        **kwargs
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        
        device = get_device()
        self.mpc = DifferentiableMPC(horizon=horizon, **kwargs).to(device)
        self._device = device
    
    def select_action(
        self, 
        state: np.ndarray, 
        goal: np.ndarray, 
        deterministic: bool = True
    ) -> np.ndarray:
        """
        选择动作 (与 RL 智能体接口一致)
        
        Args:
            state: 状态
            goal: 子目标 [x, y] 或 [x, y, v, θ]
            deterministic: MPC 本身是确定性的
            
        Returns:
            action: [a_v, a_ω]
        """
        state_t = torch.FloatTensor(state.reshape(1, -1)).to(self._device)
        goal_t = torch.FloatTensor(goal.reshape(1, -1)).to(self._device)
        
        action, _, _ = self.mpc(state_t, goal_t)
        return action.detach().cpu().numpy().flatten()
    
    def predict_reachability(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        threshold: float = 0.8
    ) -> Tuple[bool, float, np.ndarray]:
        """
        预测子目标可达性
        
        Args:
            state: 当前状态
            goal: 子目标 [x, y] 或 [x, y, v, θ]
            threshold: 可达性阈值 (位置)
            
        Returns:
            is_reachable: 是否可达
            distance: 预测最终位置距离
            pred_goal_4d: 预测的 4D 终端目标 [x, y, v, θ]
        """
        state_t = torch.FloatTensor(state.reshape(1, -1)).to(self._device)
        goal_t = torch.FloatTensor(goal.reshape(1, -1)).to(self._device)
        
        _, _, info = self.mpc(state_t, goal_t, return_trajectory=True)
        trajectory = info['predicted_trajectory']
        
        # 提取预测的终端状态: [x, y, θ, v, ω]
        final_state = trajectory[0, -1, :5].detach().cpu().numpy()
        
        # 构造 4D 预测目标: [x, y, v, θ]
        pred_goal_4d = np.array([
            final_state[0],  # x
            final_state[1],  # y
            final_state[3],  # v (状态中是 index 3)
            final_state[2],  # θ (状态中是 index 2)
        ])
        
        # 位置可达性判断
        goal_pos = goal[:2]
        distance = np.linalg.norm(pred_goal_4d[:2] - goal_pos)
        return distance <= threshold, distance, pred_goal_4d
    
    def update(self, buffer, n_iter: int, batch_size: int) -> None:
        """MPC 不需要学习更新"""
        pass
    
    def reset(self) -> None:
        """重置 MPC 状态"""
        self.mpc.reset()
    
    def save(self, directory: str, name: str) -> None:
        """MPC 不需要保存参数"""
        pass
    
    def load(self, directory: str, name: str) -> None:
        """MPC 不需要加载参数"""
        pass
