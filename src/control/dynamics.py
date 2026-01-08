"""可微动力学模型

双轮差速机器人 (Unicycle Model) 的可微实现。

动力学方程:
    v_{t+1} = damping_v * v_t + a_v * dt
    ω_{t+1} = damping_ω * ω_t + a_ω * dt
    θ_{t+1} = θ_t + ω_{t+1} * dt
    x_{t+1} = x_t + v_{t+1} * cos(θ_{t+1}) * dt
    y_{t+1} = y_t + v_{t+1} * sin(θ_{t+1}) * dt
"""

import torch
import torch.nn as nn
from typing import Tuple


class DifferentiableDynamics(nn.Module):
    """
    可微的双轮差速机器人动力学模型
    
    State: [x, y, θ, v, ω]
    Action: [a_v, a_ω]
    
    Args:
        dt: 时间步长
        max_v: 最大线速度
        max_omega: 最大角速度
        damping_v: 线速度阻尼系数
        damping_omega: 角速度阻尼系数
    """
    
    def __init__(
        self,
        dt: float = 0.1,
        max_v: float = 2.0,
        max_omega: float = 2.0,
        damping_v: float = 0.95,
        damping_omega: float = 0.9,
    ):
        super().__init__()
        self.dt = dt
        self.max_v = max_v
        self.max_omega = max_omega
        self.damping_v = damping_v
        self.damping_omega = damping_omega
    
    @staticmethod
    def soft_clamp(
        x: torch.Tensor, 
        min_val: float, 
        max_val: float, 
        scale: float = 10.0
    ) -> torch.Tensor:
        """
        软裁剪函数 (可微)
        
        使用 tanh 实现软约束，保持梯度流动。
        
        Args:
            x: 输入张量
            min_val: 最小值
            max_val: 最大值
            scale: 软化程度 (越大越接近硬裁剪)
            
        Returns:
            软裁剪后的张量
        """
        center = (min_val + max_val) / 2
        half_range = (max_val - min_val) / 2
        normalized = (x - center) / (half_range + 1e-6)
        soft = torch.tanh(normalized / scale) * scale
        return soft * half_range + center
    
    def forward(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor,
        soft_constraints: bool = False
    ) -> torch.Tensor:
        """
        单步动力学前向传播
        
        Args:
            state: [batch, 5] - [x, y, θ, v, ω]
            action: [batch, 2] - [a_v, a_ω]
            soft_constraints: 是否使用软约束 (训练时使用)
            
        Returns:
            next_state: [batch, 5]
        """
        x, y, theta, v, omega = state.unbind(dim=1)
        a_v, a_omega = action.unbind(dim=1)
        
        # 带阻尼的速度更新
        v_new = self.damping_v * v + a_v * self.dt
        omega_new = self.damping_omega * omega + a_omega * self.dt
        
        # 速度约束
        if soft_constraints:
            v_new = self.soft_clamp(v_new, -self.max_v, self.max_v)
            omega_new = self.soft_clamp(omega_new, -self.max_omega, self.max_omega)
        else:
            v_new = torch.clamp(v_new, -self.max_v, self.max_v)
            omega_new = torch.clamp(omega_new, -self.max_omega, self.max_omega)
        
        # 更新朝向
        theta_new = theta + omega_new * self.dt
        theta_new = torch.atan2(torch.sin(theta_new), torch.cos(theta_new))
        
        # 更新位置
        x_new = x + v_new * torch.cos(theta_new) * self.dt
        y_new = y + v_new * torch.sin(theta_new) * self.dt
        
        return torch.stack([x_new, y_new, theta_new, v_new, omega_new], dim=1)
    
    def rollout(
        self, 
        init_state: torch.Tensor, 
        actions: torch.Tensor,
        soft_constraints: bool = False
    ) -> torch.Tensor:
        """
        多步轨迹展开
        
        Args:
            init_state: [batch, 5] 初始状态
            actions: [batch, horizon, 2] 控制序列
            soft_constraints: 是否使用软约束
            
        Returns:
            states: [batch, horizon+1, 5] 状态轨迹 (包含初始状态)
        """
        horizon = actions.shape[1]
        states = [init_state]
        state = init_state
        
        for t in range(horizon):
            state = self.forward(state, actions[:, t, :], soft_constraints)
            states.append(state)
        
        return torch.stack(states, dim=1)
