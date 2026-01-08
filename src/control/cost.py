"""MPC 代价函数

定义轨迹追踪的代价函数，支持 4D 目标 (x, y, v, θ)。

代价组成:
    J = Σ stage_cost + terminal_cost
    stage_cost = ||pos - goal_pos||²_Q_pos + 
                 ||v - v_desired||²_Q_vel + 
                 ||θ - θ_desired||²_Q_theta +
                 ||action||²_R
    terminal_cost = ||pos - goal_pos||²_Qf_pos + 
                    ||v - v_desired||²_Qf_vel +
                    ||θ - θ_desired||²_Qf_theta

注意: 避障由上层 HAC 负责，MPC 只负责轨迹追踪。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


def angle_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    计算角度差，正确处理 [-π, π] 边界
    
    Args:
        a, b: 角度张量 (弧度)
        
    Returns:
        diff: 归一化的角度差 ∈ [-π, π]
    """
    diff = a - b
    # 归一化到 [-π, π]
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    return diff


class MPCCost(nn.Module):
    """
    MPC 代价函数 (4D 目标: x, y, v, θ)
    
    Args:
        Q_pos: 位置误差权重 [2]
        Q_vel: 速度误差权重 (标量)
        Q_theta: 朝向误差权重 (标量)
        R: 控制代价权重 [2]
        Qf_pos: 终端位置权重 [2]
        Qf_vel: 终端速度权重 (标量)
        Qf_theta: 终端朝向权重 (标量)
    """
    
    def __init__(
        self,
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
    ):
        super().__init__()
        
        # 兼容旧接口: 如果只传了 Q/Qf，用它们作为位置权重
        if Q is not None and Q_pos is None:
            Q_pos = Q
        if Qf is not None and Qf_pos is None:
            Qf_pos = Qf
        
        # 默认权重
        Q_pos = Q_pos if Q_pos is not None else torch.tensor([10.0, 10.0])
        R = R if R is not None else torch.tensor([0.1, 0.1])
        Qf_pos = Qf_pos if Qf_pos is not None else torch.tensor([20.0, 20.0])
        
        self.register_buffer('Q_pos', Q_pos)
        self.register_buffer('Q_vel', torch.tensor(Q_vel))
        self.register_buffer('Q_theta', torch.tensor(Q_theta))
        self.register_buffer('R', R)
        self.register_buffer('Qf_pos', Qf_pos)
        self.register_buffer('Qf_vel', torch.tensor(Qf_vel))
        self.register_buffer('Qf_theta', torch.tensor(Qf_theta))
    
    def position_cost(
        self, 
        states: torch.Tensor, 
        goal: torch.Tensor,
        terminal: bool = False
    ) -> torch.Tensor:
        """
        位置误差代价
        
        Args:
            states: [batch, 5] 或 [batch, horizon, 5]
            goal: [batch, 4] 目标 (x, y, v, θ) 或 [batch, 2] 旧格式
            terminal: 是否为终端代价
            
        Returns:
            cost: [batch] 或 [batch, horizon]
        """
        weight = self.Qf_pos if terminal else self.Q_pos
        
        # 提取目标位置 (兼容 2D 和 4D)
        goal_pos = goal[:, :2]
        
        if states.dim() == 2:
            # [batch, 5]
            pos = states[:, :2]
            error = pos - goal_pos
            return (error ** 2 * weight).sum(dim=1)
        else:
            # [batch, horizon, 5]
            pos = states[:, :, :2]
            goal_expanded = goal_pos.unsqueeze(1)
            error = pos - goal_expanded
            return (error ** 2 * weight).sum(dim=2)
    
    def velocity_cost(
        self,
        states: torch.Tensor,
        goal: torch.Tensor,
        terminal: bool = False
    ) -> torch.Tensor:
        """
        速度误差代价
        
        Args:
            states: [batch, 5] 或 [batch, horizon, 5]
                    状态: [x, y, θ, v, ω]
            goal: [batch, 4] 目标 (x, y, v_desired, θ_desired)
            terminal: 是否为终端代价
            
        Returns:
            cost: [batch] 或 [batch, horizon]
        """
        # 如果是旧的 2D 目标格式，跳过速度代价
        if goal.shape[-1] < 3:
            if states.dim() == 2:
                return torch.zeros(states.shape[0], device=states.device)
            else:
                return torch.zeros(states.shape[0], states.shape[1], device=states.device)
        
        weight = self.Qf_vel if terminal else self.Q_vel
        v_desired = goal[:, 2]  # 期望速度
        
        if states.dim() == 2:
            # [batch, 5]
            v = states[:, 3]  # 当前速度
            error = v - v_desired
            return error ** 2 * weight
        else:
            # [batch, horizon, 5]
            v = states[:, :, 3]
            v_desired_expanded = v_desired.unsqueeze(1)
            error = v - v_desired_expanded
            return error ** 2 * weight
    
    def orientation_cost(
        self,
        states: torch.Tensor,
        goal: torch.Tensor,
        terminal: bool = False
    ) -> torch.Tensor:
        """
        朝向误差代价 (正确处理角度周期性)
        
        Args:
            states: [batch, 5] 或 [batch, horizon, 5]
                    状态: [x, y, θ, v, ω]
            goal: [batch, 4] 目标 (x, y, v_desired, θ_desired)
            terminal: 是否为终端代价
            
        Returns:
            cost: [batch] 或 [batch, horizon]
        """
        # 如果是旧的 2D 目标格式，跳过朝向代价
        if goal.shape[-1] < 4:
            if states.dim() == 2:
                return torch.zeros(states.shape[0], device=states.device)
            else:
                return torch.zeros(states.shape[0], states.shape[1], device=states.device)
        
        weight = self.Qf_theta if terminal else self.Q_theta
        theta_desired = goal[:, 3]  # 期望朝向
        
        if states.dim() == 2:
            # [batch, 5]
            theta = states[:, 2]  # 当前朝向
            error = angle_diff(theta, theta_desired)
            return error ** 2 * weight
        else:
            # [batch, horizon, 5]
            theta = states[:, :, 2]
            theta_desired_expanded = theta_desired.unsqueeze(1)
            error = angle_diff(theta, theta_desired_expanded)
            return error ** 2 * weight
    
    def control_cost(self, actions: torch.Tensor) -> torch.Tensor:
        """
        控制代价
        
        Args:
            actions: [batch, horizon, 2]
            
        Returns:
            cost: [batch, horizon]
        """
        return (actions ** 2 * self.R).sum(dim=2)
    
    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算总代价
        
        Args:
            states: [batch, horizon+1, 5] 状态轨迹
            actions: [batch, horizon, 2] 控制序列
            goal: [batch, 2] 或 [batch, 4] 目标
            
        Returns:
            total_cost: [batch]
        """
        # Stage costs
        stage_states = states[:, :-1, :]
        pos_cost = self.position_cost(stage_states, goal, terminal=False)
        vel_cost = self.velocity_cost(stage_states, goal, terminal=False)
        ori_cost = self.orientation_cost(stage_states, goal, terminal=False)
        ctrl_cost = self.control_cost(actions)
        
        # Terminal cost
        terminal_state = states[:, -1, :]
        term_pos_cost = self.position_cost(terminal_state, goal, terminal=True)
        term_vel_cost = self.velocity_cost(terminal_state, goal, terminal=True)
        term_ori_cost = self.orientation_cost(terminal_state, goal, terminal=True)
        
        stage_total = pos_cost.sum(dim=1) + vel_cost.sum(dim=1) + ori_cost.sum(dim=1) + ctrl_cost.sum(dim=1)
        terminal_total = term_pos_cost + term_vel_cost + term_ori_cost
        
        return stage_total + terminal_total
