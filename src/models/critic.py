"""Critic 网络

实现 SAC 使用的 Soft Q 网络。

设计选择:
- Bounded Q-value [-H, 0]: 使用 Sigmoid 输出
- 单 Q 网络: HAC 场景下，Bounded Q + HER 已足够稳定
"""

import torch
import torch.nn as nn


class SoftQNetwork(nn.Module):
    """
    Soft Q 网络 - Bounded Q-value [-H, 0]
    
    Args:
        state_dim: 状态维度
        action_dim: 动作维度
        goal_dim: 目标维度
        H: 时间范围 (用于 Q 值边界)
        hidden_dim: 隐藏层维度
    """
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        goal_dim: int,
        H: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        
        self.H = H
        input_dim = state_dim + action_dim + goal_dim
        
        # MLP + Sigmoid 输出
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出 [0, 1]
        )
    
    def forward(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor, 
        goal: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 Q 值
        
        Args:
            state: [batch, state_dim]
            action: [batch, action_dim]
            goal: [batch, goal_dim]
            
        Returns:
            q_value: [batch, 1], 范围 [-H, 0]
        """
        x = torch.cat([state, action, goal], dim=1)
        # Sigmoid [0, 1] -> [-H, 0]
        return -self.net(x) * self.H
