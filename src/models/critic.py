"""Critic 网络

实现 SAC 使用的 Soft Q 网络，支持 Bounded Q-value 和可选的深度编码器。

设计选择:
- Bounded Q-value [-H, 0]: 使用 Sigmoid 输出，限制 Q 值范围
- 单 Q 网络: HAC 场景下，Bounded Q + HER 已足够稳定，无需双 Q
"""

import torch
import torch.nn as nn
from typing import Optional

from src.models.encoder import DepthEncoder


class SoftQNetwork(nn.Module):
    """
    Soft Q 网络 - Bounded Q-value [-H, 0]
    
    支持可选的深度编码器。
    
    Args:
        state_dim: 状态维度
        action_dim: 动作维度
        goal_dim: 目标维度
        H: 时间范围 (用于 Q 值边界)
        hidden_dim: 隐藏层维度
        depth_encoder: 可选的深度编码器 (与 Actor 共享)
        base_state_dim: 不含深度的基础状态维度
    """
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        goal_dim: int,
        H: int,
        hidden_dim: int = 64,
        depth_encoder: Optional[DepthEncoder] = None,
        base_state_dim: int = 5
    ):
        super().__init__()
        
        self.H = H
        self.depth_encoder = depth_encoder
        self.base_state_dim = base_state_dim
        
        # 计算实际输入维度
        if depth_encoder is not None:
            input_dim = base_state_dim + depth_encoder.embedding_dim + action_dim + goal_dim
        else:
            input_dim = state_dim + action_dim + goal_dim
        
        # MLP 主干网络 + Sigmoid 输出 (4层)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出 [0, 1]
        )
    
    def _encode_state(
        self, 
        state: torch.Tensor, 
        detach_encoder: bool = False
    ) -> torch.Tensor:
        """
        编码状态
        
        Args:
            state: [batch, state_dim]
            detach_encoder: 是否阻断梯度流向编码器
            
        Returns:
            encoded: [batch, encoded_dim]
        """
        if self.depth_encoder is not None:
            base_state = state[:, :self.base_state_dim]
            depth = state[:, self.base_state_dim:]
            depth_embedding = self.depth_encoder(depth)
            if detach_encoder:
                depth_embedding = depth_embedding.detach()
            return torch.cat([base_state, depth_embedding], dim=1)
        return state
        
    def forward(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor, 
        goal: torch.Tensor, 
        detach_encoder: bool = False
    ) -> torch.Tensor:
        """
        计算 Q 值
        
        Args:
            state: [batch, state_dim]
            action: [batch, action_dim]
            goal: [batch, goal_dim]
            detach_encoder: 是否阻断编码器梯度
            
        Returns:
            q_value: [batch, 1], 范围 [-H, 0]
        """
        encoded_state = self._encode_state(state, detach_encoder)
        x = torch.cat([encoded_state, action, goal], dim=1)
        # Sigmoid 输出 [0, 1]，映射到 [-H, 0]
        return -self.net(x) * self.H
