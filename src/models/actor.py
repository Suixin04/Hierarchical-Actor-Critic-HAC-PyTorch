"""Actor 网络

实现 SAC 使用的 Gaussian 策略网络，支持可选的深度编码器。

策略表示:
    π(a|s,g) = tanh(μ + σ * ε), ε ~ N(0, I)
    
输出经过 tanh squashing 限制在 [-1, 1]，再缩放到动作范围。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Optional, Tuple

from src.models.encoder import DepthEncoder


# Log std 范围限制
LOG_STD_MIN = -20
LOG_STD_MAX = 2


class GaussianActor(nn.Module):
    """
    高斯策略 Actor 网络 (SAC)
    
    使用 tanh squashing 将输出限制在 [-1, 1]，再缩放到动作范围。
    支持可选的深度编码器。
    
    Args:
        state_dim: 状态维度 (包含深度)
        action_dim: 动作维度
        goal_dim: 目标维度
        action_bounds: 动作缩放因子 [action_dim]
        action_offset: 动作偏移 [action_dim]
        hidden_dim: 隐藏层维度
        depth_encoder: 可选的深度编码器
        base_state_dim: 不含深度的基础状态维度
    """
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        goal_dim: int,
        action_bounds: torch.Tensor, 
        action_offset: torch.Tensor,
        hidden_dim: int = 64,
        depth_encoder: Optional[DepthEncoder] = None,
        base_state_dim: int = 5
    ):
        super().__init__()
        
        self.depth_encoder = depth_encoder
        self.base_state_dim = base_state_dim
        
        # 计算实际输入维度
        if depth_encoder is not None:
            # 使用编码器: base_state + embedding + goal
            input_dim = base_state_dim + depth_encoder.embedding_dim + goal_dim
        else:
            # 不使用编码器: state + goal
            input_dim = state_dim + goal_dim
        
        # MLP 主干网络 (4层)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        
        # 均值和标准差输出头
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        # 动作范围参数 (不可训练)
        # action = tanh(x) * scale + offset
        self.register_buffer('action_scale', action_bounds.flatten())
        self.register_buffer('action_offset', action_offset.flatten())
    
    def _encode_state(
        self, 
        state: torch.Tensor, 
        detach_encoder: bool = False
    ) -> torch.Tensor:
        """
        编码状态 (分离基础状态和深度)
        
        Args:
            state: [batch, state_dim] 完整状态
            detach_encoder: 是否阻断梯度流向编码器
            
        Returns:
            encoded: [batch, base_state_dim + embedding_dim] 或 [batch, state_dim]
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
        goal: torch.Tensor, 
        detach_encoder: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播，输出均值和 log 标准差
        
        Args:
            state: [batch, state_dim]
            goal: [batch, goal_dim]
            detach_encoder: 是否阻断编码器梯度
            
        Returns:
            mean: [batch, action_dim]
            log_std: [batch, action_dim]
        """
        encoded_state = self._encode_state(state, detach_encoder)
        x = torch.cat([encoded_state, goal], dim=1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        
        return mean, log_std
    
    def sample(
        self, 
        state: torch.Tensor, 
        goal: torch.Tensor, 
        detach_encoder: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样动作并计算 log 概率 (含 tanh 修正)
        
        Args:
            state: [batch, state_dim]
            goal: [batch, goal_dim]
            detach_encoder: 是否阻断编码器梯度
            
        Returns:
            action: [batch, action_dim] 缩放后的动作
            log_prob: [batch, 1] log 概率
        """
        mean, log_std = self.forward(state, goal, detach_encoder)
        std = log_std.exp()
        
        # 重参数化采样
        dist = Normal(mean, std)
        x_t = dist.rsample()  # reparameterization trick
        
        # tanh squashing
        y_t = torch.tanh(x_t)
        
        # 缩放到动作范围
        action = y_t * self.action_scale + self.action_offset
        
        # log_prob 计算 (含 tanh 的雅可比修正)
        # log π(a|s) = log μ(u|s) - Σ log(1 - tanh²(u_i))
        log_prob = dist.log_prob(x_t)
        log_prob = log_prob - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return action, log_prob
    
    def get_action(
        self, 
        state: torch.Tensor, 
        goal: torch.Tensor, 
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        获取动作 (推理模式，无梯度)
        
        Args:
            state: [batch, state_dim]
            goal: [batch, goal_dim]
            deterministic: 是否使用确定性动作 (均值)
            
        Returns:
            action: [batch, action_dim]
        """
        mean, log_std = self.forward(state, goal, detach_encoder=True)
        
        if deterministic:
            y_t = torch.tanh(mean)
        else:
            std = log_std.exp()
            dist = Normal(mean, std)
            x_t = dist.rsample()
            y_t = torch.tanh(x_t)
        
        return y_t * self.action_scale + self.action_offset
