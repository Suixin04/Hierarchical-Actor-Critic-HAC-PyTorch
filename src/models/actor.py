"""Actor 网络

实现 SAC 使用的 Gaussian 策略网络。

策略表示:
    π(a|s,g) = tanh(μ + σ * ε), ε ~ N(0, I)
    
输出经过 tanh squashing 限制在 [-1, 1]，再缩放到动作范围。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple


LOG_STD_MIN = -20
LOG_STD_MAX = 2


class GaussianActor(nn.Module):
    """
    高斯策略 Actor 网络 (SAC)
    
    输入: state + goal
    输出: 子目标 (世界坐标)
    
    Args:
        state_dim: 状态维度
        action_dim: 动作/子目标维度
        goal_dim: 目标维度
        action_bounds: 动作缩放因子 [action_dim]
        action_offset: 动作偏移 [action_dim]
        hidden_dim: 隐藏层维度
    """
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        goal_dim: int,
        action_bounds: torch.Tensor, 
        action_offset: torch.Tensor,
        hidden_dim: int = 128,
    ):
        super().__init__()
        
        input_dim = state_dim + goal_dim
        
        # MLP 主干网络
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        # 均值和标准差输出头
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        # 动作范围参数 (不可训练)
        self.register_buffer('action_scale', action_bounds.flatten())
        self.register_buffer('action_offset', action_offset.flatten())
    
    def forward(
        self, 
        state: torch.Tensor, 
        goal: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: [batch, state_dim]
            goal: [batch, goal_dim]
            
        Returns:
            mean: [batch, action_dim]
            log_std: [batch, action_dim]
        """
        x = torch.cat([state, goal], dim=1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        
        return mean, log_std
    
    def sample(
        self, 
        state: torch.Tensor, 
        goal: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样动作并计算 log 概率
        
        Args:
            state: [batch, state_dim]
            goal: [batch, goal_dim]
            
        Returns:
            action: [batch, action_dim]
            log_prob: [batch, 1]
        """
        mean, log_std = self.forward(state, goal)
        std = log_std.exp()
        
        # 重参数化采样
        dist = Normal(mean, std)
        x_t = dist.rsample()
        
        # tanh squashing
        y_t = torch.tanh(x_t)
        
        # 缩放到动作范围
        action = y_t * self.action_scale + self.action_offset
        
        # log_prob (含 tanh 修正)
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
        获取动作 (推理模式)
        
        Args:
            state: [batch, state_dim]
            goal: [batch, goal_dim]
            deterministic: 是否使用确定性动作
            
        Returns:
            action: [batch, action_dim]
        """
        mean, log_std = self.forward(state, goal)
        
        if deterministic:
            y_t = torch.tanh(mean)
        else:
            std = log_std.exp()
            dist = Normal(mean, std)
            x_t = dist.rsample()
            y_t = torch.tanh(x_t)
        
        return y_t * self.action_scale + self.action_offset
