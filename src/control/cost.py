"""MPC 代价函数

定义轨迹追踪的代价函数。

代价组成:
    J = Σ stage_cost + terminal_cost
    stage_cost = ||pos - goal||²_Q + ||action||²_R
    terminal_cost = ||pos - goal||²_Qf

注意: 避障由上层 HAC 负责，MPC 只负责轨迹追踪。
"""

import torch
import torch.nn as nn
from typing import Optional


class MPCCost(nn.Module):
    """
    MPC 代价函数 (纯轨迹追踪)
    
    Args:
        Q: 位置误差权重 [2]
        R: 控制代价权重 [2]
        Qf: 终端代价权重 [2]
    """
    
    def __init__(
        self,
        Q: Optional[torch.Tensor] = None,
        R: Optional[torch.Tensor] = None,
        Qf: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        
        # 默认权重
        Q = Q if Q is not None else torch.tensor([1.0, 1.0])
        R = R if R is not None else torch.tensor([0.1, 0.1])
        Qf = Qf if Qf is not None else torch.tensor([10.0, 10.0])
        
        self.register_buffer('Q', Q)
        self.register_buffer('R', R)
        self.register_buffer('Qf', Qf)
    
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
            goal: [batch, 2] 目标位置
            terminal: 是否为终端代价
            
        Returns:
            cost: [batch] 或 [batch, horizon]
        """
        weight = self.Qf if terminal else self.Q
        
        if states.dim() == 2:
            # [batch, 5]
            pos = states[:, :2]
            error = pos - goal
            return (error ** 2 * weight).sum(dim=1)
        else:
            # [batch, horizon, 5]
            pos = states[:, :, :2]
            goal_expanded = goal.unsqueeze(1)
            error = pos - goal_expanded
            return (error ** 2 * weight).sum(dim=2)
    
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
            goal: [batch, 2] 目标位置
            
        Returns:
            total_cost: [batch]
        """
        # Stage costs
        stage_states = states[:, :-1, :]
        pos_cost = self.position_cost(stage_states, goal, terminal=False)
        ctrl_cost = self.control_cost(actions)
        
        # Terminal cost
        terminal_state = states[:, -1, :]
        term_cost = self.position_cost(terminal_state, goal, terminal=True)
        
        return pos_cost.sum(dim=1) + ctrl_cost.sum(dim=1) + term_cost
