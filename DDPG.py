"""
Deep Deterministic Policy Gradient (DDPG) 算法实现

支持:
- 独立的目标维度 (goal_dim 可以与 state_dim 不同)
- Universal Value Function Approximator (UVFA)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    """Actor 网络: 输入 (state, goal), 输出 action"""
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        goal_dim: int,
        action_bounds: torch.Tensor, 
        action_offset: torch.Tensor,
        hidden_dim: int = 64
    ):
        super(Actor, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        self.action_bounds = action_bounds
        self.action_offset = action_offset
        
    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: 状态 [batch, state_dim]
            goal: 目标 [batch, goal_dim]
        Returns:
            action: 动作 [batch, action_dim]
        """
        x = torch.cat([state, goal], dim=1)
        return self.actor(x) * self.action_bounds + self.action_offset


class Critic(nn.Module):
    """Critic 网络: 输入 (state, action, goal), 输出 Q-value"""
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        goal_dim: int,
        H: int,
        hidden_dim: int = 64
    ):
        super(Critic, self).__init__()
        
        # Q值边界: [-H, 0]，使用负 sigmoid 实现
        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.H = H
        
    def forward(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor, 
        goal: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            state: 状态 [batch, state_dim]
            action: 动作 [batch, action_dim]
            goal: 目标 [batch, goal_dim]
        Returns:
            Q-value: [batch, 1], 范围 [-H, 0]
        """
        x = torch.cat([state, action, goal], dim=1)
        return -self.critic(x) * self.H


class DDPG:
    """DDPG 算法"""
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        goal_dim: int,
        action_bounds: torch.Tensor, 
        action_offset: torch.Tensor, 
        lr: float, 
        H: int,
        hidden_dim: int = 64
    ):
        """
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            goal_dim: 目标维度 (可以与state_dim不同)
            action_bounds: 动作边界
            action_offset: 动作偏移
            lr: 学习率
            H: 时间范围
            hidden_dim: 隐藏层维度
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        
        self.actor = Actor(
            state_dim, action_dim, goal_dim, 
            action_bounds, action_offset, hidden_dim
        ).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic = Critic(
            state_dim, action_dim, goal_dim, H, hidden_dim
        ).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.mse_loss = nn.MSELoss()
    
    def select_action(self, state, goal):
        """选择动作"""
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        goal = torch.FloatTensor(goal.reshape(1, -1)).to(device)
        return self.actor(state, goal).detach().cpu().numpy().flatten()
    
    def update(self, buffer, n_iter: int, batch_size: int):
        """更新策略"""
        for _ in range(n_iter):
            # 从经验池采样
            state, action, reward, next_state, goal, gamma, done = buffer.sample(batch_size)
            
            # 转换为张量
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action).to(device)
            reward = torch.FloatTensor(reward).reshape(-1, 1).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            goal = torch.FloatTensor(goal).to(device)
            gamma = torch.FloatTensor(gamma).reshape(-1, 1).to(device)
            done = torch.FloatTensor(done).reshape(-1, 1).to(device)
            
            # 计算目标 Q 值
            with torch.no_grad():
                next_action = self.actor(next_state, goal)
                target_Q = self.critic(next_state, next_action, goal)
                target_Q = reward + (1 - done) * gamma * target_Q
            
            # 更新 Critic
            current_Q = self.critic(state, action, goal)
            critic_loss = self.mse_loss(current_Q, target_Q)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # 更新 Actor
            actor_loss = -self.critic(state, self.actor(state, goal), goal).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
    
    def save(self, directory: str, name: str):
        """保存模型"""
        torch.save(self.actor.state_dict(), f'{directory}/{name}_actor.pth')
        torch.save(self.critic.state_dict(), f'{directory}/{name}_critic.pth')
        
    def load(self, directory: str, name: str):
        """加载模型"""
        # 兼容旧版本的拼写错误 (crtic -> critic)
        import os
        actor_path = f'{directory}/{name}_actor.pth'
        critic_path = f'{directory}/{name}_critic.pth'
        critic_old_path = f'{directory}/{name}_crtic.pth'
        
        self.actor.load_state_dict(
            torch.load(actor_path, map_location='cpu', weights_only=True)
        )
        
        if os.path.exists(critic_path):
            self.critic.load_state_dict(
                torch.load(critic_path, map_location='cpu', weights_only=True)
            )
        elif os.path.exists(critic_old_path):
            self.critic.load_state_dict(
                torch.load(critic_old_path, map_location='cpu', weights_only=True)
            )
