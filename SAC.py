"""
Soft Actor-Critic (SAC) 算法实现 - 适配 HAC

特点:
- 标准 Gaussian + tanh squashing 策略
- 自动熵调节
- 双Q网络减少过估计
- 支持共享深度编码器 (Shared Depth Encoder)
- 无目标网络 (HAC场景下不需要：bounded Q + HER 已足够稳定)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Optional

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LOG_STD_MIN = -20
LOG_STD_MAX = 2


class DepthEncoder(nn.Module):
    """
    深度编码器 (每层独立)
    
    将原始深度信息 (16维) 编码为低维嵌入 (embedding_dim)
    
    设计选择：
    - 使用 LeakyReLU：避免 ReLU 的 dead neuron 和 Tanh 的饱和问题
    - 不归一化输入：保留深度的绝对值信息
    - 不限制输出范围：让后续网络自己学习如何利用 embedding
    """
    
    def __init__(self, depth_dim: int = 16, embedding_dim: int = 8, depth_max_range: float = 5.0):
        super(DepthEncoder, self).__init__()
        
        # depth_max_range 保留但不使用，保持接口兼容
        self.depth_max_range = depth_max_range
        
        self.encoder = nn.Sequential(
            nn.Linear(depth_dim, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, embedding_dim),
            nn.LeakyReLU(0.1)  # 最后也用 LeakyReLU，保持梯度流动
        )
        self.depth_dim = depth_dim
        self.embedding_dim = embedding_dim
    
    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth: [batch, depth_dim] 原始深度读数 (范围 [0, depth_max_range])
        Returns:
            embedding: [batch, embedding_dim] 深度嵌入 (无范围限制)
        """
        # 直接处理原始深度，不归一化
        return self.encoder(depth)


class GaussianActor(nn.Module):
    """
    Gaussian 策略 Actor 网络 (标准 SAC)
    
    使用 tanh squashing 将输出限制在 [-1, 1]，再缩放到动作范围
    支持可选的共享深度编码器
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
        base_state_dim: int = 5  # 不含深度的基础状态维度
    ):
        super(GaussianActor, self).__init__()
        
        self.depth_encoder = depth_encoder
        self.base_state_dim = base_state_dim
        
        # 计算实际输入维度
        if depth_encoder is not None:
            # 使用编码器: base_state + embedding + goal
            input_dim = base_state_dim + depth_encoder.embedding_dim + goal_dim
        else:
            # 不使用编码器: state + goal
            input_dim = state_dim + goal_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        # 动作范围: action = tanh(x) * scale + offset
        self.register_buffer('action_scale', action_bounds.flatten())
        self.register_buffer('action_offset', action_offset.flatten())
    
    def _encode_state(self, state: torch.Tensor) -> torch.Tensor:
        """编码状态（如果有深度编码器）"""
        if self.depth_encoder is not None:
            # 分离基础状态和深度
            base_state = state[:, :self.base_state_dim]
            depth = state[:, self.base_state_dim:]
            # 编码深度
            depth_embedding = self.depth_encoder(depth)
            # 拼接
            return torch.cat([base_state, depth_embedding], dim=1)
        return state
        
    def forward(self, state: torch.Tensor, goal: torch.Tensor):
        encoded_state = self._encode_state(state)
        x = torch.cat([encoded_state, goal], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor, goal: torch.Tensor):
        """采样动作并计算 log 概率（含 tanh 修正）"""
        mean, log_std = self.forward(state, goal)
        std = log_std.exp()
        
        # 从高斯分布采样
        dist = Normal(mean, std)
        x_t = dist.rsample()  # reparameterization trick
        
        # tanh squashing
        y_t = torch.tanh(x_t)
        
        # 缩放到动作范围
        action = y_t * self.action_scale + self.action_offset
        
        # log_prob 计算 (含 tanh 的雅可比修正)
        # log π(a|s) = log μ(u|s) - Σ log(1 - tanh²(u_i))
        log_prob = dist.log_prob(x_t)
        # 雅可比修正: log(1 - tanh²(x)) = log(1 - y²)
        log_prob = log_prob - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return action, log_prob
    
    def get_action(self, state: torch.Tensor, goal: torch.Tensor, deterministic: bool = False):
        """获取动作"""
        mean, log_std = self.forward(state, goal)
        
        if deterministic:
            # 确定性: 直接用均值
            y_t = torch.tanh(mean)
        else:
            # 随机: 从分布采样
            std = log_std.exp()
            dist = Normal(mean, std)
            x_t = dist.rsample()
            y_t = torch.tanh(x_t)
        
        # 缩放到动作范围
        action = y_t * self.action_scale + self.action_offset
        return action


class SoftQNetwork(nn.Module):
    """Soft Q网络 - Bounded Q-value [-H, 0]，支持共享深度编码器"""
    
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
        super(SoftQNetwork, self).__init__()
        
        self.H = H
        self.depth_encoder = depth_encoder
        self.base_state_dim = base_state_dim
        
        # 计算实际输入维度
        if depth_encoder is not None:
            input_dim = base_state_dim + depth_encoder.embedding_dim + action_dim + goal_dim
        else:
            input_dim = state_dim + action_dim + goal_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def _encode_state(self, state: torch.Tensor) -> torch.Tensor:
        """编码状态（如果有深度编码器）"""
        if self.depth_encoder is not None:
            base_state = state[:, :self.base_state_dim]
            depth = state[:, self.base_state_dim:]
            depth_embedding = self.depth_encoder(depth)
            return torch.cat([base_state, depth_embedding], dim=1)
        return state
        
    def forward(self, state: torch.Tensor, action: torch.Tensor, goal: torch.Tensor):
        encoded_state = self._encode_state(state)
        x = torch.cat([encoded_state, action, goal], dim=1)
        return -self.net(x) * self.H  # [-H, 0]


class SAC:
    """
    Soft Actor-Critic with Gaussian Policy (标准实现)
    
    特点:
    - Gaussian + tanh squashing，稳定可靠
    - Bounded Q-value [-H, 0] 限制Q值范围
    - 自动熵调节
    - 每层独立深度编码器（避免梯度冲突）
    """
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        goal_dim: int,
        action_bounds: torch.Tensor, 
        action_offset: torch.Tensor, 
        lr: float, 
        H: int,
        hidden_dim: int = 64,
        alpha: float = 0.2,
        auto_entropy: bool = True,
        # 独立编码器参数
        use_depth_encoder: bool = False,
        base_state_dim: int = 5,
        depth_dim: int = 16,
        embedding_dim: int = 8,
        depth_max_range: float = 5.0,
        level: int = 0
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.H = H
        self.level = level
        
        # 创建本层独立的深度编码器
        if use_depth_encoder:
            self.depth_encoder = DepthEncoder(
                depth_dim=depth_dim,
                embedding_dim=embedding_dim,
                depth_max_range=depth_max_range
            ).to(device)
        else:
            self.depth_encoder = None
        
        # Gaussian Actor (使用本层独立编码器)
        self.actor = GaussianActor(
            state_dim, action_dim, goal_dim,
            action_bounds, action_offset, hidden_dim,
            depth_encoder=self.depth_encoder,
            base_state_dim=base_state_dim
        ).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        # 双Q网络 (使用本层独立编码器)
        self.q1 = SoftQNetwork(
            state_dim, action_dim, goal_dim, H, hidden_dim,
            depth_encoder=self.depth_encoder, base_state_dim=base_state_dim
        ).to(device)
        self.q2 = SoftQNetwork(
            state_dim, action_dim, goal_dim, H, hidden_dim,
            depth_encoder=self.depth_encoder, base_state_dim=base_state_dim
        ).to(device)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)
        
        # 自动熵调节
        self.auto_entropy = auto_entropy
        if auto_entropy:
            # 标准目标熵: -dim(A)
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha
    
    def select_action(self, state, goal, deterministic: bool = False):
        """选择动作"""
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        goal = torch.FloatTensor(goal.reshape(1, -1)).to(device)
        
        with torch.no_grad():
            action = self.actor.get_action(state, goal, deterministic)
        return action.cpu().numpy().flatten()
    
    def update(self, buffer, n_iter: int, batch_size: int):
        """更新策略"""
        for _ in range(n_iter):
            state, action, reward, next_state, goal, gamma, done = buffer.sample(batch_size)
            
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action).to(device)
            reward = torch.FloatTensor(reward).reshape(-1, 1).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            goal = torch.FloatTensor(goal).to(device)
            gamma = torch.FloatTensor(gamma).reshape(-1, 1).to(device)
            done = torch.FloatTensor(done).reshape(-1, 1).to(device)
            
            # ===== Critic Update =====
            with torch.no_grad():
                next_action, next_log_prob = self.actor.sample(next_state, goal)
                q_next = torch.min(
                    self.q1(next_state, next_action, goal),
                    self.q2(next_state, next_action, goal)
                ) - self.alpha * next_log_prob
                target_q = reward + (1 - done) * gamma * q_next
            
            # Q1
            q1_loss = F.mse_loss(self.q1(state, action, goal), target_q)
            self.q1_optimizer.zero_grad()
            q1_loss.backward()
            self.q1_optimizer.step()
            
            # Q2
            q2_loss = F.mse_loss(self.q2(state, action, goal), target_q)
            self.q2_optimizer.zero_grad()
            q2_loss.backward()
            self.q2_optimizer.step()
            
            # ===== Actor Update =====
            new_action, log_prob = self.actor.sample(state, goal)
            q_new = torch.min(
                self.q1(state, new_action, goal),
                self.q2(state, new_action, goal)
            )
            actor_loss = (self.alpha * log_prob - q_new).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # ===== Entropy Coefficient Update =====
            if self.auto_entropy:
                alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp().item()
    
    def save(self, directory: str, name: str):
        torch.save(self.actor.state_dict(), f'{directory}/{name}_actor.pth')
        torch.save(self.q1.state_dict(), f'{directory}/{name}_q1.pth')
        torch.save(self.q2.state_dict(), f'{directory}/{name}_q2.pth')
        # 保存本层的编码器
        if self.depth_encoder is not None:
            torch.save(self.depth_encoder.state_dict(), f'{directory}/{name}_encoder.pth')
        
    def load(self, directory: str, name: str):
        self.actor.load_state_dict(
            torch.load(f'{directory}/{name}_actor.pth', map_location='cpu', weights_only=True)
        )
        self.q1.load_state_dict(
            torch.load(f'{directory}/{name}_q1.pth', map_location='cpu', weights_only=True)
        )
        self.q2.load_state_dict(
            torch.load(f'{directory}/{name}_q2.pth', map_location='cpu', weights_only=True)
        )
        # 加载本层的编码器
        if self.depth_encoder is not None:
            import os
            encoder_path = f'{directory}/{name}_encoder.pth'
            if os.path.exists(encoder_path):
                self.depth_encoder.load_state_dict(
                    torch.load(encoder_path, map_location='cpu', weights_only=True)
                )
