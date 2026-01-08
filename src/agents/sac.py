"""Soft Actor-Critic 智能体

实现 SAC 算法，特点:
- Gaussian + tanh squashing 策略
- 自动熵调节
- Bounded Q-value [-H, 0]
- 支持深度编码器
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import os

from src.models.encoder import DepthEncoder
from src.models.actor import GaussianActor
from src.models.critic import SoftQNetwork
from src.buffers.replay_buffer import ReplayBuffer
from src.utils.common import get_device, to_tensor, to_numpy


class SACAgent:
    """
    Soft Actor-Critic 智能体
    
    支持:
    - 可选的深度编码器 (每层独立)
    - 灵活的编码器训练模式
    - 自动熵调节
    
    Args:
        state_dim: 状态维度
        action_dim: 动作维度
        goal_dim: 目标维度
        action_bounds: 动作缩放因子
        action_offset: 动作偏移
        lr: 学习率
        H: 时间范围 (Q 值边界)
        hidden_dim: 隐藏层维度
        alpha: 初始熵系数
        auto_entropy: 是否自动调节熵
        target_entropy: 目标熵
        alpha_lr: 熵系数学习率
        use_depth_encoder: 是否使用深度编码器
        base_state_dim: 基础状态维度
        depth_dim: 深度维度
        embedding_dim: 嵌入维度
        level: 层级 (用于日志)
        encoder_train_mode: 编码器训练模式 ('rl' 或 'finetune')
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
        target_entropy: Optional[float] = None,
        alpha_lr: Optional[float] = None,
        use_depth_encoder: bool = False,
        base_state_dim: int = 5,
        depth_dim: int = 16,
        embedding_dim: int = 8,
        level: int = 0,
        encoder_train_mode: str = 'rl',
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.H = H
        self.level = level
        self.encoder_train_mode = encoder_train_mode
        
        self._device = get_device()
        
        # 创建深度编码器 (如果需要)
        self.depth_encoder: Optional[DepthEncoder] = None
        if use_depth_encoder:
            self.depth_encoder = DepthEncoder(
                depth_dim=depth_dim,
                embedding_dim=embedding_dim
            ).to(self._device)
        
        # 创建 Actor
        self.actor = GaussianActor(
            state_dim, action_dim, goal_dim,
            action_bounds, action_offset, hidden_dim,
            depth_encoder=self.depth_encoder,
            base_state_dim=base_state_dim
        ).to(self._device)
        
        # Actor 优化器 (排除编码器参数)
        actor_params = [
            p for n, p in self.actor.named_parameters() 
            if 'depth_encoder' not in n
        ]
        self.actor_optimizer = optim.Adam(actor_params, lr=lr)
        
        # 创建 Critic (共享编码器)
        self.critic = SoftQNetwork(
            state_dim, action_dim, goal_dim, H, hidden_dim,
            depth_encoder=self.depth_encoder, 
            base_state_dim=base_state_dim
        ).to(self._device)
        
        # Critic 优化器 (排除编码器参数)
        critic_params = [
            p for n, p in self.critic.named_parameters() 
            if 'depth_encoder' not in n
        ]
        self.critic_optimizer = optim.Adam(critic_params, lr=lr)
        
        # 自动熵调节
        self.auto_entropy = auto_entropy
        if auto_entropy:
            self.target_entropy = target_entropy if target_entropy is not None else -action_dim
            # 使用配置的初始 alpha 值，而不是默认的 1.0
            init_log_alpha = np.log(alpha) if alpha > 0 else 0.0
            self.log_alpha = torch.tensor([init_log_alpha], requires_grad=True, device=self._device)
            _alpha_lr = alpha_lr if alpha_lr is not None else lr
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=_alpha_lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha
        
        # 编码器优化器 (可选)
        self.encoder_optimizer: Optional[optim.Adam] = None
        if self.depth_encoder is not None and encoder_train_mode == 'rl':
            self.encoder_optimizer = optim.Adam(
                self.depth_encoder.parameters(), lr=lr
            )
    
    def setup_encoder_finetune(self, finetune_lr: float) -> None:
        """
        设置编码器微调 (用于 Phase 2)
        
        Args:
            finetune_lr: 微调学习率
        """
        if self.depth_encoder is not None:
            self.encoder_optimizer = optim.Adam(
                self.depth_encoder.parameters(), 
                lr=finetune_lr
            )
    
    def freeze_encoder(self) -> None:
        """冻结编码器参数"""
        if self.depth_encoder is not None:
            for param in self.depth_encoder.parameters():
                param.requires_grad = False
            self.depth_encoder.eval()
    
    def select_action(
        self, 
        state: np.ndarray, 
        goal: np.ndarray, 
        deterministic: bool = False
    ) -> np.ndarray:
        """
        选择动作
        
        Args:
            state: 状态
            goal: 目标
            deterministic: 是否确定性
            
        Returns:
            动作
        """
        state_t = to_tensor(state.reshape(1, -1))
        goal_t = to_tensor(goal.reshape(1, -1))
        
        with torch.no_grad():
            action = self.actor.get_action(state_t, goal_t, deterministic)
        
        return to_numpy(action).flatten()
    
    def update(self, buffer: ReplayBuffer, n_iter: int, batch_size: int) -> None:
        """
        RL 更新
        
        Args:
            buffer: 经验缓冲区
            n_iter: 迭代次数
            batch_size: 批大小
        """
        # 判断是否阻断编码器梯度
        detach_encoder = (
            self.encoder_optimizer is None and 
            self.depth_encoder is not None
        )
        
        for _ in range(n_iter):
            # 采样
            states, actions, rewards, next_states, goals, gammas, dones = buffer.sample(batch_size)
            
            states = to_tensor(states)
            actions = to_tensor(actions)
            rewards = to_tensor(rewards).unsqueeze(1)
            next_states = to_tensor(next_states)
            goals = to_tensor(goals)
            gammas = to_tensor(gammas).unsqueeze(1)
            dones = to_tensor(dones).unsqueeze(1)
            
            # ===== Critic Update =====
            with torch.no_grad():
                next_action, next_log_prob = self.actor.sample(
                    next_states, goals, detach_encoder=True
                )
                q_next = self.critic(
                    next_states, next_action, goals, detach_encoder=True
                )
                target_q = rewards + (1 - dones) * gammas * (q_next - self.alpha * next_log_prob)
            
            q_loss = F.mse_loss(
                self.critic(states, actions, goals, detach_encoder=detach_encoder), 
                target_q
            )
            
            self.critic_optimizer.zero_grad()
            q_loss.backward()
            self.critic_optimizer.step()
            
            # ===== Actor Update =====
            new_action, log_prob = self.actor.sample(
                states, goals, detach_encoder=detach_encoder
            )
            q_new = self.critic(
                states, new_action, goals, detach_encoder=detach_encoder
            )
            actor_loss = (self.alpha * log_prob - q_new).mean()
            
            self.actor_optimizer.zero_grad()
            if self.encoder_optimizer is not None:
                self.encoder_optimizer.zero_grad()
            
            actor_loss.backward()
            
            self.actor_optimizer.step()
            if self.encoder_optimizer is not None:
                self.encoder_optimizer.step()
            
            # ===== Entropy Update =====
            if self.auto_entropy:
                alpha_loss = -(
                    self.log_alpha * (log_prob + self.target_entropy).detach()
                ).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp().item()
    
    def save(self, directory: str, name: str) -> None:
        """保存模型"""
        os.makedirs(directory, exist_ok=True)
        torch.save(self.actor.state_dict(), f'{directory}/{name}_actor.pth')
        torch.save(self.critic.state_dict(), f'{directory}/{name}_critic.pth')
        if self.depth_encoder is not None:
            torch.save(
                self.depth_encoder.state_dict(), 
                f'{directory}/{name}_encoder.pth'
            )
        
    def load(self, directory: str, name: str) -> None:
        """加载模型"""
        self.actor.load_state_dict(
            torch.load(f'{directory}/{name}_actor.pth', map_location='cpu', weights_only=True)
        )
        self.critic.load_state_dict(
            torch.load(f'{directory}/{name}_critic.pth', map_location='cpu', weights_only=True)
        )
        if self.depth_encoder is not None:
            encoder_path = f'{directory}/{name}_encoder.pth'
            if os.path.exists(encoder_path):
                self.depth_encoder.load_state_dict(
                    torch.load(encoder_path, map_location='cpu', weights_only=True)
                )
