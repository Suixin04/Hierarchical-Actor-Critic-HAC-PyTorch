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
        if self.depth_encoder is not None and encoder_train_mode in ['rl', 'finetune']:
            # 'rl' 模式：正常学习率
            # 'finetune' 模式：初始化优化器，但后续可通过 setup_encoder_finetune 调整学习率
            encoder_lr = lr if encoder_train_mode == 'rl' else lr * 0.1  # finetune 用较小学习率
            self.encoder_optimizer = optim.Adam(
                self.depth_encoder.parameters(), lr=encoder_lr
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
    
    def update_with_privileged_loss(
        self, 
        buffer: ReplayBuffer, 
        n_iter: int, 
        batch_size: int,
        obstacles: list,
        skeleton_weight: float = 1.0,
        safe_distance: float = 0.5
    ) -> dict:
        """
        带特权学习的 RL 更新
        
        设计原理 (基于 Actor-Critic 框架):
        - Critic 负责评估：通过骨架引导损失学会给危险子目标低 Q 值
        - Encoder 通过 Critic 更新获得梯度：学习"哪里有障碍物"
        - Actor 通过 max Q 间接学会避障：不直接接收特权信息
        
        这样的设计：
        1. 不与 SAC 的最大熵目标冲突
        2. 更符合 RL 的 Actor-Critic 原理
        3. 避免直接约束 Actor 导致探索受限
        
        Args:
            buffer: 经验缓冲区
            n_iter: 迭代次数
            batch_size: 批大小
            obstacles: 障碍物列表 [(x, y, radius), ...]
            skeleton_weight: 骨架引导损失权重
            safe_distance: 安全距离
            
        Returns:
            训练统计信息
        """
        if len(buffer) < batch_size:
            return {'skeleton_loss': 0.0}
        
        # Critic 更新时不 detach encoder，让梯度流过
        # Actor 更新时 detach encoder，只通过 Q 值间接影响
        
        total_skeleton_loss = 0.0
        
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
            
            # ===== Critic Update with Skeleton Guidance =====
            # 特权信息通过 Critic 更新流向 Encoder
            with torch.no_grad():
                next_action, next_log_prob = self.actor.sample(
                    next_states, goals, detach_encoder=True
                )
                q_next = self.critic(
                    next_states, next_action, goals, detach_encoder=True
                )
                target_q = rewards + (1 - dones) * gammas * (q_next - self.alpha * next_log_prob)
            
            # Critic 前向传播：不 detach encoder，让梯度流过
            q_pred = self.critic(states, actions, goals, detach_encoder=False)
            q_loss = F.mse_loss(q_pred, target_q)
            
            # 骨架引导损失：惩罚 Critic 对危险子目标的高估
            # 这让 Critic 学会给危险子目标低 Q 值
            skeleton_loss = self._compute_critic_skeleton_loss(
                states, actions, goals, obstacles, safe_distance
            )
            
            critic_total_loss = q_loss + skeleton_weight * skeleton_loss
            total_skeleton_loss += skeleton_loss.item()
            
            self.critic_optimizer.zero_grad()
            if self.encoder_optimizer is not None:
                self.encoder_optimizer.zero_grad()
            
            critic_total_loss.backward()
            
            self.critic_optimizer.step()
            if self.encoder_optimizer is not None:
                self.encoder_optimizer.step()
            
            # ===== Actor Update (标准 SAC，通过 Q 值间接学习) =====
            # Actor 不直接接收特权信息，而是通过 max Q 学习
            new_action, log_prob = self.actor.sample(
                states, goals, detach_encoder=True  # detach encoder
            )
            q_new = self.critic(
                states, new_action, goals, detach_encoder=True  # detach encoder
            )
            
            actor_loss = (self.alpha * log_prob - q_new).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # ===== Entropy Update =====
            if self.auto_entropy:
                alpha_loss = -(
                    self.log_alpha * (log_prob + self.target_entropy).detach()
                ).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp().item()
        
        return {'skeleton_loss': total_skeleton_loss / max(n_iter, 1)}
    
    def _compute_critic_skeleton_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        goals: torch.Tensor,
        obstacles: list,
        safe_distance: float = 0.5
    ) -> torch.Tensor:
        """
        计算 Critic 的骨架引导损失 (改进版)
        
        原理：让 Critic 学会给危险子目标（靠近障碍物）低 Q 值
        
        改进：不使用 (Q + H) 作为因子，因为 Bounded Q 容易饱和导致梯度消失
        改为直接用 MSE 将危险子目标的 Q 值推向 -H
        
        Args:
            states: 状态 [batch, state_dim]
            actions: 动作/子目标 [batch, action_dim]
            goals: 目标 [batch, goal_dim]
            obstacles: 障碍物列表
            safe_distance: 安全距离
            
        Returns:
            骨架引导损失
        """
        if not obstacles:
            return torch.tensor(0.0, device=self._device)
        
        # 计算当前 Q 值
        q_pred = self.critic(states, actions, goals, detach_encoder=False)
        
        # 计算子目标的危险程度 (0 = 安全, >0 = 危险)
        danger_score = torch.zeros(actions.shape[0], device=self._device)
        
        for ox, oy, orad in obstacles:
            center = torch.tensor([ox, oy], device=self._device, dtype=torch.float32)
            dist_to_obs = torch.norm(actions[:, :2] - center, dim=1)
            min_safe = orad + safe_distance
            
            # 危险程度：越近越危险，使用 softplus 平滑
            danger = F.softplus(min_safe - dist_to_obs, beta=5.0)
            danger_score = danger_score + danger
        
        # 改进的损失计算：
        # 1. 对于危险子目标 (danger > 0)，将 Q 值推向 -H
        # 2. 使用 MSE 损失，避免 (Q+H) 饱和问题
        # 3. danger_score 作为权重，越危险惩罚越大
        
        # 目标 Q 值：危险子目标应该是 -H
        target_q_for_danger = torch.full_like(q_pred, -self.H)
        
        # 加权 MSE：只惩罚危险的子目标
        # 当 danger_score = 0 时，损失 = 0
        # 当 danger_score > 0 时，将 Q 推向 -H
        loss = (danger_score * (q_pred.squeeze() - target_q_for_danger.squeeze()) ** 2).mean()
        
        return loss
    
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
