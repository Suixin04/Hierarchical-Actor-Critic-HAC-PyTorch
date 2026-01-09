"""Soft Actor-Critic 智能体

标准 SAC 实现，用于 HAC 的高层策略。

每层 SAC 独立训练，梯度不互相干扰。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Optional
import os

from src.models.actor import GaussianActor
from src.models.critic import SoftQNetwork
from src.buffers.replay_buffer import ReplayBuffer
from src.utils.common import get_device, to_tensor, to_numpy


class SACAgent:
    """
    Soft Actor-Critic 智能体
    
    每个 HAC 层级对应一个独立的 SAC 智能体。
    
    Args:
        state_dim: 状态维度
        action_dim: 动作/子目标维度
        goal_dim: 目标维度
        action_bounds: 动作缩放因子
        action_offset: 动作偏移
        lr: 学习率
        H: 时间范围 (Q 值边界)
        hidden_dim: 隐藏层维度
        alpha: 初始熵系数
        auto_entropy: 是否自动调节熵
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
        hidden_dim: int = 128,
        alpha: float = 0.2,
        auto_entropy: bool = True,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.H = H
        
        self._device = get_device()
        
        # Actor
        self.actor = GaussianActor(
            state_dim, action_dim, goal_dim,
            action_bounds, action_offset, hidden_dim
        ).to(self._device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        # Critic
        self.critic = SoftQNetwork(
            state_dim, action_dim, goal_dim, H, hidden_dim
        ).to(self._device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # 自动熵调节
        self.auto_entropy = auto_entropy
        if auto_entropy:
            self.target_entropy = -action_dim
            init_log_alpha = np.log(alpha) if alpha > 0 else 0.0
            self.log_alpha = torch.tensor(
                [init_log_alpha], requires_grad=True, device=self._device
            )
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha
    
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
            action: 子目标 (世界坐标)
        """
        state_t = to_tensor(state.reshape(1, -1))
        goal_t = to_tensor(goal.reshape(1, -1))
        
        with torch.no_grad():
            action = self.actor.get_action(state_t, goal_t, deterministic)
        
        return to_numpy(action).flatten()
    
    def update(self, buffer: ReplayBuffer, n_iter: int, batch_size: int) -> dict:
        """
        RL 更新
        
        Args:
            buffer: 经验缓冲区
            n_iter: 迭代次数
            batch_size: 批大小
            
        Returns:
            训练统计 (包含loss和梯度信息)
        """
        if len(buffer) < batch_size:
            return {}
        
        total_critic_loss = 0.0
        total_actor_loss = 0.0
        
        # 梯度统计累积
        total_critic_grad_norm = 0.0
        total_critic_grad_max = 0.0
        total_critic_grad_mean = 0.0
        critic_layer_grads_accum = {}
        
        for _ in range(n_iter):
            # 采样
            batch = buffer.sample(batch_size)
            states, actions, rewards, next_states, goals, gammas, dones = batch
            
            states = to_tensor(states)
            actions = to_tensor(actions)
            rewards = to_tensor(rewards).unsqueeze(1)
            next_states = to_tensor(next_states)
            goals = to_tensor(goals)
            gammas = to_tensor(gammas).unsqueeze(1)
            dones = to_tensor(dones).unsqueeze(1)
            
            # ===== Critic Update =====
            with torch.no_grad():
                next_action, next_log_prob = self.actor.sample(next_states, goals)
                q_next = self.critic(next_states, next_action, goals)
                # 注意: SAC 的熵项 (-alpha * log_prob) 是正的，可能导致 target > 0
                # 但 HAC 要求 Q ∈ [-H, 0]，所以需要 clamp
                # 这里采用 clamp 而非扩展边界，保持 HAC 的语义：Q 值代表到达目标的负步数
                target_q_raw = rewards + (1 - dones) * gammas * (q_next - self.alpha * next_log_prob)
                target_q = torch.clamp(target_q_raw, -self.H, 0)
            
            q_pred = self.critic(states, actions, goals)
            critic_loss = F.mse_loss(q_pred, target_q)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            
            # 收集 Critic 梯度统计
            grad_stats = self._compute_gradient_stats(self.critic, 'critic')
            total_critic_grad_norm += grad_stats['grad_norm']
            total_critic_grad_max += grad_stats['grad_max']
            total_critic_grad_mean += grad_stats['grad_mean']
            
            # 收集各层梯度
            for layer_name, layer_grad in grad_stats['layer_grads'].items():
                if layer_name not in critic_layer_grads_accum:
                    critic_layer_grads_accum[layer_name] = {'norm': 0.0, 'mean': 0.0, 'max': 0.0}
                critic_layer_grads_accum[layer_name]['norm'] += layer_grad['norm']
                critic_layer_grads_accum[layer_name]['mean'] += layer_grad['mean']
                critic_layer_grads_accum[layer_name]['max'] += layer_grad['max']
            
            self.critic_optimizer.step()
            
            total_critic_loss += critic_loss.item()
            
            # ===== Actor Update =====
            new_action, log_prob = self.actor.sample(states, goals)
            q_new = self.critic(states, new_action, goals)
            actor_loss = (self.alpha * log_prob - q_new).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            total_actor_loss += actor_loss.item()
            
            # ===== Entropy Update =====
            if self.auto_entropy:
                alpha_loss = -(
                    self.log_alpha * (log_prob + self.target_entropy).detach()
                ).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp().item()
        
        # 平均梯度统计
        critic_layer_grads_avg = {}
        for layer_name, grads in critic_layer_grads_accum.items():
            critic_layer_grads_avg[layer_name] = {
                'norm': grads['norm'] / n_iter,
                'mean': grads['mean'] / n_iter,
                'max': grads['max'] / n_iter,
            }
        
        return {
            'critic_loss': total_critic_loss / n_iter,
            'actor_loss': total_actor_loss / n_iter,
            'alpha': self.alpha,
            # 梯度统计
            'critic_grad_norm': total_critic_grad_norm / n_iter,
            'critic_grad_max': total_critic_grad_max / n_iter,
            'critic_grad_mean': total_critic_grad_mean / n_iter,
            'critic_layer_grads': critic_layer_grads_avg,
        }
    
    def _compute_gradient_stats(self, model: nn.Module, model_name: str) -> dict:
        """
        计算模型梯度统计
        
        Args:
            model: 神经网络模型
            model_name: 模型名称 (用于日志)
            
        Returns:
            梯度统计字典
        """
        all_grads = []
        layer_grads = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach().cpu()
                grad_flat = grad.flatten()
                all_grads.append(grad_flat)
                
                # 记录各层梯度
                layer_name = name.replace('.', '_')
                layer_grads[layer_name] = {
                    'norm': grad.norm().item(),
                    'mean': grad.abs().mean().item(),
                    'max': grad.abs().max().item(),
                }
        
        if all_grads:
            all_grads_flat = torch.cat(all_grads)
            return {
                'grad_norm': all_grads_flat.norm().item(),
                'grad_max': all_grads_flat.abs().max().item(),
                'grad_mean': all_grads_flat.abs().mean().item(),
                'layer_grads': layer_grads,
            }
        else:
            return {
                'grad_norm': 0.0,
                'grad_max': 0.0,
                'grad_mean': 0.0,
                'layer_grads': {},
            }
    
    def save(self, directory: str, name: str) -> None:
        """保存模型"""
        os.makedirs(directory, exist_ok=True)
        torch.save(self.actor.state_dict(), f'{directory}/{name}_actor.pth')
        torch.save(self.critic.state_dict(), f'{directory}/{name}_critic.pth')
    
    def load(self, directory: str, name: str) -> None:
        """加载模型"""
        self.actor.load_state_dict(
            torch.load(f'{directory}/{name}_actor.pth', map_location='cpu', weights_only=True)
        )
        self.critic.load_state_dict(
            torch.load(f'{directory}/{name}_critic.pth', map_location='cpu', weights_only=True)
        )
