"""
训练指标收集和监控

提供训练过程中的各类指标记录和分析功能
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
import os


@dataclass
class EpisodeStats:
    """单回合统计"""
    episode: int
    reward: float
    steps: int
    success: bool
    phase: int  # 1 = E2E, 2 = RL
    e2e_loss: Optional[float] = None
    actor_loss: Optional[float] = None
    critic_loss: Optional[float] = None
    alpha: Optional[float] = None


class TrainingMetrics:
    """
    训练指标收集器
    
    收集并分析训练过程中的各类指标，用于：
    1. 监控训练进度
    2. 早停判断
    3. 可视化分析
    4. 超参数调优
    """
    
    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: 滑动窗口大小，用于计算移动平均
        """
        self.window_size = window_size
        
        # 回合统计
        self.episodes: List[EpisodeStats] = []
        
        # 滑动窗口
        self.reward_window = deque(maxlen=window_size)
        self.steps_window = deque(maxlen=window_size)
        self.success_window = deque(maxlen=window_size)
        
        # 编码器相关指标
        self.encoder_metrics: List[Dict] = []
        
        # 最佳记录
        self.best_reward = float('-inf')
        self.best_episode = 0
        self.best_success_rate = 0.0
        
        # 阶段切换记录
        self.phase_switch_episode: Optional[int] = None
    
    def log_episode(
        self,
        episode: int,
        reward: float,
        steps: int,
        success: bool,
        phase: int,
        e2e_loss: Optional[float] = None,
        actor_loss: Optional[float] = None,
        critic_loss: Optional[float] = None,
        alpha: Optional[float] = None
    ) -> None:
        """记录一个回合的统计"""
        stats = EpisodeStats(
            episode=episode,
            reward=reward,
            steps=steps,
            success=success,
            phase=phase,
            e2e_loss=e2e_loss,
            actor_loss=actor_loss,
            critic_loss=critic_loss,
            alpha=alpha
        )
        self.episodes.append(stats)
        
        # 更新滑动窗口
        self.reward_window.append(reward)
        self.steps_window.append(steps)
        self.success_window.append(float(success))
        
        # 更新最佳记录
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_episode = episode
        
        success_rate = self.get_success_rate()
        if success_rate > self.best_success_rate:
            self.best_success_rate = success_rate
    
    def log_encoder_metrics(
        self,
        episode: int,
        embedding_variance: float,
        gradient_norm: float,
        feature_correlation: Optional[Dict[str, float]] = None
    ) -> None:
        """记录编码器指标"""
        metrics = {
            'episode': episode,
            'embedding_variance': embedding_variance,
            'gradient_norm': gradient_norm,
            'feature_correlation': feature_correlation or {}
        }
        self.encoder_metrics.append(metrics)
    
    def log_phase_switch(self, episode: int) -> None:
        """记录阶段切换"""
        self.phase_switch_episode = episode
    
    def get_mean_reward(self) -> float:
        """获取滑动窗口平均奖励"""
        if not self.reward_window:
            return 0.0
        return np.mean(self.reward_window)
    
    def get_mean_steps(self) -> float:
        """获取滑动窗口平均步数"""
        if not self.steps_window:
            return 0.0
        return np.mean(self.steps_window)
    
    def get_success_rate(self) -> float:
        """获取滑动窗口成功率"""
        if not self.success_window:
            return 0.0
        return np.mean(self.success_window)
    
    def get_improvement_rate(self, last_n: int = 100) -> float:
        """
        计算最近 n 个回合的改进率
        
        Returns:
            改进率 (正值表示在改进)
        """
        if len(self.episodes) < last_n * 2:
            return 0.0
        
        recent = [e.reward for e in self.episodes[-last_n:]]
        previous = [e.reward for e in self.episodes[-2*last_n:-last_n]]
        
        return np.mean(recent) - np.mean(previous)
    
    def should_early_stop(
        self,
        patience: int = 200,
        min_improvement: float = 0.01
    ) -> bool:
        """
        判断是否应该早停
        
        Args:
            patience: 无改进的最大回合数
            min_improvement: 最小改进阈值
        """
        if len(self.episodes) < patience:
            return False
        
        # 检查最近 patience 个回合是否有显著改进
        episodes_since_best = len(self.episodes) - 1 - self.best_episode
        
        if episodes_since_best > patience:
            recent_improvement = self.get_improvement_rate(patience // 2)
            if recent_improvement < min_improvement:
                return True
        
        return False
    
    def get_summary(self) -> Dict:
        """获取训练总结"""
        if not self.episodes:
            return {}
        
        rewards = [e.reward for e in self.episodes]
        steps = [e.steps for e in self.episodes]
        successes = [e.success for e in self.episodes]
        
        # Phase 1 和 Phase 2 分开统计
        p1_rewards = [e.reward for e in self.episodes if e.phase == 1]
        p2_rewards = [e.reward for e in self.episodes if e.phase == 2]
        
        summary = {
            'total_episodes': len(self.episodes),
            'best_reward': self.best_reward,
            'best_episode': self.best_episode,
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_steps': np.mean(steps),
            'success_rate': np.mean(successes),
            'phase_switch_episode': self.phase_switch_episode,
            'phase1_mean_reward': np.mean(p1_rewards) if p1_rewards else 0.0,
            'phase2_mean_reward': np.mean(p2_rewards) if p2_rewards else 0.0,
            'final_100_mean_reward': np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards),
            'final_100_success_rate': np.mean(successes[-100:]) if len(successes) >= 100 else np.mean(successes),
        }
        
        return summary
    
    def get_rewards_array(self) -> np.ndarray:
        """获取所有奖励数组"""
        return np.array([e.reward for e in self.episodes])
    
    def get_e2e_losses_array(self) -> np.ndarray:
        """获取 E2E 损失数组"""
        losses = [e.e2e_loss for e in self.episodes if e.e2e_loss is not None]
        return np.array(losses) if losses else np.array([])
    
    def save(self, filepath: str) -> None:
        """保存指标到文件"""
        data = {
            'episodes': [
                {
                    'episode': e.episode,
                    'reward': e.reward,
                    'steps': e.steps,
                    'success': e.success,
                    'phase': e.phase,
                    'e2e_loss': e.e2e_loss,
                    'actor_loss': e.actor_loss,
                    'critic_loss': e.critic_loss,
                    'alpha': e.alpha,
                }
                for e in self.episodes
            ],
            'encoder_metrics': self.encoder_metrics,
            'summary': self.get_summary(),
        }
        
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved metrics to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'TrainingMetrics':
        """从文件加载指标"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        metrics = cls()
        for ep_data in data['episodes']:
            metrics.log_episode(
                episode=ep_data['episode'],
                reward=ep_data['reward'],
                steps=ep_data['steps'],
                success=ep_data['success'],
                phase=ep_data['phase'],
                e2e_loss=ep_data.get('e2e_loss'),
                actor_loss=ep_data.get('actor_loss'),
                critic_loss=ep_data.get('critic_loss'),
                alpha=ep_data.get('alpha'),
            )
        
        metrics.encoder_metrics = data.get('encoder_metrics', [])
        return metrics
    
    def print_progress(self, episode: int, phase: int) -> str:
        """生成进度字符串"""
        phase_str = f"[P{phase}]"
        reward_str = f"R:{self.episodes[-1].reward:.2f}" if self.episodes else "R:--"
        mean_str = f"Avg:{self.get_mean_reward():.2f}"
        sr_str = f"SR:{self.get_success_rate()*100:.1f}%"
        best_str = f"Best:{self.best_reward:.2f}@{self.best_episode}"
        
        return f"{phase_str} Ep:{episode} {reward_str} {mean_str} {sr_str} {best_str}"
