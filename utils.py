"""
工具函数和类
"""
import numpy as np
from typing import Tuple, List
from collections import deque


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, max_size: int = 500000):
        """
        Args:
            max_size: 最大容量
        """
        self.buffer = deque(maxlen=max_size)
    
    def add(self, transition: Tuple):
        """
        添加一条转换记录
        
        Args:
            transition: (state, action, reward, next_state, goal, gamma, done)
        """
        assert len(transition) == 7, "transition must have length = 7"
        self.buffer.append(transition)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        随机采样一批数据
        
        Args:
            batch_size: 批大小
            
        Returns:
            states, actions, rewards, next_states, goals, gammas, dones
        """
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        
        states = []
        actions = []
        rewards = []
        next_states = []
        goals = []
        gammas = []
        dones = []
        
        for i in indices:
            s, a, r, ns, g, gam, d = self.buffer[i]
            states.append(np.asarray(s))
            actions.append(np.asarray(a))
            rewards.append(np.asarray(r))
            next_states.append(np.asarray(ns))
            goals.append(np.asarray(g))
            gammas.append(np.asarray(gam))
            dones.append(np.asarray(d))
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(goals),
            np.array(gammas),
            np.array(dones)
        )
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()


class Logger:
    """简单的训练日志记录器"""
    
    def __init__(self, log_file: str = "log.txt"):
        self.log_file = log_file
        self.episode_rewards = []
        self.episode_steps = []
        self.file = None
    
    def start(self):
        """开始记录"""
        self.file = open(self.log_file, 'w')
        self.file.write("episode,reward,steps\n")
    
    def log(self, episode: int, reward: float, steps: int):
        """记录一个episode"""
        self.episode_rewards.append(reward)
        self.episode_steps.append(steps)
        
        if self.file:
            self.file.write(f"{episode},{reward},{steps}\n")
            self.file.flush()
    
    def close(self):
        """关闭日志文件"""
        if self.file:
            self.file.close()
            self.file = None
    
    def get_stats(self, last_n: int = 100) -> dict:
        """获取最近n个episode的统计信息"""
        recent_rewards = self.episode_rewards[-last_n:]
        recent_steps = self.episode_steps[-last_n:]
        
        return {
            'mean_reward': np.mean(recent_rewards) if recent_rewards else 0,
            'std_reward': np.std(recent_rewards) if recent_rewards else 0,
            'mean_steps': np.mean(recent_steps) if recent_steps else 0,
            'max_reward': np.max(recent_rewards) if recent_rewards else 0,
            'min_reward': np.min(recent_rewards) if recent_rewards else 0,
        }
