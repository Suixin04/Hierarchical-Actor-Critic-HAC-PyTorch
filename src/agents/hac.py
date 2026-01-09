"""Hierarchical Actor-Critic 智能体

实现分层强化学习:
- 高层 (Level 1+): SAC 策略，输出子目标 (世界坐标)
- 底层 (Level 0): MPC 控制器，追踪子目标

设计原则:
- 每层梯度独立，不互相干扰
- 统一使用世界坐标系
- MPC 只负责追踪，不参与学习
"""

import torch
import numpy as np
from typing import Optional, List, Tuple
import os
import time

from src.agents.sac import SACAgent
from src.control.mpc import MPCController
from src.buffers.replay_buffer import ReplayBuffer
from src.utils.common import get_device, to_tensor


class HACAgent:
    """
    分层 Actor-Critic 智能体
    
    层级结构:
    - Level k-1 (最高层): SAC, 输出子目标给下层
    - ...
    - Level 1: SAC, 输出子目标给 MPC
    - Level 0: MPC 控制器, 输出原始动作 [a_v, a_ω]
    
    Args:
        config: 环境配置对象
        render: 是否渲染
    """
    
    def __init__(self, config, render: bool = False):
        self.config = config
        self.render = render
        self._device = get_device()
        
        # 从配置提取参数
        self.k_level = config.k_level
        self.H = config.H
        self.lamda = config.lamda
        self.gamma = config.gamma
        self.threshold = config.goal_threshold
        
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.goal_dim = config.goal_dim
        self.world_size = config.world_size
        
        # 子目标边界 (世界坐标)
        self.subgoal_bounds = torch.FloatTensor(
            config.get_subgoal_bounds().reshape(1, -1)
        ).to(self._device)
        self.subgoal_offset = torch.FloatTensor(
            config.get_subgoal_offset().reshape(1, -1)
        ).to(self._device)
        
        # 构建层级
        self._build_hierarchy(config)
        
        # 运行时状态
        self.goals: List[Optional[np.ndarray]] = [None] * self.k_level
        self.reward = 0.0
        self.timestep = 0
        
        self._print_summary()
    
    def _build_hierarchy(self, config) -> None:
        """构建分层策略"""
        self.policies: List = []
        self.buffers: List[Optional[ReplayBuffer]] = []
        
        # Level 0: MPC 控制器
        self.mpc = MPCController(
            horizon=config.mpc_horizon,
            dt=config.dt,
            max_v=config.max_v,
            max_omega=config.max_omega,
            max_a_v=config.max_a_v,
            max_a_omega=config.max_a_omega,
            damping_v=config.damping_v,
            damping_omega=config.damping_omega,
        )
        self.policies.append(self.mpc)
        self.buffers.append(None)  # MPC 不需要 buffer
        
        # Level 1+: SAC 策略
        for i in range(self.k_level - 1):
            policy = SACAgent(
                state_dim=self.state_dim,
                action_dim=self.goal_dim,  # 输出子目标
                goal_dim=self.goal_dim,
                action_bounds=self.subgoal_bounds,
                action_offset=self.subgoal_offset,
                lr=config.lr,
                H=self.H,
                hidden_dim=config.hidden_dim,
                alpha=config.sac_alpha,
                auto_entropy=config.sac_auto_entropy,
            )
            self.policies.append(policy)
            self.buffers.append(ReplayBuffer())
    
    def _print_summary(self) -> None:
        """打印初始化摘要"""
        print(f"HAC initialized:")
        print(f"  Levels: {self.k_level}")
        print(f"  High-level (Level 1+): SAC")
        print(f"  Low-level (Level 0): MPC (horizon={self.config.mpc_horizon})")
        print(f"  State dim: {self.state_dim}, Goal dim: {self.goal_dim}")
    
    def check_goal(
        self, 
        state: np.ndarray, 
        goal: np.ndarray, 
        threshold: float
    ) -> bool:
        """检查是否达成目标"""
        pos = state[:2]  # [x, y]
        return np.linalg.norm(pos - goal[:2]) <= threshold
    
    def run_HAC(
        self,
        env,
        level: int,
        state: np.ndarray,
        goal: np.ndarray,
        is_subgoal_test: bool
    ) -> Tuple[np.ndarray, bool]:
        """
        运行分层策略 (训练模式)
        
        Args:
            env: 环境
            level: 当前层级
            state: 当前状态
            goal: 目标
            is_subgoal_test: 是否为子目标测试模式
            
        Returns:
            (最终状态, 是否结束)
        """
        done = False
        goal_transitions = []
        
        self.goals[level] = goal
        
        for _ in range(self.H):
            if level > 0:
                # 高层: SAC 输出子目标
                next_state, done, subgoal = self._run_high_level(
                    env, level, state, goal, is_subgoal_test, goal_transitions
                )
            else:
                # 底层: MPC 控制
                next_state, done = self._run_low_level(env, state, goal)
                subgoal = None
            
            state = next_state
            
            if done or self.check_goal(next_state, goal, self.threshold):
                break
        
        # Hindsight Goal Transitions
        if level > 0 and goal_transitions:
            self._add_hindsight_transitions(level, next_state, goal_transitions)
        
        return next_state, done
    
    def _run_high_level(
        self,
        env,
        level: int,
        state: np.ndarray,
        goal: np.ndarray,
        is_subgoal_test: bool,
        goal_transitions: List
    ) -> Tuple[np.ndarray, bool, np.ndarray]:
        """运行高层策略"""
        policy = self.policies[level]
        
        # 选择子目标 (世界坐标)
        subgoal = policy.select_action(state, goal, deterministic=is_subgoal_test)
        
        # 边界裁剪
        margin = 0.1
        subgoal = np.clip(subgoal, margin, self.world_size - margin)
        
        # 递归调用下一层
        is_next_test = is_subgoal_test or np.random.random() < self.lamda
        next_state, done = self.run_HAC(env, level - 1, state, subgoal, is_next_test)
        
        # 子目标测试: 如果未达成子目标，添加惩罚转换
        if is_next_test and not self.check_goal(next_state, subgoal, self.threshold):
            self.buffers[level].add((
                state, subgoal, -self.H, next_state, goal, 0.0, float(done)
            ))
        
        # 存储正常转换
        self._store_transition(
            level, state, subgoal, next_state, goal, done, goal_transitions
        )
        
        return next_state, done, subgoal
    
    def _run_low_level(
        self,
        env,
        state: np.ndarray,
        goal: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        """运行底层 MPC"""
        action = self.mpc.select_action(state, goal)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # 渲染：显示所有层级的子目标
        if self.render:
            self._render_with_goals(env, next_state)
        
        self.reward += reward
        self.timestep += 1
        
        return next_state, done
    
    def _store_transition(
        self,
        level: int,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        goal: np.ndarray,
        done: bool,
        goal_transitions: List
    ) -> None:
        """存储转换"""
        goal_achieved = self.check_goal(next_state, goal, self.threshold)
        
        if goal_achieved:
            self.buffers[level].add((
                state, action, 0.0, next_state, goal, 0.0, float(done)
            ))
        else:
            self.buffers[level].add((
                state, action, -1.0, next_state, goal, self.gamma, float(done)
            ))
        
        # 保存用于 Hindsight
        goal_transitions.append([
            state, action, -1.0, next_state, None, self.gamma, float(done)
        ])
    
    def _add_hindsight_transitions(
        self,
        level: int,
        final_state: np.ndarray,
        goal_transitions: List
    ) -> None:
        """添加 Hindsight Goal 转换"""
        hindsight_goal = final_state[:2].copy()  # [x, y]
        
        goal_transitions[-1][2] = 0.0   # 最后一步 reward = 0
        goal_transitions[-1][5] = 0.0   # gamma = 0
        
        for transition in goal_transitions:
            transition[4] = hindsight_goal
            self.buffers[level].add(tuple(transition))
    
    def run_HAC_inference(
        self,
        env,
        level: int,
        state: np.ndarray,
        goal: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        """
        推理模式的 HAC 执行
        
        与训练模式的区别:
        - 不存储转换
        - 使用确定性策略
        """
        done = False
        self.goals[level] = goal
        
        for _ in range(self.H):
            if level > 0:
                # 高层: SAC
                policy = self.policies[level]
                subgoal = policy.select_action(state, goal, deterministic=True)
                
                # 边界裁剪
                margin = 0.1
                subgoal = np.clip(subgoal, margin, self.world_size - margin)
                
                # 递归
                next_state, done = self.run_HAC_inference(env, level - 1, state, subgoal)
            else:
                # 底层: MPC
                action = self.mpc.select_action(state, goal)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # 渲染：显示所有层级的子目标
                if self.render:
                    self._render_with_goals(env, next_state)
                
                self.reward += reward
                self.timestep += 1
            
            state = next_state
            
            if done or self.check_goal(next_state, goal, self.threshold):
                break
        
        return next_state, done
    
    def _render_with_goals(self, env, state: np.ndarray) -> None:
        """
        渲染带有所有层级子目标的场景
        
        Args:
            env: 环境
            state: 当前状态
        """
        # 收集所有层级的目标 (从底层到高层)
        goals_to_render = []
        for i in range(self.k_level):
            if self.goals[i] is not None:
                goals_to_render.append(self.goals[i])
            else:
                goals_to_render.append(None)
        
        # 调用环境的渲染方法
        if hasattr(env, 'unwrapped'):
            env.unwrapped.render_subgoals(goals_to_render)
        else:
            env.render_subgoals(goals_to_render)
        
        # 添加延时以便观察
        time.sleep(0.03)
    
    def update(self, n_iter: int, batch_size: int) -> dict:
        """更新所有高层策略"""
        stats = {}
        for i in range(1, self.k_level):
            if self.buffers[i] is not None and len(self.buffers[i]) > batch_size:
                level_stats = self.policies[i].update(self.buffers[i], n_iter, batch_size)
                stats[f'level_{i}'] = level_stats
        return stats
    
    def reset_episode(self) -> None:
        """重置 episode"""
        self.reward = 0.0
        self.timestep = 0
        self.goals = [None] * self.k_level
        self.mpc.reset()
    
    def reset(self) -> None:
        """重置智能体"""
        self.reset_episode()
    
    def save(self, directory: str, name: str) -> None:
        """保存模型"""
        os.makedirs(directory, exist_ok=True)
        for i in range(1, self.k_level):
            self.policies[i].save(directory, f'{name}_level_{i}')
        print(f"  [SAVE] {directory}/{name}")
    
    def load(self, directory: str, name: str) -> None:
        """加载模型"""
        for i in range(1, self.k_level):
            self.policies[i].load(directory, f'{name}_level_{i}')
        print(f"  [LOAD] {directory}/{name}")
