"""Hierarchical Actor-Critic 智能体

实现分层强化学习:
- 高层 (Level 1+): SAC 策略网络，输出子目标
- 底层 (Level 0): MPC 控制器，追踪子目标

特点:
- 底层使用已知动力学模型，无需学习基础控制
- 高层专注于学习高级规划策略
- Level 1 使用 MPC 预测可达性
- 支持端到端梯度回传
"""

import torch
import torch.optim as optim
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
import os

from src.agents.sac import SACAgent
from src.control.mpc import MPCController
from src.buffers.replay_buffer import ReplayBuffer
from src.utils.common import get_device, to_tensor, to_numpy
from src.utils.coordinate import world_to_polar


class HACAgent:
    """
    分层 Actor-Critic 智能体
    
    高层: SAC 策略 (学习输出子目标)
    底层: MPC 控制器 (追踪子目标)
    
    Args:
        config: 环境配置对象
        render: 是否渲染
    """
    
    def __init__(self, config: Any, render: bool = False):
        self.config = config
        self.render = render
        self._device = get_device()
        
        # 从配置提取参数
        self._init_dimensions(config)
        self._init_bounds(config)
        self._init_algorithm_params(config)
        self._init_mpc_params(config)
        
        # 构建层级策略
        self._build_hierarchy(config.lr)
        
        # 运行时状态
        self.goals: List[Optional[np.ndarray]] = [None] * self.k_level
        self.goals_world: List[Optional[np.ndarray]] = [None] * self.k_level
        self.reward = 0.0
        self.timestep = 0
        self.privileged_obstacles: List[Tuple[float, float, float]] = []
        
        self._print_summary()
    
    def _init_dimensions(self, config: Any) -> None:
        """初始化维度参数"""
        self.k_level = config.k_level
        self.H = config.H
        
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.goal_dim = config.effective_goal_dim
        self.goal_indices = config.goal_indices
        
        # 深度编码器参数
        self.use_depth_encoder = getattr(config, 'use_depth_encoder', False)
        self.base_state_dim = getattr(config, 'base_state_dim', 5)
        self.depth_dim = getattr(config, 'depth_dim', 0)
        self.embedding_dim = getattr(config, 'embedding_dim', 8)
        self.depth_max_range = getattr(config, 'depth_max_range', 5.0)
    
    def _init_bounds(self, config: Any) -> None:
        """初始化边界参数"""
        # 子目标空间
        self.subgoal_bounds = torch.FloatTensor(
            config.get_subgoal_bounds(level=2).reshape(1, -1)
        ).to(self._device)
        self.subgoal_offset = torch.FloatTensor(
            config.get_subgoal_offset(level=2).reshape(1, -1)
        ).to(self._device)
        
        # Level 1 极坐标参数
        self.level1_use_polar = getattr(config, 'level1_use_polar', False)
        if self.level1_use_polar:
            self.subgoal_r_min = config.subgoal_r_min
            self.subgoal_r_max = config.subgoal_r_max
            self.subgoal_fov = config.subgoal_fov
            self.subgoal_safety_margin = getattr(config, 'subgoal_safety_margin', 0.3)
            self.depth_rays = getattr(config, 'depth_rays', 16)
            self.depth_fov = getattr(config, 'depth_fov', 2 * np.pi)
        
        # 动作空间
        self.action_bounds = torch.FloatTensor(
            config.action_bounds.reshape(1, -1)
        ).to(self._device)
        self.action_offset = torch.FloatTensor(
            config.action_offset.reshape(1, -1)
        ).to(self._device)
        
        self.boundary_margin = getattr(config, 'boundary_margin', 0.1)
    
    def _init_algorithm_params(self, config: Any) -> None:
        """初始化算法参数"""
        self.lamda = config.lamda
        self.gamma = config.gamma
        self.threshold = config.goal_threshold
        
        # SAC 参数
        self.hidden_dim = getattr(config, 'hidden_dim', 64)
        self.sac_alpha = getattr(config, 'sac_alpha', 0.2)
        self.sac_auto_entropy = getattr(config, 'sac_auto_entropy', True)
        self.sac_target_entropy = getattr(config, 'sac_target_entropy', None)
        self.sac_alpha_lr = getattr(config, 'sac_alpha_lr', None)
        
        self.encoder_finetune_lr = getattr(config, 'encoder_finetune_lr', None)
    
    def _init_mpc_params(self, config: Any) -> None:
        """初始化 MPC 参数"""
        self.dt = getattr(config, 'dt', 0.1)
        self.max_v = getattr(config, 'max_v', 2.0)
        self.max_omega = getattr(config, 'max_omega', 2.0)
        self.max_a_v = getattr(config, 'max_a_v', 1.0)
        self.max_a_omega = getattr(config, 'max_a_omega', 2.0)
        self.damping_v = getattr(config, 'damping_v', 0.95)
        self.damping_omega = getattr(config, 'damping_omega', 0.9)
        self.mpc_iterations = getattr(config, 'mpc_iterations', 5)
        self.mpc_lr = getattr(config, 'mpc_lr', 0.5)
        self.mpc_Q = getattr(config, 'mpc_Q', [10.0, 10.0])
        self.mpc_R = getattr(config, 'mpc_R', [0.1, 0.1])
        self.mpc_Qf = getattr(config, 'mpc_Qf', [20.0, 20.0])
        self.mpc_reachability_threshold = getattr(config, 'mpc_reachability_threshold', 0.8)
    
    def _build_hierarchy(self, lr: float) -> None:
        """构建分层策略"""
        self.policies: List = []
        self.buffers: List[Optional[ReplayBuffer]] = []
        
        # Level 0: MPC 控制器
        Q = torch.FloatTensor(self.mpc_Q)
        R = torch.FloatTensor(self.mpc_R)
        Qf = torch.FloatTensor(self.mpc_Qf)
        
        self.mpc = MPCController(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            goal_dim=self.goal_dim,
            horizon=self.H,
            dt=self.dt,
            max_v=self.max_v,
            max_omega=self.max_omega,
            max_a_v=self.max_a_v,
            max_a_omega=self.max_a_omega,
            damping_v=self.damping_v,
            damping_omega=self.damping_omega,
            num_iterations=self.mpc_iterations,
            lr=self.mpc_lr,
            Q=Q, R=R, Qf=Qf,
        )
        
        self.policies.append(self.mpc)
        self.buffers.append(None)
        
        # Level 1+: SAC
        for i in range(self.k_level - 1):
            level = i + 1
            encoder_mode = 'finetune' if level == 1 else 'rl'
            
            policy = SACAgent(
                state_dim=self.state_dim,
                action_dim=self.goal_dim,
                goal_dim=self.goal_dim,
                action_bounds=self.subgoal_bounds,
                action_offset=self.subgoal_offset,
                lr=lr,
                H=self.H,
                hidden_dim=self.hidden_dim,
                alpha=self.sac_alpha,
                auto_entropy=self.sac_auto_entropy,
                target_entropy=self.sac_target_entropy,
                alpha_lr=self.sac_alpha_lr,
                use_depth_encoder=self.use_depth_encoder,
                base_state_dim=self.base_state_dim,
                depth_dim=self.depth_dim,
                embedding_dim=self.embedding_dim,
                level=level,
                encoder_train_mode=encoder_mode,
            )
            
            self.policies.append(policy)
            self.buffers.append(ReplayBuffer())
    
    def _print_summary(self) -> None:
        """打印初始化摘要"""
        print(f"HAC initialized:")
        print(f"  High-level (Level 1+): SAC")
        print(f"  Low-level (Level 0): MPC (horizon={self.H})")
        print(f"  Dynamics: damping_v={self.damping_v}, damping_omega={self.damping_omega}")
        if self.level1_use_polar:
            print(f"  Level 1 Safety: Depth constraint as post-processing")
        if self.use_depth_encoder:
            print(f"  Depth Encoder: {self.depth_dim}D → {self.embedding_dim}D")
    
    # ==================== 坐标转换 ====================
    
    def extract_goal_state(self, state: np.ndarray) -> np.ndarray:
        """从状态提取目标相关部分"""
        if self.goal_indices is not None:
            return state[self.goal_indices]
        return state
    
    def apply_depth_constraint(
        self, 
        state: np.ndarray, 
        subgoal: np.ndarray
    ) -> np.ndarray:
        """
        应用基于深度的安全约束
        
        Args:
            state: 当前状态
            subgoal: 世界坐标子目标 [x, y]
            
        Returns:
            约束后的子目标 [x, y]
        """
        if not self.level1_use_polar:
            return subgoal
        
        agent_pos = state[:2]
        agent_theta = state[2]
        
        # 世界坐标 -> 极坐标
        r, theta_rel = world_to_polar(agent_pos, agent_theta, subgoal)
        
        # 获取该方向的深度
        depth = self._get_depth_at_angle(state, theta_rel)
        r_max_depth = max(depth - self.subgoal_safety_margin, self.subgoal_r_min)
        
        # 边界约束
        r_max_boundary = self._compute_boundary_r_max(agent_pos, agent_theta + theta_rel)
        
        # 应用约束
        r_constrained = np.clip(
            min(r, r_max_depth, r_max_boundary), 
            self.subgoal_r_min, 
            self.subgoal_r_max
        )
        
        # 极坐标 -> 世界坐标
        theta_world = agent_theta + theta_rel
        dx = r_constrained * np.cos(theta_world)
        dy = r_constrained * np.sin(theta_world)
        
        return agent_pos + np.array([dx, dy])
    
    def _get_depth_at_angle(self, state: np.ndarray, theta_rel: float) -> float:
        """获取指定角度的深度读数 (插值)"""
        depth_readings = state[self.base_state_dim:]
        ray_angles = np.linspace(-self.depth_fov/2, self.depth_fov/2, self.depth_rays)
        
        theta_rel = np.arctan2(np.sin(theta_rel), np.cos(theta_rel))
        
        if theta_rel < ray_angles[0] or theta_rel > ray_angles[-1]:
            return self.depth_max_range
        
        idx = np.searchsorted(ray_angles, theta_rel)
        if idx == 0:
            return depth_readings[0]
        if idx >= self.depth_rays:
            return depth_readings[-1]
        
        t = (theta_rel - ray_angles[idx-1]) / (ray_angles[idx] - ray_angles[idx-1])
        return (1 - t) * depth_readings[idx-1] + t * depth_readings[idx]
    
    def _compute_boundary_r_max(
        self, 
        agent_pos: np.ndarray, 
        theta_world: float
    ) -> float:
        """计算边界约束的最大距离"""
        world_size = self.config.world_size
        margin = self.boundary_margin
        cos_t, sin_t = np.cos(theta_world), np.sin(theta_world)
        r_max = float('inf')
        
        if cos_t > 1e-6:
            r_max = min(r_max, (world_size - margin - agent_pos[0]) / cos_t)
        elif cos_t < -1e-6:
            r_max = min(r_max, (margin - agent_pos[0]) / cos_t)
        
        if sin_t > 1e-6:
            r_max = min(r_max, (world_size - margin - agent_pos[1]) / sin_t)
        elif sin_t < -1e-6:
            r_max = min(r_max, (margin - agent_pos[1]) / sin_t)
        
        return max(r_max, self.subgoal_r_min)
    
    # ==================== 核心方法 ====================
    
    def check_goal(
        self, 
        state: np.ndarray, 
        goal: np.ndarray, 
        threshold: np.ndarray
    ) -> bool:
        """检查是否达成目标"""
        state_goal = self.extract_goal_state(state)
        return np.all(np.abs(state_goal - goal) <= threshold)
    
    def set_obstacles(self, obstacles: List[Tuple[float, float, float]]) -> None:
        """设置障碍物 (特权学习)"""
        self.privileged_obstacles = obstacles
    
    def run_HAC(
        self,
        env,
        level: int,
        state: np.ndarray,
        goal: np.ndarray,
        is_subgoal_test: bool
    ) -> Tuple[np.ndarray, bool]:
        """
        运行分层策略
        
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
        self.goals_world[level] = goal[:2] if len(goal) >= 2 else goal
        
        for _ in range(self.H):
            # 高层 (Level > 0): SAC
            if level > 0:
                next_state, done, action = self._run_high_level(
                    env, level, state, goal, is_subgoal_test, goal_transitions
                )
            # 底层 (Level 0): MPC
            else:
                next_state, done, action = self._run_low_level(env, state, goal)
            
            # 存储转换 (仅高层)
            if level > 0:
                self._store_transition(
                    level, state, action, next_state, goal, done, goal_transitions
                )
            
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
        
        # 选择子目标
        subgoal = policy.select_action(state, goal, deterministic=is_subgoal_test)
        
        # 边界裁剪
        subgoal = np.clip(
            subgoal,
            [self.boundary_margin, self.boundary_margin],
            [self.config.world_size - self.boundary_margin] * 2
        )
        
        # Level 1: 深度约束
        if level == 1 and self.level1_use_polar:
            subgoal = self.apply_depth_constraint(state, subgoal)
        
        # Level 1: MPC 可达性预测
        if level == 1:
            is_reachable, _, pred_final = self.mpc.predict_reachability(
                state, subgoal, self.mpc_reachability_threshold
            )
            
            if not is_reachable:
                # 不可达，惩罚并使用预测位置
                self.buffers[level].add((
                    state, subgoal, -self.H, state, goal, 0.0, 0.0
                ))
                next_state, done = self.run_HAC(env, 0, state, pred_final, False)
            else:
                next_state, done = self.run_HAC(env, 0, state, subgoal, False)
                
                if not self.check_goal(next_state, subgoal, self.threshold):
                    self.buffers[level].add((
                        state, subgoal, -self.H, next_state, goal, 0.0, float(done)
                    ))
            
            action = self.extract_goal_state(next_state)
        
        # Level 2+: 传统子目标测试
        else:
            is_next_test = is_subgoal_test or np.random.random() < self.lamda
            next_state, done = self.run_HAC(env, level - 1, state, subgoal, is_next_test)
            
            if is_next_test and not self.check_goal(next_state, subgoal, self.threshold):
                self.buffers[level].add((
                    state, subgoal, -self.H, next_state, goal, 0.0, float(done)
                ))
            
            action = self.extract_goal_state(next_state)
        
        return next_state, done, action
    
    def _run_low_level(
        self,
        env,
        state: np.ndarray,
        goal: np.ndarray
    ) -> Tuple[np.ndarray, bool, np.ndarray]:
        """运行底层 MPC"""
        action = self.mpc.select_action(state, goal)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        if self.render and hasattr(env.unwrapped, 'render_subgoals'):
            env.unwrapped.render_subgoals(self.goals_world)
        
        self.reward += reward
        self.timestep += 1
        
        return next_state, done, action
    
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
        hindsight_goal = self.extract_goal_state(final_state)
        goal_transitions[-1][2] = 0.0
        goal_transitions[-1][5] = 0.0
        
        for transition in goal_transitions:
            transition[4] = hindsight_goal
            self.buffers[level].add(tuple(transition))
    
    # ==================== 训练方法 ====================
    
    def update(self, n_iter: int, batch_size: int) -> None:
        """更新高层策略"""
        for i in range(1, self.k_level):
            if self.buffers[i] is not None and len(self.buffers[i]) > batch_size:
                self.policies[i].update(self.buffers[i], n_iter, batch_size)
    
    def enable_level1_encoder_finetune(
        self, 
        finetune_lr: Optional[float] = None
    ) -> None:
        """启用 Level 1 编码器微调"""
        if finetune_lr is None:
            finetune_lr = self.encoder_finetune_lr
        
        if finetune_lr is None:
            print("  Level 1 encoder: frozen")
            self.policies[1].freeze_encoder()
            return
        
        self.policies[1].setup_encoder_finetune(finetune_lr)
        print(f"  Level 1 encoder: finetune enabled (lr={finetune_lr})")
    
    def reset_episode(self) -> None:
        """重置 episode"""
        self.reward = 0.0
        self.timestep = 0
        self.goals = [None] * self.k_level
        self.goals_world = [None] * self.k_level
        self.mpc.reset()
    
    def reset(self) -> None:
        """重置智能体 (兼容推理接口)"""
        self.reset_episode()
    
    def act(
        self, 
        state: np.ndarray, 
        deterministic: bool = True
    ) -> np.ndarray:
        """
        推理模式：给定状态返回动作
        
        用于测试/评估，不更新任何策略。
        
        Args:
            state: 当前状态
            deterministic: 是否使用确定性策略
            
        Returns:
            action: 底层控制动作
        """
        # 从 Level k-1 开始，逐层计算子目标
        current_goal = self.goals[self.k_level - 1]  # 最终目标
        
        if current_goal is None:
            # 如果没有设置目标，使用状态中的目标信息
            # 假设目标在状态最后几维
            current_goal = self.extract_goal_state(state)
        
        # 从高层到底层逐层选择动作
        for level in range(self.k_level - 1, 0, -1):
            policy = self.policies[level]
            subgoal = policy.select_action(state, current_goal, deterministic=deterministic)
            
            # 边界裁剪
            subgoal = np.clip(
                subgoal,
                [self.boundary_margin, self.boundary_margin],
                [self.config.world_size - self.boundary_margin] * 2
            )
            
            # Level 1: 深度约束
            if level == 1 and self.level1_use_polar:
                subgoal = self.apply_depth_constraint(state, subgoal)
            
            current_goal = subgoal
            self.goals[level - 1] = current_goal
            self.goals_world[level - 1] = current_goal[:2] if len(current_goal) >= 2 else current_goal
        
        # 底层 MPC 控制
        action = self.mpc.select_action(state, current_goal)
        return action
    
    def set_goal(self, goal: np.ndarray) -> None:
        """设置最终目标"""
        self.goals[self.k_level - 1] = goal
        self.goals_world[self.k_level - 1] = goal[:2] if len(goal) >= 2 else goal
    
    # ==================== 端到端训练 ====================
    
    def train_end_to_end_batch(
        self,
        states: np.ndarray,
        final_goal: np.ndarray,
        num_steps: int = 5,
        lr: float = 3e-4,
        verbose: bool = False
    ) -> List[Dict]:
        """
        端到端训练 (特权学习)
        
        通过可微 MPC 更新高层策略。
        """
        if self.k_level < 2:
            return [{'error': 'Need at least 2 levels'}]
        
        policy = self.policies[1]
        if not hasattr(policy, 'actor'):
            return [{'error': 'Policy has no actor'}]
        
        # 优化器
        actor_params = [
            p for n, p in policy.actor.named_parameters() 
            if 'depth_encoder' not in n
        ]
        params = actor_params
        if policy.depth_encoder is not None:
            params = params + list(policy.depth_encoder.parameters())
        optimizer = optim.Adam(params, lr=lr)
        
        batch_size = states.shape[0]
        states_t = to_tensor(states)
        goal_t = to_tensor(final_goal.reshape(1, -1)).expand(batch_size, -1)
        
        optimizer.zero_grad()
        
        # 多步展开
        all_subgoals = []
        all_trajectories = []
        current_state = states_t.clone()
        
        for _ in range(num_steps):
            subgoal, _ = policy.actor.sample(current_state, goal_t)
            
            # 边界约束
            margin = self.boundary_margin
            world_size = self.config.world_size
            subgoal = torch.clamp(subgoal, margin, world_size - margin)
            
            # MPC 展开
            action, trajectory, final_pos = self.mpc.mpc.get_action_with_gradient(
                current_state, subgoal
            )
            
            all_subgoals.append(subgoal)
            all_trajectories.append(trajectory)
            current_state = torch.cat([
                trajectory[:, -1, :5],
                states_t[:, 5:]
            ], dim=1) if states_t.shape[1] > 5 else trajectory[:, -1, :5]
        
        # 损失计算
        final_pos = all_trajectories[-1][:, -1, :2]
        goal_loss = ((final_pos - goal_t) ** 2).sum(dim=1).mean()
        
        # 避障损失
        obstacle_loss = self._compute_obstacle_loss(all_subgoals, all_trajectories)
        
        total_loss = goal_loss + 10.0 * obstacle_loss
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(policy.actor.parameters(), max_norm=1.0)
        optimizer.step()
        
        if verbose:
            dist = torch.norm(final_pos - goal_t, dim=1).mean().item()
            print(f"  E2E: dist={dist:.3f}, loss={total_loss.item():.3f}")
        
        return [{'total': torch.norm(final_pos - goal_t, dim=1).mean().item()}]
    
    def _compute_obstacle_loss(
        self,
        subgoals: List[torch.Tensor],
        trajectories: List[torch.Tensor],
        safe_dist: float = 0.3
    ) -> torch.Tensor:
        """计算避障损失"""
        if not self.privileged_obstacles:
            return torch.tensor(0.0, device=self._device)
        
        loss = torch.tensor(0.0, device=self._device)
        
        for sg in subgoals:
            for ox, oy, r in self.privileged_obstacles:
                center = torch.tensor([ox, oy], device=self._device)
                dist = torch.norm(sg - center, dim=1)
                violation = torch.clamp(r + safe_dist - dist, min=0)
                loss = loss + (violation ** 2).mean()
        
        return loss
    
    # ==================== 保存/加载 ====================
    
    def save(self, directory: str, name: str, verbose: bool = True) -> None:
        """保存模型"""
        os.makedirs(directory, exist_ok=True)
        for i in range(1, self.k_level):
            self.policies[i].save(directory, f'{name}_level_{i}')
        if verbose:
            print(f"  [SAVE] {directory}/{name}_level_*.pth")
    
    def load(self, directory: str, name: str, verbose: bool = True) -> None:
        """加载模型"""
        for i in range(1, self.k_level):
            self.policies[i].load(directory, f'{name}_level_{i}')
        if verbose:
            print(f"  [LOAD] {directory}/{name}_level_*.pth")
