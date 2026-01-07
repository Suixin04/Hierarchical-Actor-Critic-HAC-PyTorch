"""
Hierarchical Actor-Critic (HAC) + MPC 底层控制

架构:
- 高层 (Level 1+): SAC/DDPG 策略网络，输出子目标
- 底层 (Level 0): MPC 控制器，追踪子目标

特点:
- 底层使用已知动力学模型，无需学习基础控制
- 高层专注于学习高级规划策略
- 支持端到端梯度回传 (第二阶段)
"""
import torch
import torch.optim as optim
import numpy as np
from typing import Optional, List, Tuple, Literal
from SAC import SAC
from DDPG import DDPG
from MPC import MPCWrapper, DifferentiableMPC
from utils import ReplayBuffer
from configs.base_config import BaseConfig

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class HAC:
    """
    分层 Actor-Critic 算法 (底层使用 MPC)
    
    高层: SAC/DDPG 策略 (学习输出子目标)

    底层: MPC 控制器 (追踪子目标)
    """
    
    def __init__(
        self,
        config: BaseConfig,
        render: bool = False,
        algorithm: Literal['ddpg', 'sac'] = 'sac',
    ):
        """
        初始化 HAC 智能体
        
        Args:
            config: 环境配置对象
            render: 是否渲染
            algorithm: 高层算法 ('ddpg' 或 'sac')
        """
        self.config = config
        self.render = render
        self.algorithm = algorithm.lower()
        
        # 从配置中提取参数
        self.k_level = config.k_level
        self.H = config.H
        
        # MPC 预测步长 (从配置读取或默认等于 H)
        self.mpc_horizon = getattr(config, 'mpc_horizon', self.H)
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.goal_dim = config.effective_goal_dim
        self.goal_indices = config.goal_indices
        
        # 获取子目标空间参数
        self.subgoal_bounds = torch.FloatTensor(
            config.get_subgoal_bounds().reshape(1, -1)
        ).to(device)
        self.subgoal_offset = torch.FloatTensor(
            config.get_subgoal_offset().reshape(1, -1)
        ).to(device)
        self.subgoal_clip_low = config.get_subgoal_clip_low()
        self.subgoal_clip_high = config.get_subgoal_clip_high()
        
        # 动作空间参数
        self.action_bounds = torch.FloatTensor(
            config.action_bounds.reshape(1, -1)
        ).to(device)
        self.action_offset = torch.FloatTensor(
            config.action_offset.reshape(1, -1)
        ).to(device)
        self.action_clip_low = config.action_clip_low
        self.action_clip_high = config.action_clip_high
        
        # 算法参数
        self.lamda = config.lamda
        self.gamma = config.gamma
        self.threshold = config.goal_threshold
        
        # 探索噪声 (仅高层使用)
        self.exploration_state_noise = config.exploration_state_noise
        
        # MPC 相关参数 (从配置读取)
        self.dt = getattr(config, 'dt', 0.1)
        self.max_v = getattr(config, 'max_v', 2.0)
        self.max_omega = getattr(config, 'max_omega', 2.0)
        self.max_a_v = getattr(config, 'max_a_v', 1.0)
        self.max_a_omega = getattr(config, 'max_a_omega', 2.0)
        self.mpc_iterations = getattr(config, 'mpc_iterations', 5)
        self.mpc_lr = getattr(config, 'mpc_lr', 0.5)
        self.mpc_Q = getattr(config, 'mpc_Q', [10.0, 10.0])
        self.mpc_R = getattr(config, 'mpc_R', [0.1, 0.1])
        self.mpc_Qf = getattr(config, 'mpc_Qf', [20.0, 20.0])
        self.mpc_obstacle_weight = getattr(config, 'mpc_obstacle_weight', 10.0)
        self.mpc_safe_distance = getattr(config, 'mpc_safe_distance', 0.5)
        
        # 深度编码器相关参数
        self.use_depth_encoder = getattr(config, 'use_depth_encoder', False)
        self.base_state_dim = getattr(config, 'base_state_dim', 5)
        self.depth_dim = getattr(config, 'depth_dim', 0)
        self.embedding_dim = getattr(config, 'embedding_dim', 8)
        self.depth_max_range = getattr(config, 'depth_max_range', 5.0)
        
        # SAC 专用参数
        self.hidden_dim = getattr(config, 'hidden_dim', 64)
        self.sac_alpha = getattr(config, 'sac_alpha', 0.2)
        self.sac_auto_entropy = getattr(config, 'sac_auto_entropy', True)
        self.sac_target_entropy = getattr(config, 'sac_target_entropy', None)
        self.sac_alpha_lr = getattr(config, 'sac_alpha_lr', None)
        
        # 编码器训练模式: 'e2e' = 只由E2E更新 (RL时detach)
        self.encoder_train_mode = getattr(config, 'encoder_train_mode', 'e2e')
        
        # 构建层级策略网络
        self._build_hierarchy(config.lr)
        
        # MPC 预测偏差缓冲区 (用于高层感知)
        self.prediction_errors = []
        self.max_error_history = 10
        
        # 日志参数
        self.goals = [None] * self.k_level
        self.reward = 0
        self.timestep = 0
        
        print(f"HAC initialized:")
        print(f"  High-level (Level 1+): {self.algorithm.upper()}")
        print(f"  Low-level (Level 0): MPC (horizon={self.mpc_horizon})")
        if self.use_depth_encoder:
            print(f"  Depth Encoder: {self.depth_dim}D → {self.embedding_dim}D")
    
    def _create_high_level_policy(
        self,
        state_dim: int,
        action_dim: int,
        goal_dim: int,
        action_bounds: torch.Tensor,
        action_offset: torch.Tensor,
        lr: float,
        level: int
    ):
        """创建高层策略网络 (SAC 或 DDPG)"""
        if self.algorithm == 'sac':
            return SAC(
                state_dim=state_dim,
                action_dim=action_dim,
                goal_dim=goal_dim,
                action_bounds=action_bounds,
                action_offset=action_offset,
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
                depth_max_range=self.depth_max_range,
                level=level,
            )
        else:
            return DDPG(
                state_dim=state_dim,
                action_dim=action_dim,
                goal_dim=goal_dim,
                action_bounds=action_bounds,
                action_offset=action_offset,
                lr=lr,
                H=self.H
            )
    
    def _build_hierarchy(self, lr: float):
        """构建混合分层策略"""
        self.HAC = []
        self.replay_buffer = []
        
        # ============ 底层 (Level 0): MPC 控制器 ============
        # 使用配置中的 MPC 参数
        Q = torch.FloatTensor(self.mpc_Q)
        R = torch.FloatTensor(self.mpc_R)
        Qf = torch.FloatTensor(self.mpc_Qf)
        
        self.mpc = MPCWrapper(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            goal_dim=self.goal_dim,
            action_bounds=self.action_bounds,
            action_offset=self.action_offset,
            horizon=self.mpc_horizon,
            dt=self.dt,
            max_v=self.max_v,
            max_omega=self.max_omega,
            max_a_v=self.max_a_v,
            max_a_omega=self.max_a_omega,
            num_iterations=self.mpc_iterations,
            lr=self.mpc_lr,
            Q=Q,
            R=R,
            Qf=Qf,
            obstacle_weight=self.mpc_obstacle_weight,
            safe_distance=self.mpc_safe_distance,
        )
        
        # MPC 不需要 replay buffer，但保持接口一致
        self.HAC.append(self.mpc)
        self.replay_buffer.append(None)  # 底层不用 buffer
        
        # ============ 高层 (Level 1+): SAC/DDPG ============
        for i in range(self.k_level - 1):
            self.HAC.append(self._create_high_level_policy(
                state_dim=self.state_dim,
                action_dim=self.goal_dim,  # 输出子目标
                goal_dim=self.goal_dim,
                action_bounds=self.subgoal_bounds,
                action_offset=self.subgoal_offset,
                lr=lr,
                level=i + 1
            ))
            self.replay_buffer.append(ReplayBuffer())
    
    def set_obstacles(self, obstacles: List[Tuple[float, float, float]]):
        """
        设置障碍物信息给 MPC
        
        Args:
            obstacles: List of (x, y, radius)
        """
        self.mpc.set_obstacles(obstacles)
    
    def extract_goal_state(self, state: np.ndarray) -> np.ndarray:
        """从完整状态中提取目标相关的部分"""
        if self.goal_indices is not None:
            return state[self.goal_indices]
        return state
    
    def check_goal(self, state: np.ndarray, goal: np.ndarray, threshold: np.ndarray) -> bool:
        """检查是否达成目标"""
        state_goal = self.extract_goal_state(state)
        for i in range(len(goal)):
            if abs(state_goal[i] - goal[i]) > threshold[i]:
                return False
        return True
    
    def get_prediction_error_feature(self) -> np.ndarray:
        """
        获取 MPC 预测偏差特征 (供高层使用)
        
        Returns:
            error_feature: [mean_error, max_error, error_std]
        """
        if len(self.prediction_errors) == 0:
            return np.zeros(3)
        
        errors = np.array(self.prediction_errors)
        return np.array([
            errors.mean(),
            errors.max(),
            errors.std()
        ])
    
    def run_HAC(
        self,
        env,
        i_level: int,
        state: np.ndarray,
        goal: np.ndarray,
        is_subgoal_test: bool
    ) -> Tuple[np.ndarray, bool]:
        """
        运行混合分层策略
        
        Args:
            env: 环境
            i_level: 当前层级
            state: 当前状态
            goal: 目标 (目标空间维度)
            is_subgoal_test: 是否为子目标测试
            
        Returns:
            next_state: 最终状态
            done: 是否结束
        """
        next_state = None
        done = False
        goal_transitions = []
        
        self.goals[i_level] = goal
        
        for attempt in range(self.H):
            # ================ 高层策略 (i > 0): SAC/DDPG ================
            if i_level > 0:
                # 选择子目标
                if self.algorithm == 'sac':
                    action = self.HAC[i_level].select_action(
                        state, goal, deterministic=is_subgoal_test
                    )
                else:
                    action = self.HAC[i_level].select_action(state, goal)
                    # DDPG 需要手动探索噪声
                    if not is_subgoal_test:
                        if np.random.random_sample() > 0.2:
                            action = action + np.random.normal(0, self.exploration_state_noise)
                            action = action.clip(self.subgoal_clip_low, self.subgoal_clip_high)
                        else:
                            action = np.random.uniform(self.subgoal_clip_low, self.subgoal_clip_high)
                
                # 决定是否测试子目标
                is_next_subgoal_test = is_subgoal_test
                if np.random.random_sample() < self.lamda:
                    is_next_subgoal_test = True
                
                # 递归调用下一层
                next_state, done = self.run_HAC(env, i_level - 1, state, action, is_next_subgoal_test)
                
                # Subgoal Testing Transition (HAC 论文核心机制)
                # 如果子目标被测试且未达成，惩罚该子目标
                if is_next_subgoal_test and not self.check_goal(next_state, action, self.threshold):
                    self.replay_buffer[i_level].add((
                        state, action, -self.H, next_state, goal, 0.0, float(done)
                    ))
                
                # Hindsight Action（用实际达到的状态替换）
                action = self.extract_goal_state(next_state)
                
            # ================ 底层策略 (i = 0): MPC ================
            else:
                # MPC 需要障碍物信息
                if hasattr(env.unwrapped, 'obstacles'):
                    self.set_obstacles(env.unwrapped.obstacles)
                
                # 使用 MPC 计算动作
                action = self.mpc.select_action(state, goal)
                
                # 获取 MPC 预测的下一状态 (用于计算偏差)
                state_t = torch.FloatTensor(state.reshape(1, -1)).to(device)
                goal_t = torch.FloatTensor(goal.reshape(1, -1)).to(device)
                _, _, info = self.mpc.mpc(state_t, goal_t, self.mpc.obstacles, return_trajectory=True)
                predicted_next = info['predicted_trajectory'][0, 1, :5].cpu().numpy()
                
                # 执行动作
                next_state, rew, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # 计算预测偏差
                actual_next = next_state[:5]
                prediction_error = np.linalg.norm(actual_next[:2] - predicted_next[:2])
                self.prediction_errors.append(prediction_error)
                if len(self.prediction_errors) > self.max_error_history:
                    self.prediction_errors.pop(0)
                
                # 渲染
                if self.render:
                    if hasattr(env.unwrapped, 'render_subgoals'):
                        env.unwrapped.render_subgoals(self.goals)
                    else:
                        env.render()
                
                self.reward += rew
                self.timestep += 1
            
            # ================ 创建 Replay Transitions (仅高层) ================
            if i_level > 0:
                goal_achieved = self.check_goal(next_state, goal, self.threshold)
                store_action = action
                
                if goal_achieved:
                    self.replay_buffer[i_level].add((
                        state, store_action, 0.0, next_state, goal, 0.0, float(done)
                    ))
                else:
                    self.replay_buffer[i_level].add((
                        state, store_action, -1.0, next_state, goal, self.gamma, float(done)
                    ))
                
                goal_transitions.append([
                    state, store_action, -1.0, next_state, None, self.gamma, float(done)
                ])
            
            state = next_state
            
            # 检查是否达成目标
            goal_achieved = self.check_goal(next_state, goal, self.threshold)
            if done or goal_achieved:
                break
        
        # ================ Hindsight Goal Transitions (仅高层) ================
        if i_level > 0 and len(goal_transitions) > 0:
            hindsight_goal = self.extract_goal_state(next_state)
            goal_transitions[-1][2] = 0.0
            goal_transitions[-1][5] = 0.0
            
            for transition in goal_transitions:
                transition[4] = hindsight_goal
                self.replay_buffer[i_level].add(tuple(transition))
        
        return next_state, done
    
    def update(self, n_iter: int, batch_size: int):
        """更新高层策略 (底层 MPC 不需要更新)"""
        for i in range(1, self.k_level):  # 从 level 1 开始
            if self.replay_buffer[i] is not None and len(self.replay_buffer[i]) > batch_size:
                self.HAC[i].update(self.replay_buffer[i], n_iter, batch_size)
    
    def freeze_all_encoders(self):
        """
        冻结所有层的编码器 (Phase 2 开始时调用)
        
        冻结后编码器参数不再更新，但前向传播正常工作
        """
        for i in range(1, self.k_level):
            policy = self.HAC[i]
            if hasattr(policy, 'depth_encoder') and policy.depth_encoder is not None:
                # 冻结参数
                for param in policy.depth_encoder.parameters():
                    param.requires_grad = False
                # 设置为 eval 模式 (关闭 dropout/batchnorm 等)
                policy.depth_encoder.eval()
                print(f"  Level {i} encoder frozen ({sum(p.numel() for p in policy.depth_encoder.parameters())} params)")
    
    def unfreeze_all_encoders(self):
        """解冻所有层的编码器"""
        for i in range(1, self.k_level):
            policy = self.HAC[i]
            if hasattr(policy, 'depth_encoder') and policy.depth_encoder is not None:
                for param in policy.depth_encoder.parameters():
                    param.requires_grad = True
                policy.depth_encoder.train()
                print(f"  Level {i} encoder unfrozen")
    
    def save(self, directory: str, name: str):
        """保存高层模型"""
        import os
        os.makedirs(directory, exist_ok=True)
        for i in range(1, self.k_level):  # 从 level 1 开始
            self.HAC[i].save(directory, f'{name}_level_{i}')
    
    def load(self, directory: str, name: str):
        """加载高层模型"""
        for i in range(1, self.k_level):
            self.HAC[i].load(directory, f'{name}_level_{i}')
    
    def reset_episode(self):
        """重置 episode 计数"""
        self.reward = 0
        self.timestep = 0
        self.goals = [None] * self.k_level
        self.prediction_errors = []
        # 重置 MPC 暖启动
        self.mpc.mpc.prev_actions = None
    
    # ============== 端到端训练方法 ==============
    
    def train_end_to_end_batch(
        self,
        states: np.ndarray,
        final_goal: np.ndarray,
        num_steps: int = 5,
        lr: float = 3e-4,
        verbose: bool = False
    ) -> List[dict]:
        """
        真正的批量端到端训练 (向量化处理整个 batch)
        
        Args:
            states: [batch, state_dim] 初始状态批次
            final_goal: 最终目标
            num_steps: 每条轨迹的展开步数
            lr: 学习率
            verbose: 是否打印详情
            
        Returns:
            loss_history: 损失历史
        """
        if self.k_level < 2:
            return [{'error': 'Need at least 2 levels for E2E training'}]
        
        # 获取高层策略 (Level 1)
        high_level_policy = self.HAC[1]
        
        if not hasattr(high_level_policy, 'actor'):
            return [{'error': 'High level policy has no actor'}]
        
        # 优化器: actor.parameters() 已包含 encoder (因为 actor 内部引用了 encoder)
        optimizer = torch.optim.Adam(high_level_policy.actor.parameters(), lr=lr)
        
        # 转换为 tensor [batch, state_dim]
        batch_size = states.shape[0]
        states_t = torch.FloatTensor(states).to(device)
        goal_t = torch.FloatTensor(final_goal.reshape(1, -1)).expand(batch_size, -1).to(device)
        
        # 获取障碍物
        obstacles = self.mpc.obstacles
        
        # 保留深度部分
        if states_t.shape[1] > 5:
            depth_part = states_t[:, 5:]
        else:
            depth_part = None
        
        optimizer.zero_grad()
        
        # ===== 批量展开轨迹 =====
        final_positions = []
        all_actions = []
        current_state = states_t.clone()
        
        for t in range(num_steps):
            # 高层策略输出子目标 [batch, goal_dim]
            subgoal, _ = high_level_policy.actor.sample(current_state, goal_t)
            
            # MPC 追踪子目标 (批量处理!)
            action, trajectory, final_pos = self.mpc.mpc.get_action_with_gradient(
                current_state, subgoal, obstacles
            )
            
            final_positions.append(final_pos)
            all_actions.append(action)
            
            # 更新状态到 MPC 预测轨迹的终点
            next_base = trajectory[:, -1, :5]
            if depth_part is not None:
                current_state = torch.cat([next_base, depth_part], dim=1)
            else:
                current_state = next_base
        
        # ===== 计算批量任务损失 =====
        # 1. 到达目标损失 [batch]
        final_pos = final_positions[-1]
        goal_loss = ((final_pos - goal_t) ** 2).sum(dim=1).mean()
        
        # 2. 避障损失
        obstacle_loss = torch.tensor(0.0, device=device)
        if obstacles is not None and len(obstacles) > 0:
            for fp in final_positions:
                # fp: [batch, 2], obstacles: [num_obs, 3]
                obs_pos = obstacles[:, :2]  # [num_obs, 2]
                obs_rad = obstacles[:, 2]   # [num_obs]
                
                # 计算每个 batch 到每个障碍物的距离
                fp_expanded = fp.unsqueeze(1)  # [batch, 1, 2]
                obs_expanded = obs_pos.unsqueeze(0)  # [1, num_obs, 2]
                dist = torch.norm(fp_expanded - obs_expanded, dim=2)  # [batch, num_obs]
                
                # 违反约束
                margin = self.mpc_safe_distance + obs_rad.unsqueeze(0) - dist
                violation = torch.relu(margin) ** 2
                obstacle_loss += 5.0 * violation.sum(dim=1).mean()
        
        # 3. 控制平滑损失
        smooth_loss = torch.tensor(0.0, device=device)
        if len(all_actions) > 1:
            for t in range(1, len(all_actions)):
                diff = all_actions[t] - all_actions[t-1]
                smooth_loss += 0.1 * (diff ** 2).sum(dim=1).mean()
        
        # 总损失
        total_loss = goal_loss + obstacle_loss + smooth_loss
        
        # ===== 反向传播 =====
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(high_level_policy.actor.parameters(), max_norm=1.0)
        
        # 检查梯度
        grad_info = {}
        if high_level_policy.depth_encoder is not None:
            encoder_grad_norm = sum(
                p.grad.norm().item() 
                for p in high_level_policy.depth_encoder.parameters() 
                if p.grad is not None
            )
            grad_info['encoder_grad_norm'] = encoder_grad_norm
        
        actor_grad_norm = sum(
            p.grad.norm().item() 
            for p in high_level_policy.actor.parameters() 
            if p.grad is not None
        )
        grad_info['actor_grad_norm'] = actor_grad_norm
        
        # 更新参数
        optimizer.step()
        
        # 计算最终距离 (归一化指标，更稳定)
        final_dist = torch.norm(final_pos - goal_t, dim=1).mean().item()
        
        # 返回损失信息
        # 注意：'total' 现在返回 final_dist (更有意义的指标)
        loss_info = {
            'total': final_dist,  # 用最终距离作为主要指标，更易理解
            'loss_value': total_loss.item(),  # 原始 loss 值
            'goal': goal_loss.item(),
            'obstacle': obstacle_loss.item(),
            'smooth': smooth_loss.item(),
            'final_dist': final_dist,
            **grad_info
        }
        
        if verbose:
            print(f"  E2E: dist={final_dist:.3f}, loss={total_loss.item():.3f} "
                  f"(goal={goal_loss.item():.3f}, obs={obstacle_loss.item():.3f})")
            if 'encoder_grad_norm' in grad_info:
                print(f"  Gradients: encoder={grad_info['encoder_grad_norm']:.4f}, "
                      f"actor={grad_info['actor_grad_norm']:.4f}")
        
        return [loss_info]
