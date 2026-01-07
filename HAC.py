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
                level=level
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
        import torch
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
    
    def check_any_goal_achieved(self, state: np.ndarray, i_level: int) -> bool:
        """检查是否达成任何高层目标"""
        for level in range(i_level, self.k_level):
            if self.goals[level] is not None:
                if self.check_goal(state, self.goals[level], self.threshold):
                    return True
        return False
    
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
                
                # Subgoal Testing Transition
                if is_next_subgoal_test and not self.check_goal(next_state, action, self.threshold):
                    self.replay_buffer[i_level].add((
                        state, action, -self.H, next_state, goal, 0.0, float(done)
                    ))
                
                # Hindsight Action
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
    
    def train_end_to_end(
        self,
        state: np.ndarray,
        final_goal: np.ndarray,
        num_steps: int = 3,
        verbose: bool = False
    ) -> dict:
        """
        端到端训练：通过可微 MPC 将梯度从任务损失传回高层策略
        
        梯度流: TaskLoss → MPC预测轨迹 → 子目标 → Actor → DepthEncoder
        
        关键洞察：由于动力学特性（加速度→速度→位置），需要使用 MPC 的
        多步预测位置（final_position）来有效传递梯度。
        
        Args:
            state: 初始状态
            final_goal: 最终目标
            num_steps: 高层子目标更新次数
            verbose: 是否打印详情
            
        Returns:
            loss_info: 损失信息字典
        """
        if self.k_level < 2:
            return {'error': 'Need at least 2 levels for E2E training'}
        
        # 获取高层策略 (Level 1)
        high_level_policy = self.HAC[1]  # SAC/DDPG
        
        if not hasattr(high_level_policy, 'actor'):
            return {'error': 'High level policy has no actor'}
        
        # 准备优化器 (只优化 Actor 和 Encoder)
        optimizer = torch.optim.Adam(high_level_policy.actor.parameters(), lr=1e-4)
        
        # 转换为 tensor
        state_t = torch.FloatTensor(state.reshape(1, -1)).to(device)
        goal_t = torch.FloatTensor(final_goal.reshape(1, -1)).to(device)
        
        # 获取障碍物
        obstacles = self.mpc.obstacles
        
        # 保留深度部分
        if state_t.shape[1] > 5:
            depth_part = state_t[:, 5:]
        else:
            depth_part = None
        
        optimizer.zero_grad()
        
        # ===== 展开轨迹 =====
        final_positions = []  # 每个子目标的 MPC 预测最终位置
        actions = []
        current_state = state_t.clone()
        
        for t in range(num_steps):
            # 高层策略输出子目标
            subgoal, _ = high_level_policy.actor.sample(current_state, goal_t)
            
            # MPC 追踪子目标 (使用可微版本!)
            # 返回: action, trajectory, final_position
            action, trajectory, final_pos = self.mpc.mpc.get_action_with_gradient(
                current_state, subgoal, obstacles
            )
            
            final_positions.append(final_pos)
            actions.append(action)
            
            # 更新状态到 MPC 预测轨迹的终点
            next_base = trajectory[:, -1, :5]
            if depth_part is not None:
                current_state = torch.cat([next_base, depth_part], dim=1)
            else:
                current_state = next_base
        
        # ===== 计算任务损失 =====
        # 1. 到达目标损失 (使用最终位置)
        final_pos = final_positions[-1]
        goal_loss = ((final_pos - goal_t) ** 2).sum()
        
        # 2. 进度损失 (鼓励每一步都接近目标)
        progress_loss = torch.tensor(0.0, device=device)
        for t, fp in enumerate(final_positions):
            weight = 0.3 * (t + 1) / num_steps
            progress_loss += weight * ((fp - goal_t) ** 2).sum()
        
        # 3. 避障损失
        obstacle_loss = torch.tensor(0.0, device=device)
        if obstacles is not None and len(obstacles) > 0:
            for fp in final_positions:
                for obs in obstacles:
                    ox, oy, radius = obs
                    dist = torch.norm(fp - obs[:2].unsqueeze(0), dim=1)
                    violation = torch.relu(self.mpc_safe_distance + radius - dist)
                    obstacle_loss += 5.0 * (violation ** 2).sum()
        
        # 4. 控制平滑损失
        smooth_loss = torch.tensor(0.0, device=device)
        if len(actions) > 1:
            for t in range(1, len(actions)):
                diff = actions[t] - actions[t-1]
                smooth_loss += 0.1 * (diff ** 2).sum()
        
        # 总损失
        total_loss = goal_loss + progress_loss + obstacle_loss + smooth_loss
        
        # ===== 反向传播 =====
        total_loss.backward()
        
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
            if p.grad is not None and 'depth_encoder' not in str(p)
        )
        grad_info['actor_grad_norm'] = actor_grad_norm
        
        # 更新参数
        optimizer.step()
        
        # 返回损失信息
        loss_info = {
            'total': total_loss.item(),
            'goal': goal_loss.item(),
            'progress': progress_loss.item(),
            'obstacle': obstacle_loss.item(),
            'smooth': smooth_loss.item(),
            'final_dist': torch.norm(final_pos - goal_t).item(),
            **grad_info
        }
        
        if verbose:
            print(f"  E2E Loss: {total_loss.item():.4f} "
                  f"(goal={goal_loss.item():.3f}, "
                  f"obs={obstacle_loss.item():.3f})")
            print(f"  Final distance: {loss_info['final_dist']:.3f}")
            print(f"  Gradients: encoder={grad_info.get('encoder_grad_norm', 0):.4f}, "
                  f"actor={grad_info.get('actor_grad_norm', 0):.4f}")
        
        return loss_info
    
    def train_end_to_end_batch(
        self,
        states: np.ndarray,
        final_goal: np.ndarray,
        num_steps: int = 10,
        num_epochs: int = 5,
        verbose: bool = False
    ) -> List[dict]:
        """
        批量端到端训练
        
        Args:
            states: [batch, state_dim] 初始状态批次
            final_goal: 最终目标
            num_steps: 每条轨迹的展开步数
            num_epochs: 训练轮数
            verbose: 是否打印详情
            
        Returns:
            loss_history: 损失历史
        """
        loss_history = []
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            for state in states:
                loss_info = self.train_end_to_end(
                    state=state,
                    final_goal=final_goal,
                    num_steps=num_steps,
                    verbose=False
                )
                epoch_losses.append(loss_info['total'])
            
            avg_loss = np.mean(epoch_losses)
            loss_history.append({'epoch': epoch, 'avg_loss': avg_loss})
            
            if verbose:
                print(f"E2E Epoch {epoch}: avg_loss = {avg_loss:.4f}")
        
        return loss_history
