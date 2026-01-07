"""
Hierarchical Actor-Critic (HAC) + MPC 底层控制

架构 (已确定):
- 高层 (Level 1+): SAC 策略网络，输出子目标
- 底层 (Level 0): MPC 控制器，追踪子目标

特点:
- 底层使用已知动力学模型，无需学习基础控制
- 高层专注于学习高级规划策略
- Level 1 使用 MPC 预测可达性 (替代传统 Subgoal Testing)
- 支持端到端梯度回传 (特权学习)
"""
import torch
import torch.optim as optim
import numpy as np
from typing import Optional, List, Tuple
from SAC import SAC
from MPC import MPCWrapper, DifferentiableMPC
from utils import ReplayBuffer
from configs.base_config import BaseConfig

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class HAC:
    """
    分层 Actor-Critic 算法 (底层使用 MPC)
    
    高层: SAC 策略 (学习输出子目标)
    底层: MPC 控制器 (追踪子目标)
    """
    
    def __init__(
        self,
        config: BaseConfig,
        render: bool = False,
    ):
        """
        初始化 HAC 智能体
        
        Args:
            config: 环境配置对象
            render: 是否渲染
        """
        self.config = config
        self.render = render
        
        # 从配置中提取参数
        self.k_level = config.k_level
        self.H = config.H
        self.mpc_horizon = self.H
        
        # 状态/动作维度
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.goal_dim = config.effective_goal_dim
        self.goal_indices = config.goal_indices
        
        # 子目标空间参数
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
        
        # MPC 参数
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
        
        # MPC 可达性预测参数
        self.mpc_reachability_threshold = getattr(config, 'mpc_reachability_threshold', 0.8)
        
        # 深度编码器参数
        self.use_depth_encoder = getattr(config, 'use_depth_encoder', False)
        self.base_state_dim = getattr(config, 'base_state_dim', 5)
        self.depth_dim = getattr(config, 'depth_dim', 0)
        self.embedding_dim = getattr(config, 'embedding_dim', 8)
        self.depth_max_range = getattr(config, 'depth_max_range', 5.0)
        
        # SAC 参数
        self.hidden_dim = getattr(config, 'hidden_dim', 64)
        self.sac_alpha = getattr(config, 'sac_alpha', 0.2)
        self.sac_auto_entropy = getattr(config, 'sac_auto_entropy', True)
        self.sac_target_entropy = getattr(config, 'sac_target_entropy', None)
        self.sac_alpha_lr = getattr(config, 'sac_alpha_lr', None)
        
        # Encoder 微调学习率 (RL 阶段 Level 1)
        self.encoder_finetune_lr = getattr(config, 'encoder_finetune_lr', None)  # None = 不微调
        
        # 构建层级策略网络
        self._build_hierarchy(config.lr)
        
        # MPC 预测偏差缓冲区
        self.prediction_errors = []
        self.max_error_history = 10
        
        # 日志参数
        self.goals = [None] * self.k_level
        self.reward = 0
        self.timestep = 0
        
        print(f"HAC initialized:")
        print(f"  High-level (Level 1+): SAC")
        print(f"  Low-level (Level 0): MPC (horizon={self.mpc_horizon})")
        if self.use_depth_encoder:
            print(f"  Depth Encoder: {self.depth_dim}D → {self.embedding_dim}D")
    
    def _build_hierarchy(self, lr: float):
        """构建混合分层策略"""
        self.HAC = []
        self.replay_buffer = []
        
        # ============ 底层 (Level 0): MPC 控制器 ============
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
            Q=Q, R=R, Qf=Qf,
        )
        
        self.HAC.append(self.mpc)
        self.replay_buffer.append(None)
        
        # ============ 高层 (Level 1+): SAC ============
        for i in range(self.k_level - 1):
            level = i + 1
            # Level 1: finetune 模式 (encoder 预训练后用小学习率微调)
            # Level 2+: rl 模式 (encoder 由 RL 正常更新)
            encoder_mode = 'finetune' if level == 1 else 'rl'
            
            self.HAC.append(SAC(
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
                depth_max_range=self.depth_max_range,
                level=level,
                encoder_train_mode=encoder_mode,
            ))
            self.replay_buffer.append(ReplayBuffer())
    
    def set_obstacles(self, obstacles: List[Tuple[float, float, float]]):
        """设置障碍物信息 (用于特权学习)"""
        self.privileged_obstacles = obstacles
    
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
    
    def predict_subgoal_reachability(
        self,
        state: np.ndarray,
        subgoal: np.ndarray,
        reachability_threshold: float = None
    ) -> Tuple[bool, float, np.ndarray]:
        """
        使用 MPC 预测子目标是否在 H 步内可达
        
        Args:
            state: 当前状态
            subgoal: 子目标位置 [x, y]
            reachability_threshold: 可达性判定阈值
            
        Returns:
            is_reachable: 是否可达
            predicted_distance: 预测的最终距离
            predicted_final_pos: 预测的最终位置
        """
        if reachability_threshold is None:
            reachability_threshold = self.mpc_reachability_threshold
        
        state_t = torch.FloatTensor(state.reshape(1, -1)).to(device)
        subgoal_t = torch.FloatTensor(subgoal[:2].reshape(1, -1)).to(device)
        
        _, _, info = self.mpc.mpc(state_t, subgoal_t, return_trajectory=True)
        predicted_trajectory = info['predicted_trajectory']
        predicted_final_pos = predicted_trajectory[0, -1, :2].detach().cpu().numpy()
        
        predicted_distance = np.linalg.norm(predicted_final_pos - subgoal[:2])
        is_reachable = predicted_distance <= reachability_threshold
        
        return is_reachable, predicted_distance, predicted_final_pos
    
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
        
        Level 1: MPC 预测可达性 (替代 Subgoal Testing)
        Level 2+: 传统 Subgoal Testing
        """
        next_state = None
        done = False
        goal_transitions = []
        
        self.goals[i_level] = goal
        
        for attempt in range(self.H):
            # ================ 高层策略 (i > 0): SAC ================
            if i_level > 0:
                # 选择子目标 (原始输出)
                original_action = self.HAC[i_level].select_action(
                    state, goal, deterministic=is_subgoal_test
                )
                
                # Level 1: MPC 预测可达性
                if i_level == 1:
                    is_reachable, pred_dist, pred_final = self.predict_subgoal_reachability(state, original_action)
                    
                    if not is_reachable:
                        # 子目标不可达，惩罚原始 action
                        self.replay_buffer[i_level].add((
                            state, original_action, -self.H, state, goal, 0.0, 0.0
                        ))
                        # 使用 MPC 预测的最终位置执行
                        next_state, done = self.run_HAC(env, 0, state, pred_final, False)
                    else:
                        # 子目标可达，正常执行
                        next_state, done = self.run_HAC(env, 0, state, original_action, False)
                        
                        # 验证实际到达情况
                        if not self.check_goal(next_state, original_action, self.threshold):
                            # 预测可达但实际未达，惩罚原始 action
                            self.replay_buffer[i_level].add((
                                state, original_action, -self.H, next_state, goal, 0.0, float(done)
                            ))
                    
                    # Hindsight Action (用于后续的正常转移)
                    action = self.extract_goal_state(next_state)
                    
                # Level 2+: 传统 Subgoal Testing
                else:
                    is_next_subgoal_test = is_subgoal_test
                    if np.random.random_sample() < self.lamda:
                        is_next_subgoal_test = True
                    
                    next_state, done = self.run_HAC(env, i_level - 1, state, original_action, is_next_subgoal_test)
                    
                    # Subgoal Testing: 惩罚原始 action
                    if is_next_subgoal_test and not self.check_goal(next_state, original_action, self.threshold):
                        self.replay_buffer[i_level].add((
                            state, original_action, -self.H, next_state, goal, 0.0, float(done)
                        ))
                    
                    # Hindsight Action
                    action = self.extract_goal_state(next_state)
                
            # ================ 底层策略 (i = 0): MPC ================
            else:
                action = self.mpc.select_action(state, goal)
                
                state_t = torch.FloatTensor(state.reshape(1, -1)).to(device)
                goal_t = torch.FloatTensor(goal.reshape(1, -1)).to(device)
                _, _, info = self.mpc.mpc(state_t, goal_t, return_trajectory=True)
                predicted_next = info['predicted_trajectory'][0, 1, :5].cpu().numpy()
                
                next_state, rew, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                actual_next = next_state[:5]
                prediction_error = np.linalg.norm(actual_next[:2] - predicted_next[:2])
                self.prediction_errors.append(prediction_error)
                if len(self.prediction_errors) > self.max_error_history:
                    self.prediction_errors.pop(0)
                
                if self.render:
                    if hasattr(env.unwrapped, 'render_subgoals'):
                        env.unwrapped.render_subgoals(self.goals)
                    else:
                        env.render()
                
                self.reward += rew
                self.timestep += 1
            
            # ================ Replay Transitions (仅高层) ================
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
            
            goal_achieved = self.check_goal(next_state, goal, self.threshold)
            if done or goal_achieved:
                break
        
        # ================ Hindsight Goal Transitions ================
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
        for i in range(1, self.k_level):
            if self.replay_buffer[i] is not None and len(self.replay_buffer[i]) > batch_size:
                self.HAC[i].update(self.replay_buffer[i], n_iter, batch_size)
    
    def enable_level1_encoder_finetune(self, finetune_lr: float = None):
        """
        Phase 2 开始时: 为 Level 1 encoder 启用小学习率微调
        
        Level 1 初始化时 encoder_train_mode='finetune'，encoder 没有 optimizer。
        调用此函数后，创建小学习率的 encoder optimizer，开始微调。
        
        Args:
            finetune_lr: 微调学习率，None 则使用配置中的 encoder_finetune_lr
        """
        if finetune_lr is None:
            finetune_lr = self.encoder_finetune_lr
        
        if finetune_lr is None:
            print("  Level 1 encoder: frozen (no finetune_lr specified)")
            self.freeze_level1_encoder()
            return
        
        policy = self.HAC[1]
        if hasattr(policy, 'depth_encoder') and policy.depth_encoder is not None:
            # 为 encoder 创建小学习率的 optimizer
            policy.setup_encoder_finetune(finetune_lr)
            print(f"  Level 1 encoder: finetune enabled (lr={finetune_lr})")
    
    def freeze_level1_encoder(self):
        """冻结 Level 1 的编码器"""
        policy = self.HAC[1]
        if hasattr(policy, 'depth_encoder') and policy.depth_encoder is not None:
            for param in policy.depth_encoder.parameters():
                param.requires_grad = False
            policy.depth_encoder.eval()
            print(f"  Level 1 encoder frozen")
    
    def save(self, directory: str, name: str):
        """保存高层模型"""
        import os
        os.makedirs(directory, exist_ok=True)
        for i in range(1, self.k_level):
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
        self.mpc.mpc.prev_actions = None
    
    # ============== 特权学习 ==============
    
    def compute_subgoal_obstacle_loss(
        self,
        subgoals: torch.Tensor,
        safe_distance: float = 0.3
    ) -> torch.Tensor:
        """计算子目标的避障损失 (使用特权信息)"""
        if not hasattr(self, 'privileged_obstacles') or not self.privileged_obstacles:
            return torch.tensor(0.0, device=subgoals.device)
        
        total_loss = torch.tensor(0.0, device=subgoals.device)
        
        for (ox, oy, r) in self.privileged_obstacles:
            obs_center = torch.tensor([ox, oy], device=subgoals.device, dtype=subgoals.dtype)
            dist = torch.norm(subgoals - obs_center, dim=1)
            safe_boundary = r + safe_distance
            violation = torch.clamp(safe_boundary - dist, min=0)
            loss = (violation ** 2).mean()
            total_loss = total_loss + loss
        
        return total_loss
    
    def compute_trajectory_obstacle_loss(
        self,
        trajectories: List[torch.Tensor],
        safe_distance: float = 0.2
    ) -> torch.Tensor:
        """计算轨迹的避障损失 (使用特权信息)"""
        if not hasattr(self, 'privileged_obstacles') or not self.privileged_obstacles:
            return torch.tensor(0.0, device=trajectories[0].device if trajectories else 'cpu')
        
        total_loss = torch.tensor(0.0, device=trajectories[0].device)
        
        for traj in trajectories:
            positions = traj[:, :, :2]
            
            for (ox, oy, r) in self.privileged_obstacles:
                obs_center = torch.tensor([ox, oy], device=traj.device, dtype=traj.dtype)
                diff = positions - obs_center
                dist = torch.norm(diff, dim=2)
                safe_boundary = r + safe_distance
                violation = torch.clamp(safe_boundary - dist, min=0)
                loss = (violation ** 2).mean()
                total_loss = total_loss + loss
        
        return total_loss
    
    def train_end_to_end_batch(
        self,
        states: np.ndarray,
        final_goal: np.ndarray,
        num_steps: int = 5,
        lr: float = 3e-4,
        verbose: bool = False
    ) -> List[dict]:
        """端到端训练 (特权学习)"""
        if self.k_level < 2:
            return [{'error': 'Need at least 2 levels for E2E training'}]
        
        high_level_policy = self.HAC[1]
        
        if not hasattr(high_level_policy, 'actor'):
            return [{'error': 'High level policy has no actor'}]
        
        optimizer = torch.optim.Adam(high_level_policy.actor.parameters(), lr=lr)
        
        batch_size = states.shape[0]
        states_t = torch.FloatTensor(states).to(device)
        goal_t = torch.FloatTensor(final_goal.reshape(1, -1)).expand(batch_size, -1).to(device)
        
        depth_part = states_t[:, 5:] if states_t.shape[1] > 5 else None
        
        optimizer.zero_grad()
        
        final_positions = []
        all_actions = []
        all_trajectories = []
        all_subgoals = []
        current_state = states_t.clone()
        
        for t in range(num_steps):
            subgoal, _ = high_level_policy.actor.sample(current_state, goal_t)
            action, trajectory, final_pos = self.mpc.mpc.get_action_with_gradient(
                current_state, subgoal
            )
            
            final_positions.append(final_pos)
            all_actions.append(action)
            all_trajectories.append(trajectory)
            all_subgoals.append(subgoal)
            
            next_base = trajectory[:, -1, :5]
            if depth_part is not None:
                current_state = torch.cat([next_base, depth_part], dim=1)
            else:
                current_state = next_base
        
        # 损失计算
        final_pos = final_positions[-1]
        goal_loss = ((final_pos - goal_t) ** 2).sum(dim=1).mean()
        
        smooth_loss = torch.tensor(0.0, device=device)
        if len(all_actions) > 1:
            for t in range(1, len(all_actions)):
                diff = all_actions[t] - all_actions[t-1]
                smooth_loss += 0.1 * (diff ** 2).sum(dim=1).mean()
        
        subgoal_safe_dist = getattr(self.config, 'e2e_safe_distance', 0.3)
        traj_safe_dist = getattr(self.config, 'e2e_traj_safe_distance', 0.2)
        subgoal_obs_loss = self.compute_subgoal_obstacle_loss(subgoal, safe_distance=subgoal_safe_dist)
        traj_obs_loss = self.compute_trajectory_obstacle_loss(all_trajectories, safe_distance=traj_safe_dist)
        obstacle_loss = subgoal_obs_loss + 0.5 * traj_obs_loss
        
        obstacle_weight = getattr(self.config, 'e2e_obstacle_weight', 10.0)
        total_loss = goal_loss + smooth_loss + obstacle_weight * obstacle_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(high_level_policy.actor.parameters(), max_norm=1.0)
        
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
        
        optimizer.step()
        
        final_dist = torch.norm(final_pos - goal_t, dim=1).mean().item()
        
        loss_info = {
            'total': final_dist,
            'loss_value': total_loss.item(),
            'goal': goal_loss.item(),
            'smooth': smooth_loss.item(),
            'obstacle': obstacle_loss.item(),
            'final_dist': final_dist,
            **grad_info
        }
        
        if verbose:
            print(f"  E2E: dist={final_dist:.3f}, loss={total_loss.item():.3f} "
                  f"(goal={goal_loss.item():.3f}, obs={obstacle_loss.item():.3f})")
        
        return [loss_info]
