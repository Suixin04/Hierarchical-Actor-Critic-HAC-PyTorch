"""
Hierarchical Actor-Critic (HAC) 算法实现

支持:
- 多层级策略
- 目标可以是状态的子集 (通过 goal_indices 指定)
- 基于配置的参数管理
- DDPG/SAC 底层算法切换
- 共享深度编码器 (Shared Depth Encoder)
"""
import torch
import torch.optim as optim
import numpy as np
from typing import Optional, List, Tuple, Literal
from DDPG import DDPG
from SAC import SAC
from utils import ReplayBuffer
from configs.base_config import BaseConfig

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class HAC:
    """分层Actor-Critic算法 - 支持 DDPG/SAC 切换"""
    
    def __init__(
        self, 
        config: BaseConfig, 
        render: bool = False,
        algorithm: Literal['ddpg', 'sac'] = 'ddpg'
    ):
        """
        初始化HAC智能体
        
        Args:
            config: 环境配置对象
            render: 是否渲染
            algorithm: 底层算法 ('ddpg' 或 'sac')
        """
        self.config = config
        self.render = render
        self.algorithm = algorithm.lower()
        
        # 从配置中提取参数
        self.k_level = config.k_level
        self.H = config.H
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
        
        # 探索噪声
        self.exploration_action_noise = config.exploration_action_noise
        self.exploration_state_noise = config.exploration_state_noise
        
        # 深度编码器相关参数 (每层独立编码器)
        self.use_depth_encoder = getattr(config, 'use_depth_encoder', False)
        self.base_state_dim = getattr(config, 'base_state_dim', self.state_dim)
        self.depth_dim = getattr(config, 'depth_dim', 0)
        self.embedding_dim = getattr(config, 'embedding_dim', 8)
        self.depth_max_range = getattr(config, 'depth_max_range', 5.0)
        
        if self.use_depth_encoder and self.algorithm == 'sac':
            print(f"  Independent Depth Encoders: {self.depth_dim}D → {self.embedding_dim}D (per level)")
        
        # 构建层级策略网络
        self._build_hierarchy(config.lr)
        
        # 日志参数
        self.goals = [None] * self.k_level
        self.reward = 0
        self.timestep = 0
        
        print(f"HAC initialized with {self.algorithm.upper()} as base algorithm")
    
    def _create_policy(
        self, 
        state_dim: int, 
        action_dim: int, 
        goal_dim: int,
        action_bounds: torch.Tensor,
        action_offset: torch.Tensor,
        lr: float,
        level: int
    ):
        """创建策略网络 (根据algorithm选择DDPG或SAC)"""
        if self.algorithm == 'sac':
            return SAC(
                state_dim=state_dim,
                action_dim=action_dim,
                goal_dim=goal_dim,
                action_bounds=action_bounds,
                action_offset=action_offset,
                lr=lr,
                H=self.H,
                auto_entropy=True,
                # 每层独立编码器参数
                use_depth_encoder=self.use_depth_encoder,
                base_state_dim=self.base_state_dim,
                depth_dim=self.depth_dim,
                embedding_dim=self.embedding_dim,
                depth_max_range=self.depth_max_range,
                level=level
            )
        else:  # ddpg
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
        """构建分层策略网络（每层有独立的深度编码器）"""
        self.HAC = []
        self.replay_buffer = []
        
        # 最底层: 输入(状态+目标), 输出原始动作
        self.HAC.append(self._create_policy(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            goal_dim=self.goal_dim,
            action_bounds=self.action_bounds,
            action_offset=self.action_offset,
            lr=lr,
            level=0
        ))
        self.replay_buffer.append(ReplayBuffer())
        
        # 高层: 输入(状态+目标), 输出子目标 (目标空间维度)
        for i in range(self.k_level - 1):
            self.HAC.append(self._create_policy(
                state_dim=self.state_dim,
                action_dim=self.goal_dim,  # 输出子目标
                goal_dim=self.goal_dim,
                action_bounds=self.subgoal_bounds,
                action_offset=self.subgoal_offset,
                lr=lr,
                level=i + 1
            ))
            self.replay_buffer.append(ReplayBuffer())
    
    def extract_goal_state(self, state: np.ndarray) -> np.ndarray:
        """从完整状态中提取目标相关的部分"""
        if self.goal_indices is not None:
            return state[self.goal_indices]
        return state
    
    def check_goal(self, state: np.ndarray, goal: np.ndarray, threshold: np.ndarray) -> bool:
        """检查是否达成目标"""
        # 提取状态中与目标对应的部分
        state_goal = self.extract_goal_state(state)
        
        for i in range(len(goal)):
            if abs(state_goal[i] - goal[i]) > threshold[i]:
                return False
        return True
    
    def check_any_goal_achieved(self, state: np.ndarray, i_level: int) -> bool:
        """
        检查是否达成任何高层目标 (论文: gn, n >= i-1 achieved)
        
        Args:
            state: 当前状态
            i_level: 当前层级
        Returns:
            True if any goal at level >= i_level is achieved
        """
        for level in range(i_level, self.k_level):
            if self.goals[level] is not None:
                if self.check_goal(state, self.goals[level], self.threshold):
                    return True
        return False
    
    def run_HAC(
        self, 
        env, 
        i_level: int, 
        state: np.ndarray, 
        goal: np.ndarray, 
        is_subgoal_test: bool
    ) -> Tuple[np.ndarray, bool]:
        """
        运行分层策略
        
        Args:
            env: 环境
            i_level: 当前层级
            state: 当前状态
            goal: 目标 (目标空间维度)
            is_subgoal_test: 是否为子目标测试 (所有下层都不能加噪声)
            
        Returns:
            next_state: 最终状态
            done: 是否结束
        """
        next_state = None
        done = False
        goal_transitions = []  # 用于 Hindsight Goal Transitions
        
        # 记录当前层级的目标
        self.goals[i_level] = goal
        
        # H 次尝试 (论文: for H attempts or until gn achieved)
        for attempt in range(self.H):
            # 选择动作
            # SAC: 使用随机策略探索，subgoal test时用确定性策略
            # DDPG: 策略输出 + 手动噪声
            if self.algorithm == 'sac':
                action = self.HAC[i_level].select_action(state, goal, deterministic=is_subgoal_test)
            else:
                action = self.HAC[i_level].select_action(state, goal)
            
            # ================ 高层策略 (i > 0) ================
            if i_level > 0:
                # DDPG需要手动探索噪声，SAC自带探索
                if self.algorithm == 'ddpg' and not is_subgoal_test:
                    if np.random.random_sample() > 0.2:
                        action = action + np.random.normal(0, self.exploration_state_noise)
                        action = action.clip(self.subgoal_clip_low, self.subgoal_clip_high)
                    else:
                        action = np.random.uniform(self.subgoal_clip_low, self.subgoal_clip_high)
                
                # 决定是否测试这个子目标
                is_next_subgoal_test = is_subgoal_test  # 继承父层的测试状态
                if np.random.random_sample() < self.lamda:
                    is_next_subgoal_test = True
                
                # 将子目标传递给下一层
                next_state, done = self.run_HAC(env, i_level - 1, state, action, is_next_subgoal_test)
                
                # ============ Subgoal Testing Transition ============
                # 论文 Section 4.5 & Algorithm 1:
                # - 只有在 subgoal 被测试 (is_next_subgoal_test=True) 且 miss 时才惩罚
                # - 原始仓库: check_goal(action, next_state, threshold)
                #   即检查提出的子目标 action 是否被 next_state 达成
                # - reward = penalty = -H, gamma = 0
                if is_next_subgoal_test and not self.check_goal(next_state, action, self.threshold):
                    self.replay_buffer[i_level].add((
                        state, action, -self.H, next_state, goal, 0.0, float(done)
                    ))
                
                # ============ Hindsight Action ============
                # 论文 Algorithm 1 & 原始仓库: 
                # "for hindsight action transition: action = next_state"
                # 关键: 无条件将 action 替换为实际到达的状态 (用于后续的 hindsight action transition)
                # 原始仓库直接用 next_state，当 goal_indices=None 时一样
                action = next_state if self.goal_indices is None else self.extract_goal_state(next_state)
                
            # ================ 底层策略 (i = 0) ================
            else:
                # DDPG需要手动探索噪声，SAC自带探索
                if self.algorithm == 'ddpg' and not is_subgoal_test:
                    if np.random.random_sample() > 0.2:
                        action = action + np.random.normal(0, self.exploration_action_noise)
                        action = action.clip(self.action_clip_low, self.action_clip_high)
                    else:
                        action = np.random.uniform(self.action_clip_low, self.action_clip_high)
                
                # 执行原始动作
                next_state, rew, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # 渲染
                if self.render:
                    # 优先使用通用的 render_subgoals 方法
                    if hasattr(env.unwrapped, 'render_subgoals'):
                        env.unwrapped.render_subgoals(self.goals)
                    # 兼容旧接口
                    elif hasattr(env.unwrapped, 'render_goal') and self.k_level == 2:
                        env.unwrapped.render_goal(self.goals[0], self.goals[1])
                    elif hasattr(env.unwrapped, 'render_goal_2') and self.k_level == 3:
                        env.unwrapped.render_goal_2(self.goals[0], self.goals[1], self.goals[2])
                    else:
                        # 默认渲染
                        env.render()
                
                # 记录奖励
                self.reward += rew
                self.timestep += 1
            
            # ================ 创建 Replay Transitions ================
            
            # 检查当前层目标是否达成
            goal_achieved = self.check_goal(next_state, goal, self.threshold)
            
            # 确定存储的动作
            # 论文 Algorithm 1: 
            # - 底层: 存储原始动作
            # - 高层: 存储 hindsight action (上面已经处理过，action 已经是正确的值)
            store_action = action
            
            # ============ Hindsight Action Transition ============
            # 论文 Section 4.3:
            # - [state, action, reward, next_state, goal, gamma]
            # - reward = 0 if goal achieved, -1 otherwise
            # - gamma = 0 if goal achieved, γ otherwise
            if goal_achieved:
                self.replay_buffer[i_level].add((
                    state, store_action, 0.0, next_state, goal, 0.0, float(done)
                ))
            else:
                self.replay_buffer[i_level].add((
                    state, store_action, -1.0, next_state, goal, self.gamma, float(done)
                ))
            
            # 保存用于 Hindsight Goal Transition
            # 论文 Section 4.4: 复制 transition，goal 和 reward 待定
            goal_transitions.append([
                state, store_action, -1.0, next_state, None, self.gamma, float(done)
            ])
            
            state = next_state
            
            if done or goal_achieved:
                break
        
        # ================ Hindsight Goal Transitions ================
        # 论文 Section 4.4:
        # - 使用最终到达的状态作为 hindsight goal
        # - 最后一个 transition 达成了这个 goal: reward=0, gamma=0
        # - 其他 transitions: reward=-1, gamma=γ
        
        # 原始仓库: transition[4] = next_state (直接用完整状态)
        # 当 goal_indices=None 时，hindsight_goal = next_state
        hindsight_goal = next_state if self.goal_indices is None else self.extract_goal_state(next_state)
        
        # 更新最后一个 transition (达成 hindsight goal)
        goal_transitions[-1][2] = 0.0   # reward = 0
        goal_transitions[-1][5] = 0.0   # gamma = 0
        
        for transition in goal_transitions:
            transition[4] = hindsight_goal
            self.replay_buffer[i_level].add(tuple(transition))
        
        return next_state, done
    
    def update(self, n_iter: int, batch_size: int):
        """更新所有层级的策略（包括共享编码器）"""
        for i in range(self.k_level):
            self.HAC[i].update(self.replay_buffer[i], n_iter, batch_size)
        
        # 注意：共享编码器的梯度已经通过各层 SAC 的 actor/critic 更新自动传播
        # 编码器参数被包含在 actor 和 q 网络的计算图中
    
    def save(self, directory: str, name: str):
        """保存模型（每层的编码器会随SAC一起保存）"""
        import os
        os.makedirs(directory, exist_ok=True)
        for i in range(self.k_level):
            self.HAC[i].save(directory, f'{name}_level_{i}')
    
    def load(self, directory: str, name: str):
        """加载模型"""
        for i in range(self.k_level):
            self.HAC[i].load(directory, f'{name}_level_{i}')
    
    def reset_episode(self):
        """重置episode计数"""
        self.reward = 0
        self.timestep = 0
        self.goals = [None] * self.k_level
