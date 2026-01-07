"""
可微模型预测控制 (Differentiable MPC) 模块

支持:
- 双轮差速机器人动力学模型 (Unicycle Model)
- 可微优化层 (通过 PyTorch 自动求导)
- 软约束避障
- 梯度回传到子目标

动力学模型:
    ẋ = v * cos(θ)
    ẏ = v * sin(θ)
    θ̇ = ω
    v̇ = a_v
    ω̇ = a_ω

Author: HAC + MPC Project
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DifferentiableDynamics(nn.Module):
    """
    可微的双轮差速机器人动力学模型
    
    State: [x, y, θ, v, ω]
    Action: [a_v, a_ω]
    
    离散化 (Euler):
        x_{t+1} = x_t + v_t * cos(θ_t) * dt
        y_{t+1} = y_t + v_t * sin(θ_t) * dt
        θ_{t+1} = θ_t + ω_t * dt
        v_{t+1} = v_t + a_v * dt
        ω_{t+1} = ω_t + a_ω * dt
    """
    
    def __init__(
        self,
        dt: float = 0.1,
        max_v: float = 2.0,
        max_omega: float = 2.0,
    ):
        super().__init__()
        self.dt = dt
        self.max_v = max_v
        self.max_omega = max_omega
    
    def _soft_clamp(self, x: torch.Tensor, min_val: float, max_val: float, scale: float = 10.0) -> torch.Tensor:
        """
        软裁剪函数 (可微)
        
        使用 tanh 实现软约束，保持梯度流动
        """
        center = (min_val + max_val) / 2
        half_range = (max_val - min_val) / 2
        # 映射到 [-1, 1] 再用 tanh 软约束
        normalized = (x - center) / (half_range + 1e-6)
        # tanh 会在边界附近软化
        soft = torch.tanh(normalized / scale) * scale
        return soft * half_range + center
    
    def forward(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor,
        soft_constraints: bool = False
    ) -> torch.Tensor:
        """
        单步动力学前向传播
        
        Args:
            state: [batch, 5] - [x, y, θ, v, ω]
            action: [batch, 2] - [a_v, a_ω]
            soft_constraints: 是否使用软约束 (可微训练时使用)
            
        Returns:
            next_state: [batch, 5]
        """
        x = state[:, 0]
        y = state[:, 1]
        theta = state[:, 2]
        v = state[:, 3]
        omega = state[:, 4]
        
        a_v = action[:, 0]
        a_omega = action[:, 1]
        
        # Euler 积分
        x_new = x + v * torch.cos(theta) * self.dt
        y_new = y + v * torch.sin(theta) * self.dt
        theta_new = theta + omega * self.dt
        v_new = v + a_v * self.dt
        omega_new = omega + a_omega * self.dt
        
        # 角度归一化到 [-π, π]
        theta_new = torch.atan2(torch.sin(theta_new), torch.cos(theta_new))
        
        # 速度约束
        if soft_constraints:
            # 软约束 (可微)
            v_new = self._soft_clamp(v_new, -self.max_v, self.max_v)
            omega_new = self._soft_clamp(omega_new, -self.max_omega, self.max_omega)
        else:
            # 硬约束 (用于推理)
            v_new = torch.clamp(v_new, -self.max_v, self.max_v)
            omega_new = torch.clamp(omega_new, -self.max_omega, self.max_omega)
        
        next_state = torch.stack([x_new, y_new, theta_new, v_new, omega_new], dim=1)
        return next_state
    
    def rollout(
        self, 
        init_state: torch.Tensor, 
        actions: torch.Tensor,
        soft_constraints: bool = False
    ) -> torch.Tensor:
        """
        多步轨迹展开
        
        Args:
            init_state: [batch, 5] 初始状态
            actions: [batch, horizon, 2] 控制序列
            soft_constraints: 是否使用软约束
            
        Returns:
            states: [batch, horizon+1, 5] 状态轨迹 (包含初始状态)
        """
        batch_size = init_state.shape[0]
        horizon = actions.shape[1]
        
        states = [init_state]
        state = init_state
        
        for t in range(horizon):
            state = self.forward(state, actions[:, t, :], soft_constraints=soft_constraints)
            states.append(state)
        
        return torch.stack(states, dim=1)


class MPCCost(nn.Module):
    """
    MPC 代价函数
    
    J = Σ stage_cost + terminal_cost
    
    stage_cost = ||pos - goal||²_Q + ||action||²_R + obstacle_cost
    terminal_cost = ||pos - goal||²_Qf
    """
    
    def __init__(
        self,
        Q: torch.Tensor = None,           # 位置误差权重 [2]
        R: torch.Tensor = None,           # 控制代价权重 [2]
        Qf: torch.Tensor = None,          # 终端代价权重 [2]
        obstacle_weight: float = 10.0,    # 避障权重
        safe_distance: float = 0.5,       # 安全距离
    ):
        super().__init__()
        
        # 默认权重
        if Q is None:
            Q = torch.tensor([1.0, 1.0])
        if R is None:
            R = torch.tensor([0.1, 0.1])
        if Qf is None:
            Qf = torch.tensor([10.0, 10.0])
        
        self.register_buffer('Q', Q)
        self.register_buffer('R', R)
        self.register_buffer('Qf', Qf)
        self.obstacle_weight = obstacle_weight
        self.safe_distance = safe_distance
    
    def position_cost(
        self, 
        states: torch.Tensor, 
        goal: torch.Tensor,
        terminal: bool = False
    ) -> torch.Tensor:
        """
        位置误差代价
        
        Args:
            states: [batch, 5] 或 [batch, horizon, 5]
            goal: [batch, 2] 目标位置 [gx, gy]
            terminal: 是否为终端代价
            
        Returns:
            cost: [batch] 或 [batch, horizon]
        """
        weight = self.Qf if terminal else self.Q
        
        if states.dim() == 2:
            # [batch, 5]
            pos = states[:, :2]  # [batch, 2]
            error = pos - goal  # [batch, 2]
            cost = (error ** 2 * weight).sum(dim=1)
        else:
            # [batch, horizon, 5]
            pos = states[:, :, :2]  # [batch, horizon, 2]
            goal_expanded = goal.unsqueeze(1)  # [batch, 1, 2]
            error = pos - goal_expanded  # [batch, horizon, 2]
            cost = (error ** 2 * weight).sum(dim=2)  # [batch, horizon]
        
        return cost
    
    def control_cost(self, actions: torch.Tensor) -> torch.Tensor:
        """
        控制代价
        
        Args:
            actions: [batch, horizon, 2]
            
        Returns:
            cost: [batch, horizon]
        """
        cost = (actions ** 2 * self.R).sum(dim=2)
        return cost
    
    def obstacle_cost(
        self, 
        states: torch.Tensor, 
        obstacles: torch.Tensor
    ) -> torch.Tensor:
        """
        软约束避障代价 (可微)
        
        cost = Σ_obs weight * ReLU(safe_dist - dist)²
        
        Args:
            states: [batch, horizon+1, 5]
            obstacles: [num_obs, 3] - [x, y, radius]
            
        Returns:
            cost: [batch, horizon+1]
        """
        if obstacles is None or len(obstacles) == 0:
            return torch.zeros(states.shape[0], states.shape[1], device=states.device)
        
        pos = states[:, :, :2]  # [batch, horizon+1, 2]
        batch_size, horizon_plus_1, _ = pos.shape
        num_obs = obstacles.shape[0]
        
        # 展开计算
        pos_expanded = pos.unsqueeze(2)  # [batch, horizon+1, 1, 2]
        obs_pos = obstacles[:, :2].unsqueeze(0).unsqueeze(0)  # [1, 1, num_obs, 2]
        obs_radius = obstacles[:, 2].unsqueeze(0).unsqueeze(0)  # [1, 1, num_obs]
        
        # 计算距离
        dist = torch.norm(pos_expanded - obs_pos, dim=3)  # [batch, horizon+1, num_obs]
        
        # 软约束: ReLU(safe_dist + radius - dist)²
        margin = self.safe_distance + obs_radius - dist
        violation = F.relu(margin) ** 2  # [batch, horizon+1, num_obs]
        
        # 对所有障碍物求和
        cost = self.obstacle_weight * violation.sum(dim=2)  # [batch, horizon+1]
        
        return cost
    
    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        goal: torch.Tensor,
        obstacles: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算总代价
        
        Args:
            states: [batch, horizon+1, 5] 状态轨迹
            actions: [batch, horizon, 2] 控制序列
            goal: [batch, 2] 目标位置
            obstacles: [num_obs, 3] 障碍物
            
        Returns:
            total_cost: [batch] 总代价
        """
        horizon = actions.shape[1]
        
        # Stage costs (不含最后一个状态)
        stage_states = states[:, :-1, :]  # [batch, horizon, 5]
        pos_cost = self.position_cost(stage_states, goal, terminal=False)  # [batch, horizon]
        ctrl_cost = self.control_cost(actions)  # [batch, horizon]
        
        # Terminal cost
        terminal_state = states[:, -1, :]  # [batch, 5]
        term_cost = self.position_cost(terminal_state, goal, terminal=True)  # [batch]
        
        # Obstacle cost
        obs_cost = self.obstacle_cost(states, obstacles)  # [batch, horizon+1]
        
        # 总代价
        total_cost = (
            pos_cost.sum(dim=1) + 
            ctrl_cost.sum(dim=1) + 
            term_cost + 
            obs_cost.sum(dim=1)
        )
        
        return total_cost


class DifferentiableMPC(nn.Module):
    """
    可微 MPC 控制器
    
    通过梯度下降优化控制序列，支持梯度回传到目标
    
    使用迭代 LQR / iLQR 风格的优化，但用 PyTorch 自动求导简化实现
    """
    
    def __init__(
        self,
        horizon: int = 10,
        dt: float = 0.1,
        max_v: float = 2.0,
        max_omega: float = 2.0,
        max_a_v: float = 1.0,
        max_a_omega: float = 2.0,
        num_iterations: int = 5,        # MPC 内部优化迭代次数
        lr: float = 0.5,                 # MPC 优化学习率
        Q: torch.Tensor = None,
        R: torch.Tensor = None,
        Qf: torch.Tensor = None,
        obstacle_weight: float = 10.0,
        safe_distance: float = 0.5,
        early_stop_tol: float = 1e-3,   # 早停容差
    ):
        super().__init__()
        
        self.horizon = horizon
        self.max_a_v = max_a_v
        self.max_a_omega = max_a_omega
        self.num_iterations = num_iterations
        self.lr = lr
        self.early_stop_tol = early_stop_tol
        
        # 动力学模型
        self.dynamics = DifferentiableDynamics(
            dt=dt, max_v=max_v, max_omega=max_omega
        )
        
        # 代价函数
        self.cost_fn = MPCCost(
            Q=Q, R=R, Qf=Qf,
            obstacle_weight=obstacle_weight,
            safe_distance=safe_distance
        )
        
        # 暖启动：保存上一次的解
        self.prev_actions = None
    
    def _init_actions(self, batch_size: int) -> torch.Tensor:
        """
        初始化控制序列 (暖启动或零初始化)
        
        Returns:
            actions: [batch, horizon, 2] with requires_grad=True
        """
        if self.prev_actions is not None and self.prev_actions.shape[0] == batch_size:
            # 暖启动: 左移一步，最后一步用零填充
            actions = torch.cat([
                self.prev_actions[:, 1:, :],
                torch.zeros(batch_size, 1, 2, device=self.prev_actions.device)
            ], dim=1).clone().detach()
        else:
            # 零初始化
            actions = torch.zeros(batch_size, self.horizon, 2, device=device)
        
        # 确保 requires_grad=True 用于优化
        actions = actions.requires_grad_(True)
        return actions
    
    def _clip_actions(self, actions: torch.Tensor, soft: bool = False) -> torch.Tensor:
        """
        裁剪动作到约束范围
        
        Args:
            actions: [batch, horizon, 2]
            soft: 是否使用软裁剪 (可微训练时使用)
        """
        if soft:
            # 使用 tanh 软约束
            a_v = torch.tanh(actions[:, :, 0] / self.max_a_v) * self.max_a_v
            a_omega = torch.tanh(actions[:, :, 1] / self.max_a_omega) * self.max_a_omega
        else:
            # 硬裁剪
            a_v = torch.clamp(actions[:, :, 0], -self.max_a_v, self.max_a_v)
            a_omega = torch.clamp(actions[:, :, 1], -self.max_a_omega, self.max_a_omega)
        return torch.stack([a_v, a_omega], dim=2)
    
    def forward(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
        obstacles: Optional[torch.Tensor] = None,
        return_trajectory: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        求解 MPC 并返回最优动作
        
        这是一个纯优化过程，不需要对输入求梯度。
        (端到端训练使用 get_action_with_gradient 方法)
        
        Args:
            state: [batch, state_dim] 当前状态 (可能含深度，只用前5维)
            goal: [batch, 2] 子目标位置
            obstacles: [num_obs, 3] 障碍物 [x, y, radius]
            return_trajectory: 是否返回预测轨迹
            
        Returns:
            action: [batch, 2] 最优动作 (第一步)
            cost: [batch] 总代价
            info: dict 包含预测轨迹等信息
        """
        batch_size = state.shape[0]
        
        # 只使用状态的前5维 [x, y, θ, v, ω]
        # detach 确保不影响外部计算图
        base_state = state[:, :5].detach()
        goal_detached = goal.detach()
        
        # 初始化控制序列作为可优化参数
        actions_data = torch.zeros(batch_size, self.horizon, 2, device=state.device)
        
        # 暖启动
        if self.prev_actions is not None and self.prev_actions.shape[0] == batch_size:
            actions_data = torch.cat([
                self.prev_actions[:, 1:, :],
                torch.zeros(batch_size, 1, 2, device=state.device)
            ], dim=1).clone()
        
        # 使用简单梯度下降优化 (带早停)
        prev_cost = float('inf')
        for iteration in range(self.num_iterations):
            # 创建需要梯度的副本
            actions = actions_data.clone().requires_grad_(True)
            
            # 前向传播: 展开轨迹
            clipped_actions = self._clip_actions(actions)
            states = self.dynamics.rollout(base_state, clipped_actions)
            
            # 计算代价
            cost = self.cost_fn(states, clipped_actions, goal_detached, obstacles)
            total_cost = cost.sum()
            
            # 早停检查: 如果代价变化很小，提前退出
            if abs(prev_cost - total_cost.item()) < self.early_stop_tol * batch_size:
                break
            prev_cost = total_cost.item()
            
            # 计算梯度
            total_cost.backward()
            
            # 梯度下降更新
            with torch.no_grad():
                actions_data = actions_data - self.lr * actions.grad
                # 裁剪到有效范围
                actions_data[:, :, 0].clamp_(-self.max_a_v, self.max_a_v)
                actions_data[:, :, 1].clamp_(-self.max_a_omega, self.max_a_omega)
        
        # 最终解
        final_actions = actions_data.detach()
        with torch.no_grad():
            final_states = self.dynamics.rollout(base_state, final_actions)
            final_cost = self.cost_fn(final_states, final_actions, goal_detached, obstacles)
        
        # 保存用于暖启动
        self.prev_actions = final_actions.clone()
        
        # 返回第一步动作
        first_action = final_actions[:, 0, :]
        
        info = {
            'predicted_trajectory': final_states if return_trajectory else None,
            'planned_actions': final_actions,
            'cost': final_cost
        }
        
        return first_action, final_cost, info
    
    def get_action_with_gradient(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
        obstacles: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取动作，同时保留对 goal 的梯度
        
        这是端到端训练的关键函数！
        
        原理: 使用展开优化 (Unrolled Optimization)
        - 将 MPC 的迭代优化展开为计算图
        - 梯度可以通过整个优化过程回传到 goal
        
        注意: 由于动力学的特性（加速度影响速度，速度影响位置），
        需要返回多步预测的轨迹才能有效传递梯度。
        
        Args:
            state: [batch, state_dim]
            goal: [batch, 2] (应该连接到计算图，或自己有 requires_grad)
            obstacles: [num_obs, 3]
            
        Returns:
            action: [batch, 2] 第一步动作
            predicted_trajectory: [batch, horizon+1, 5] 预测轨迹 (用于端到端训练)
            final_position: [batch, 2] 最终预测位置 (梯度连接到 goal)
        """
        batch_size = state.shape[0]
        base_state = state[:, :5].detach()  # 状态不需要梯度
        
        # 初始化动作序列
        actions = torch.zeros(
            batch_size, self.horizon, 2, 
            device=state.device,
            requires_grad=True
        )
        
        # 展开优化迭代
        for iteration in range(self.num_iterations):
            # 使用软约束裁剪
            clipped_actions = self._clip_actions(actions, soft=True)
            
            # 展开轨迹
            states = self.dynamics.rollout(base_state, clipped_actions, soft_constraints=True)
            
            # 计算代价 (goal 在这里参与计算)
            cost = self.cost_fn(states, clipped_actions, goal, obstacles)
            
            # 计算梯度
            # create_graph=True 使得 grad 成为 goal 的函数
            grad = torch.autograd.grad(
                cost.sum(), 
                actions, 
                create_graph=True,
                retain_graph=True
            )[0]
            
            # 梯度下降更新
            actions = actions - self.lr * grad
        
        # 最终轨迹 (关键: 这个轨迹依赖于 goal)
        final_actions = self._clip_actions(actions, soft=True)
        final_states = self.dynamics.rollout(base_state, final_actions, soft_constraints=True)
        
        first_action = final_actions[:, 0, :]
        # 返回最终位置 (horizon 步后的位置)，这个有到 goal 的梯度
        final_position = final_states[:, -1, :2]
        
        return first_action, final_states, final_position


class MPCWrapper:
    """
    MPC 包装器，提供与 DDPG/SAC 相同的接口
    
    用于替换 HAC 的底层策略
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        goal_dim: int,
        action_bounds: torch.Tensor,
        action_offset: torch.Tensor,
        horizon: int = 10,
        dt: float = 0.1,
        max_v: float = 2.0,
        max_omega: float = 2.0,
        max_a_v: float = 1.0,
        max_a_omega: float = 2.0,
        num_iterations: int = 5,
        lr: float = 0.5,
        Q: torch.Tensor = None,
        R: torch.Tensor = None,
        Qf: torch.Tensor = None,
        obstacle_weight: float = 10.0,
        safe_distance: float = 0.5,
        **kwargs
    ):
        """
        Args:
            state_dim: 状态维度 (可能含深度)
            action_dim: 动作维度 (2: [a_v, a_ω])
            goal_dim: 目标维度 (2: [x, y])
            action_bounds: 动作边界
            action_offset: 动作偏移
            horizon: MPC 预测步长
            dt: 时间步长
            num_iterations: MPC 优化迭代次数
            lr: MPC 优化学习率
            Q, R, Qf: 代价函数权重
            obstacle_weight: 避障权重
            safe_distance: 安全距离
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        
        self.mpc = DifferentiableMPC(
            horizon=horizon,
            dt=dt,
            max_v=max_v,
            max_omega=max_omega,
            max_a_v=max_a_v,
            max_a_omega=max_a_omega,
            num_iterations=num_iterations,
            lr=lr,
            Q=Q,
            R=R,
            Qf=Qf,
            obstacle_weight=obstacle_weight,
            safe_distance=safe_distance,
        ).to(device)
        
        # 当前障碍物信息 (需要从环境获取)
        self.obstacles = None
    
    def set_obstacles(self, obstacles: List[Tuple[float, float, float]]):
        """
        设置障碍物信息
        
        Args:
            obstacles: List of (x, y, radius)
        """
        if obstacles:
            self.obstacles = torch.tensor(obstacles, dtype=torch.float32, device=device)
        else:
            self.obstacles = None
    
    def select_action(
        self, 
        state: np.ndarray, 
        goal: np.ndarray, 
        deterministic: bool = True
    ) -> np.ndarray:
        """
        选择动作 (与 DDPG/SAC 接口一致)
        
        Args:
            state: 状态
            goal: 子目标 [x, y]
            deterministic: MPC 本身是确定性的
            
        Returns:
            action: [a_v, a_ω]
        """
        state_t = torch.FloatTensor(state.reshape(1, -1)).to(device)
        goal_t = torch.FloatTensor(goal.reshape(1, -1)).to(device)
        
        # 注意: 不用 torch.no_grad()，因为 MPC forward 内部需要计算梯度来优化
        action, _, _ = self.mpc(state_t, goal_t, self.obstacles)
        
        return action.detach().cpu().numpy().flatten()
    
    def update(self, buffer, n_iter: int, batch_size: int):
        """
        MPC 不需要学习更新 (基于模型的方法)
        保留接口兼容性
        """
        pass
    
    def save(self, directory: str, name: str):
        """MPC 不需要保存参数"""
        pass
    
    def load(self, directory: str, name: str):
        """MPC 不需要加载参数"""
        pass
