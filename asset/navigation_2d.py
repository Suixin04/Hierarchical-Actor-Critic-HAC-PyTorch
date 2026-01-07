# -*- coding: utf-8 -*-
"""
2D Navigation Environment - Differential Drive Robot

双轮差速移动机器人环境

动力学模型 (Unicycle Model):
    ẋ = v * cos(θ)
    ẏ = v * sin(θ)
    θ̇ = ω

State: [x, y, θ, v, ω] + [depth...] (机身坐标系)
    - x, y: 位置 (世界坐标系)
    - θ: 朝向角 [-π, π]
    - v: 线速度
    - ω: 角速度
    - depth: 深度传感器读数 (机身坐标系)

Action: [a_v, a_ω] 线加速度和角加速度
    或等效为 [左轮力, 右轮力]

Goal: [gx, gy, gθ] 目标位置和朝向

Author: HAC Project
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, List


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-π, π]"""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


class Navigation2DEnv(gym.Env):
    """
    2D Navigation Environment - Differential Drive Robot
    
    双轮差速机器人导航环境，带深度传感器
    """
    
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 30
    }
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        world_size: float = 10.0,
        max_v: float = 2.0,          # 最大线速度 m/s
        max_omega: float = 2.0,       # 最大角速度 rad/s
        max_a_v: float = 1.0,         # 最大线加速度
        max_a_omega: float = 2.0,     # 最大角加速度
        dt: float = 0.1,
        num_obstacles: int = 5,
        obstacle_radius_range: Tuple[float, float] = (0.3, 0.8),
        depth_rays: int = 16,
        depth_fov: float = 2 * np.pi,  # 360度
        depth_max_range: float = 5.0,
        agent_radius: float = 0.2,
        goal_threshold: float = 0.3,
        goal_theta_threshold: float = 0.3,  # 朝向阈值 rad (~17度)
        max_steps: int = 500,
    ):
        super().__init__()
        
        # World parameters
        self.world_size = world_size
        self.max_v = max_v
        self.max_omega = max_omega
        self.max_a_v = max_a_v
        self.max_a_omega = max_a_omega
        self.dt = dt
        self.agent_radius = agent_radius
        self.goal_threshold = goal_threshold
        self.goal_theta_threshold = goal_theta_threshold
        self.max_steps = max_steps
        
        # Obstacle parameters
        self.num_obstacles = num_obstacles
        self.obstacle_radius_range = obstacle_radius_range
        self.obstacles = []
        
        # Depth sensor parameters
        self.depth_rays = depth_rays
        self.depth_fov = depth_fov
        self.depth_max_range = depth_max_range
        
        # State dimensions: [x, y, θ, v, ω] + depth
        self.inertial_dim = 5
        self.depth_dim = depth_rays
        self.state_dim = self.inertial_dim + self.depth_dim
        
        # Action space: [a_v, a_ω] 线加速度和角加速度
        self.action_space = spaces.Box(
            low=np.array([-max_a_v, -max_a_omega]),
            high=np.array([max_a_v, max_a_omega]),
            dtype=np.float32
        )
        
        # Observation space: [x, y, θ, v, ω, depth...]
        low = np.concatenate([
            [0, 0, -np.pi, -max_v, -max_omega],
            [0] * depth_rays,
        ])
        high = np.concatenate([
            [world_size, world_size, np.pi, max_v, max_omega],
            [depth_max_range] * depth_rays,
        ])
        self.observation_space = spaces.Box(
            low=low.astype(np.float32),
            high=high.astype(np.float32),
            dtype=np.float32
        )
        
        # Rendering
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.screen_size = 600
        self.scale = self.screen_size / self.world_size
        
        # State variables
        self.agent_pos = None    # [x, y]
        self.agent_theta = 0.0   # 朝向角
        self.agent_v = 0.0       # 线速度
        self.agent_omega = 0.0   # 角速度
        self.goal_pos = None     # [gx, gy]
        self.goal_theta = 0.0    # 目标朝向
        self.steps = 0
        
        # For HAC visualization
        self.subgoal_pos = None
        self.high_level_goal = None
    
    def _generate_obstacles(self):
        """Generate random obstacles"""
        self.obstacles = []
        attempts = 0
        max_attempts = 1000
        
        while len(self.obstacles) < self.num_obstacles and attempts < max_attempts:
            attempts += 1
            
            radius = self.np_random.uniform(*self.obstacle_radius_range)
            margin = radius + self.agent_radius + 0.5
            x = self.np_random.uniform(margin, self.world_size - margin)
            y = self.np_random.uniform(margin, self.world_size - margin)
            
            valid = True
            for ox, oy, orad in self.obstacles:
                dist = np.sqrt((x - ox)**2 + (y - oy)**2)
                if dist < radius + orad + 0.3:
                    valid = False
                    break
            
            # 不阻挡起点区域
            if valid and x < 2.0 and y < 2.0:
                valid = False
            
            # 不阻挡终点区域
            if valid and x > self.world_size - 2.0 and y > self.world_size - 2.0:
                valid = False
            
            if valid:
                self.obstacles.append((x, y, radius))
    
    def _get_depth_readings(self) -> np.ndarray:
        """
        获取深度传感器读数 (机身坐标系)
        
        射线相对于机身朝向 θ 排列:
        - index 0: θ - FOV/2 方向
        - index n-1: θ + FOV/2 方向
        """
        depths = np.full(self.depth_rays, self.depth_max_range, dtype=np.float32)
        
        start_angle = self.agent_theta - self.depth_fov / 2
        angle_step = self.depth_fov / self.depth_rays
        
        for i in range(self.depth_rays):
            angle = start_angle + (i + 0.5) * angle_step
            dx = np.cos(angle)
            dy = np.sin(angle)
            
            min_dist = self.depth_max_range
            
            # 检测障碍物
            for ox, oy, orad in self.obstacles:
                dist = self._ray_circle_intersection(
                    self.agent_pos[0], self.agent_pos[1],
                    dx, dy, ox, oy, orad
                )
                if dist is not None and dist < min_dist:
                    min_dist = dist
            
            # 检测墙壁
            wall_dist = self._ray_wall_intersection(
                self.agent_pos[0], self.agent_pos[1], dx, dy
            )
            if wall_dist < min_dist:
                min_dist = wall_dist
            
            depths[i] = min_dist
        
        return depths
    
    def _ray_circle_intersection(
        self, rx: float, ry: float,
        dx: float, dy: float,
        cx: float, cy: float, radius: float
    ) -> Optional[float]:
        """射线-圆交点距离"""
        fx, fy = rx - cx, ry - cy
        a = dx * dx + dy * dy
        b = 2 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - radius * radius
        
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return None
        
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)
        
        if t1 > 0:
            return t1
        elif t2 > 0:
            return t2
        return None
    
    def _ray_wall_intersection(self, rx: float, ry: float, dx: float, dy: float) -> float:
        """射线-墙壁交点距离"""
        min_dist = self.depth_max_range
        
        walls = [
            (0, None), (self.world_size, None),
            (None, 0), (None, self.world_size)
        ]
        
        for wx, wy in walls:
            if wx is not None and dx != 0:
                t = (wx - rx) / dx
                if t > 0:
                    y_hit = ry + t * dy
                    if 0 <= y_hit <= self.world_size:
                        min_dist = min(min_dist, t)
            elif wy is not None and dy != 0:
                t = (wy - ry) / dy
                if t > 0:
                    x_hit = rx + t * dx
                    if 0 <= x_hit <= self.world_size:
                        min_dist = min(min_dist, t)
        
        return min_dist
    
    def _check_obstacle_collision(self) -> bool:
        """检测障碍物碰撞"""
        x, y = self.agent_pos
        for ox, oy, orad in self.obstacles:
            dist = np.sqrt((x - ox)**2 + (y - oy)**2)
            if dist < self.agent_radius + orad:
                return True
        return False
    
    def _get_obs(self) -> np.ndarray:
        """
        构建观测
        State: [x, y, θ, v, ω, depth...]
        """
        inertial = np.array([
            self.agent_pos[0], self.agent_pos[1],
            self.agent_theta,
            self.agent_v,
            self.agent_omega
        ], dtype=np.float32)
        
        depth = self._get_depth_readings()
        
        return np.concatenate([inertial, depth])
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """重置环境"""
        super().reset(seed=seed)
        
        self._generate_obstacles()
        
        # 随机起点 (左下角区域)
        self.agent_pos = np.array([
            self.np_random.uniform(0.5, 2.0),
            self.np_random.uniform(0.5, 2.0)
        ], dtype=np.float32)
        
        # 随机终点 (右上角区域)
        self.goal_pos = np.array([
            self.np_random.uniform(self.world_size - 2.0, self.world_size - 0.5),
            self.np_random.uniform(self.world_size - 2.0, self.world_size - 0.5)
        ], dtype=np.float32)
        
        # 初始朝向指向目标
        goal_dir = self.goal_pos - self.agent_pos
        self.agent_theta = np.arctan2(goal_dir[1], goal_dir[0])
        self.agent_v = 0.0
        self.agent_omega = 0.0
        
        # 目标朝向 (指向右上角)
        self.goal_theta = np.pi / 4
        
        self.steps = 0
        self.subgoal_pos = None
        self.high_level_goal = None
        
        return self._get_obs(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        执行动作 - 差速机器人动力学
        
        Action: [a_v, a_ω] 线加速度和角加速度
        
        动力学:
            v_new = v + a_v * dt
            ω_new = ω + a_ω * dt
            θ_new = θ + ω_new * dt
            x_new = x + v_new * cos(θ_new) * dt
            y_new = y + v_new * sin(θ_new) * dt
        """
        self.steps += 1
        
        # 裁剪动作
        a_v = np.clip(action[0], -self.max_a_v, self.max_a_v)
        a_omega = np.clip(action[1], -self.max_a_omega, self.max_a_omega)
        
        # 更新速度 (带阻尼)
        damping_v = 0.95
        damping_omega = 0.9
        self.agent_v = damping_v * self.agent_v + a_v * self.dt
        self.agent_omega = damping_omega * self.agent_omega + a_omega * self.dt
        
        # 裁剪速度
        self.agent_v = np.clip(self.agent_v, -self.max_v, self.max_v)
        self.agent_omega = np.clip(self.agent_omega, -self.max_omega, self.max_omega)
        
        # 更新朝向
        self.agent_theta = normalize_angle(self.agent_theta + self.agent_omega * self.dt)
        
        # 计算新位置
        new_x = self.agent_pos[0] + self.agent_v * np.cos(self.agent_theta) * self.dt
        new_y = self.agent_pos[1] + self.agent_v * np.sin(self.agent_theta) * self.dt
        new_pos = np.array([new_x, new_y])
        
        # 检测墙壁碰撞 (碰撞时速度归零，位置不更新)
        wall_collision = (
            new_pos[0] - self.agent_radius < 0 or
            new_pos[0] + self.agent_radius > self.world_size or
            new_pos[1] - self.agent_radius < 0 or
            new_pos[1] + self.agent_radius > self.world_size
        )
        
        # 检测障碍物碰撞 (在更新位置前检测)
        obstacle_collision = False
        for ox, oy, orad in self.obstacles:
            dist = np.sqrt((new_pos[0] - ox)**2 + (new_pos[1] - oy)**2)
            if dist < self.agent_radius + orad:
                obstacle_collision = True
                break
        
        # 只有在不碰撞时才更新位置
        if not wall_collision and not obstacle_collision:
            self.agent_pos = new_pos
        else:
            # 碰撞时速度反向并衰减
            self.agent_v *= -0.3
            self.agent_omega *= 0.5
            
            # 如果是墙壁碰撞，裁剪位置到边界内
            if wall_collision:
                self.agent_pos = np.clip(
                    self.agent_pos,
                    self.agent_radius + 0.01,
                    self.world_size - self.agent_radius - 0.01
                )
        
        collision = wall_collision or obstacle_collision
        
        # 检测是否到达目标
        dist_to_goal = np.linalg.norm(self.agent_pos - self.goal_pos)
        theta_error = abs(normalize_angle(self.agent_theta - self.goal_theta))
        reached_goal = dist_to_goal < self.goal_threshold
        
        timeout = self.steps >= self.max_steps
        
        # 终止条件：只有到达目标才终止，碰撞不终止（让智能体学习避障）
        terminated = reached_goal
        truncated = timeout and not terminated
        
        # 稀疏奖励
        reward = 0.0 if reached_goal else -1.0
        
        info = {
            'collision': collision,
            'wall_collision': wall_collision,
            'obstacle_collision': obstacle_collision,
            'reached_goal': reached_goal,
            'dist_to_goal': dist_to_goal,
            'theta_error': theta_error
        }
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def _init_pygame(self):
        """初始化 pygame"""
        if self.render_mode is None:
            return None
        
        try:
            import pygame
        except ImportError:
            return None
        
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
                pygame.display.set_caption("Navigation2D - Differential Drive")
            else:
                self.screen = pygame.Surface((self.screen_size, self.screen_size))
        
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        return pygame
    
    def _draw_base(self, pygame):
        """绘制基础场景（不包含 flip）"""
        self.screen.fill((255, 255, 255))
        
        # 绘制障碍物
        for ox, oy, orad in self.obstacles:
            pygame.draw.circle(
                self.screen, (100, 100, 100),
                (int(ox * self.scale), int((self.world_size - oy) * self.scale)),
                int(orad * self.scale)
            )
        
        # 绘制目标
        goal_screen = (
            int(self.goal_pos[0] * self.scale),
            int((self.world_size - self.goal_pos[1]) * self.scale)
        )
        pygame.draw.circle(self.screen, (0, 200, 0), goal_screen, 15)
        # 绘制目标朝向
        goal_arrow_end = (
            int((self.goal_pos[0] + np.cos(self.goal_theta) * 0.5) * self.scale),
            int((self.world_size - (self.goal_pos[1] + np.sin(self.goal_theta) * 0.5)) * self.scale)
        )
        pygame.draw.line(self.screen, (0, 150, 0), goal_screen, goal_arrow_end, 3)
        
        # 绘制深度射线
        self._draw_depth_rays(pygame)
        
        # 绘制机器人
        agent_screen = (
            int(self.agent_pos[0] * self.scale),
            int((self.world_size - self.agent_pos[1]) * self.scale)
        )
        pygame.draw.circle(self.screen, (50, 50, 200), agent_screen, int(self.agent_radius * self.scale))
        
        # 绘制机器人朝向
        arrow_len = 0.5
        arrow_end = (
            int((self.agent_pos[0] + np.cos(self.agent_theta) * arrow_len) * self.scale),
            int((self.world_size - (self.agent_pos[1] + np.sin(self.agent_theta) * arrow_len)) * self.scale)
        )
        pygame.draw.line(self.screen, (200, 50, 50), agent_screen, arrow_end, 3)
    
    def render(self):
        """渲染环境"""
        pygame = self._init_pygame()
        if pygame is None:
            return
        
        self._draw_base(pygame)
        
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        
        if self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))
        
        return True
    
    def _draw_depth_rays(self, pygame):
        """绘制深度射线"""
        if self.depth_rays == 0:
            return
        
        start_angle = self.agent_theta - self.depth_fov / 2
        angle_step = self.depth_fov / self.depth_rays
        depths = self._get_depth_readings()
        
        for i in range(self.depth_rays):
            angle = start_angle + (i + 0.5) * angle_step
            dx, dy = np.cos(angle), np.sin(angle)
            
            end_x = self.agent_pos[0] + dx * depths[i]
            end_y = self.agent_pos[1] + dy * depths[i]
            
            start_screen = (
                int(self.agent_pos[0] * self.scale),
                int((self.world_size - self.agent_pos[1]) * self.scale)
            )
            end_screen = (
                int(end_x * self.scale),
                int((self.world_size - end_y) * self.scale)
            )
            
            ratio = depths[i] / self.depth_max_range
            color = (int(255 * (1 - ratio)), int(255 * ratio), 0)
            pygame.draw.line(self.screen, color, start_screen, end_screen, 1)
    
    def render_subgoals(self, goals: list, mode='human'):
        """
        通用子目标渲染方法 - 支持任意层级数量
        
        Args:
            goals: 目标列表，从底层到高层 [goal_0, goal_1, ..., goal_k-1]
                   goal_0: 底层目标 (最小的圆)
                   goal_k-1: 最高层目标 (最终目标，最大的圆)
            mode: 渲染模式
        """
        if self.render_mode is None:
            return
        
        pygame = self._init_pygame()
        if pygame is None:
            return
        
        # 先绘制基础场景（不 flip）
        self._draw_base(pygame)
        
        if not goals:
            if self.render_mode == "human":
                pygame.event.pump()
                pygame.display.flip()
                self.clock.tick(self.metadata["render_fps"])
            return
        
        # 颜色梯度: 从紫色(底层)到橙色(高层)，最终目标为绿色
        # 使用HSV色彩空间: 紫色(280°) -> 橙色(30°)
        k_level = len(goals)
        
        for i, goal in enumerate(goals):
            if goal is None:
                continue
            
            # 计算颜色 (从紫色到橙色到绿色)
            if i == k_level - 1:
                # 最终目标: 绿色 (已在 render() 中绘制)
                continue
            else:
                # 子目标: 紫色(底层) -> 橙色(高层)
                t = i / max(k_level - 2, 1) if k_level > 2 else 0
                # 紫色 RGB(150, 50, 150) -> 橙色 RGB(255, 150, 0)
                r = int(150 + t * 105)
                g = int(50 + t * 100)
                b = int(150 - t * 150)
                color = (r, g, b)
            
            # 计算圆的大小: 底层最小，高层最大
            base_radius = 6
            radius = base_radius + i * 3
            
            # 绘制目标位置
            pos_screen = (
                int(goal[0] * self.scale),
                int((self.world_size - goal[1]) * self.scale)
            )
            pygame.draw.circle(self.screen, color, pos_screen, radius)
            pygame.draw.circle(self.screen, (50, 50, 50), pos_screen, radius, 1)  # 边框
            
            # 如果目标包含朝向 (第3维)，绘制朝向箭头
            if len(goal) >= 3:
                theta = goal[2]
                arrow_len = 0.3 + i * 0.1
                arrow_end = (
                    int((goal[0] + np.cos(theta) * arrow_len) * self.scale),
                    int((self.world_size - (goal[1] + np.sin(theta) * arrow_len)) * self.scale)
                )
                pygame.draw.line(self.screen, color, pos_screen, arrow_end, 2)
        
        # 统一在最后 flip
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
    
    # 保留旧接口以兼容
    def render_goal(self, subgoal: np.ndarray, end_goal: np.ndarray, mode='human'):
        """渲染HAC子目标 (2层) - 兼容旧接口"""
        self.render_subgoals([subgoal, end_goal], mode)
    
    def render_goal_2(self, low_goal, mid_goal, high_goal, mode='human'):
        """渲染HAC子目标 (3层) - 兼容旧接口"""
        self.render_subgoals([low_goal, mid_goal, high_goal], mode)
    
    def close(self):
        """关闭环境"""
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None


class Navigation2DSimpleEnv(Navigation2DEnv):
    """
    简化版差速机器人 (无深度传感器，无障碍物)
    
    State: [x, y, θ, v, ω]
    Goal: [x, y, θ]
    
    适合用于初始训练和调试
    """
    
    def __init__(self, render_mode=None, **kwargs):
        # 无障碍物，无深度传感器
        super().__init__(render_mode=render_mode, depth_rays=0, num_obstacles=0, **kwargs)
        
        self.state_dim = self.inertial_dim  # 5
        self.depth_dim = 0
        
        low = np.array([0, 0, -np.pi, -self.max_v, -self.max_omega], dtype=np.float32)
        high = np.array([self.world_size, self.world_size, np.pi, self.max_v, self.max_omega], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
    
    def _get_obs(self) -> np.ndarray:
        """State: [x, y, θ, v, ω]"""
        return np.array([
            self.agent_pos[0], self.agent_pos[1],
            self.agent_theta,
            self.agent_v,
            self.agent_omega
        ], dtype=np.float32)
