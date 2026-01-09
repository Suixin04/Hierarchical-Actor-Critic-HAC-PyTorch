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
        num_obstacles: int = 6,
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
        collision_obstacle = None
        for ox, oy, orad in self.obstacles:
            dist = np.sqrt((new_pos[0] - ox)**2 + (new_pos[1] - oy)**2)
            if dist < self.agent_radius + orad:
                obstacle_collision = True
                collision_obstacle = (ox, oy, orad)
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
            
            # 如果是障碍物碰撞，将机器人推离障碍物
            if obstacle_collision and collision_obstacle is not None:
                ox, oy, orad = collision_obstacle
                # 计算从障碍物中心指向机器人的方向
                dx = self.agent_pos[0] - ox
                dy = self.agent_pos[1] - oy
                dist = np.sqrt(dx**2 + dy**2)
                if dist > 1e-6:
                    # 归一化方向向量
                    nx, ny = dx / dist, dy / dist
                    # 计算需要移动的距离，确保机器人完全在障碍物外面
                    min_safe_dist = self.agent_radius + orad + 0.05
                    push_dist = min_safe_dist - dist
                    if push_dist > 0:
                        # 推离障碍物
                        new_x = self.agent_pos[0] + nx * push_dist
                        new_y = self.agent_pos[1] + ny * push_dist
                        # 确保不会推到墙外
                        self.agent_pos[0] = np.clip(new_x, self.agent_radius + 0.01, self.world_size - self.agent_radius - 0.01)
                        self.agent_pos[1] = np.clip(new_y, self.agent_radius + 0.01, self.world_size - self.agent_radius - 0.01)
        
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
    
    def _init_pygame(self, force_human=False):
        """初始化 pygame"""
        render_mode = self.render_mode
        if force_human:
            render_mode = "human"
            self.render_mode = "human"
        
        if render_mode is None:
            return None
        
        try:
            import pygame
        except ImportError:
            print("[WARNING] pygame not installed. Run: pip install pygame")
            return None
        
        # 初始化 pygame（不初始化 font 模块）
        if not pygame.display.get_init():
            pygame.display.init()
        
        if self.screen is None:
            if render_mode == "human":
                self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
                pygame.display.set_caption("Navigation2D - HAC")
            else:
                self.screen = pygame.Surface((self.screen_size, self.screen_size))
        
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        return pygame
    
    def _draw_base(self, pygame):
        """绘制基础场景"""
        self.screen.fill((245, 245, 245))
        
        # 网格
        for i in range(int(self.world_size) + 1):
            x = int(i * self.scale)
            pygame.draw.line(self.screen, (220, 220, 220), (x, 0), (x, self.screen_size), 1)
            y = int(i * self.scale)
            pygame.draw.line(self.screen, (220, 220, 220), (0, y), (self.screen_size, y), 1)
        
        # 边界
        pygame.draw.rect(self.screen, (100, 100, 100), (0, 0, self.screen_size, self.screen_size), 3)
        
        # 障碍物
        for ox, oy, orad in self.obstacles:
            cx = int(ox * self.scale)
            cy = int((self.world_size - oy) * self.scale)
            r = int(orad * self.scale)
            pygame.draw.circle(self.screen, (80, 80, 80), (cx + 3, cy + 3), r)
            pygame.draw.circle(self.screen, (120, 120, 120), (cx, cy), r)
        
        # 目标（绿色）
        gx = int(self.goal_pos[0] * self.scale)
        gy = int((self.world_size - self.goal_pos[1]) * self.scale)
        pygame.draw.circle(self.screen, (0, 200, 0), (gx, gy), 18, 3)
        pygame.draw.circle(self.screen, (100, 255, 100), (gx, gy), 12)
        # 目标朝向
        gax = int((self.goal_pos[0] + np.cos(self.goal_theta) * 0.6) * self.scale)
        gay = int((self.world_size - (self.goal_pos[1] + np.sin(self.goal_theta) * 0.6)) * self.scale)
        pygame.draw.line(self.screen, (0, 150, 0), (gx, gy), (gax, gay), 4)
        
        # 深度射线
        self._draw_depth_rays(pygame)
        
        # 机器人
        ax = int(self.agent_pos[0] * self.scale)
        ay = int((self.world_size - self.agent_pos[1]) * self.scale)
        ar = int(self.agent_radius * self.scale)
        pygame.draw.circle(self.screen, (30, 30, 100), (ax + 2, ay + 2), ar)
        pygame.draw.circle(self.screen, (70, 70, 220), (ax, ay), ar)
        pygame.draw.circle(self.screen, (255, 255, 255), (ax, ay), ar, 2)
        
        # 机器人朝向
        aax = int((self.agent_pos[0] + np.cos(self.agent_theta) * 0.5) * self.scale)
        aay = int((self.world_size - (self.agent_pos[1] + np.sin(self.agent_theta) * 0.5)) * self.scale)
        pygame.draw.line(self.screen, (255, 100, 100), (ax, ay), (aax, aay), 4)
    
    def render(self):
        """渲染环境"""
        pygame = self._init_pygame()
        if pygame is None:
            return None
        
        self._draw_base(pygame)
        
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None
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
        
        ax = int(self.agent_pos[0] * self.scale)
        ay = int((self.world_size - self.agent_pos[1]) * self.scale)
        
        for i in range(self.depth_rays):
            angle = start_angle + (i + 0.5) * angle_step
            end_x = int((self.agent_pos[0] + np.cos(angle) * depths[i]) * self.scale)
            end_y = int((self.world_size - (self.agent_pos[1] + np.sin(angle) * depths[i])) * self.scale)
            
            ratio = depths[i] / self.depth_max_range
            color = (int(180 * (1 - ratio)), int(180 * ratio), 80)
            pygame.draw.line(self.screen, color, (ax, ay), (end_x, end_y), 1)
    
    def render_subgoals(self, goals: list, mode='human'):
        """渲染带有分层子目标的场景"""
        pygame = self._init_pygame(force_human=(mode == 'human'))
        if pygame is None:
            return
        
        self._draw_base(pygame)
        
        if not goals:
            if self.render_mode == "human":
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.close()
                        return
                pygame.display.flip()
                self.clock.tick(self.metadata["render_fps"])
            return
        
        # 层级颜色
        colors = [
            (255, 80, 80),    # L0: 红
            (255, 200, 50),   # L1: 黄
            (100, 255, 200),  # L2: 青
            (150, 100, 255),  # L3: 紫
        ]
        
        k = len(goals)
        
        # 从高层到底层绘制
        for i in range(k - 1, -1, -1):
            g = goals[i]
            if g is None or i == k - 1:
                continue
            
            color = colors[min(i, len(colors) - 1)]
            radius = 12 - i * 2
            
            px = int(g[0] * self.scale)
            py = int((self.world_size - g[1]) * self.scale)
            
            # 圆
            pygame.draw.circle(self.screen, (50, 50, 50), (px, py), radius + 2, 2)
            pygame.draw.circle(self.screen, color, (px, py), radius)
            
            # 朝向箭头
            if len(g) >= 4:
                theta = g[3]
                arrow_len = 0.3 + g[2] * 0.2
            elif len(g) >= 3:
                theta = g[2]
                arrow_len = 0.4
            else:
                continue
            
            ex = int((g[0] + np.cos(theta) * arrow_len) * self.scale)
            ey = int((self.world_size - (g[1] + np.sin(theta) * arrow_len)) * self.scale)
            pygame.draw.line(self.screen, color, (px, py), (ex, ey), 3)
        
        # 机器人到目标的虚线
        if goals[0] is not None:
            ax = int(self.agent_pos[0] * self.scale)
            ay = int((self.world_size - self.agent_pos[1]) * self.scale)
            gx = int(goals[0][0] * self.scale)
            gy = int((self.world_size - goals[0][1]) * self.scale)
            
            dx, dy = gx - ax, gy - ay
            dist = max(1, int(np.sqrt(dx*dx + dy*dy)))
            for j in range(0, dist, 16):
                s = (ax + dx * j // dist, ay + dy * j // dist)
                e = (ax + dx * min(j + 8, dist) // dist, ay + dy * min(j + 8, dist) // dist)
                pygame.draw.line(self.screen, (200, 100, 100), s, e, 2)
        
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return
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
        """关闭环境 - 安全清理"""
        if self.screen is not None:
            try:
                import pygame
                pygame.display.quit()
                pygame.quit()
            except:
                pass
            finally:
                self.screen = None
                self.clock = None
