"""
Pendulum environment for HAC algorithm
Updated for gymnasium API (2024)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from os import path

class PendulumEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 30
    }

    def __init__(self, render_mode=None):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.isopen = True

        # HAC 使用 [theta, thetadot] 作为状态 (原始仓库修改)
        high = np.array([np.pi, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.state = None
        self.last_u = None

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = 10.
        m = 1.
        l = 1.
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        
        newth = angle_normalize(th + newthdot*dt)
        
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot], dtype=np.float32)
        
        # gymnasium API: return (obs, reward, terminated, truncated, info)
        terminated = False
        truncated = False
            
        return self.state, -costs, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high).astype(np.float32)
        self.last_u = None
        return self.state, {}

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

    def render(self):
        if self.render_mode is None:
            return None
            
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise ImportError("pygame is not installed, run `pip install pygame`")

        screen_dim = 500

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((screen_dim, screen_dim))
            else:  # rgb_array
                self.screen = pygame.Surface((screen_dim, screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((screen_dim, screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 2.2
        scale = screen_dim / (bound * 2)
        offset = screen_dim // 2

        rod_length = 1 * scale
        rod_width = 0.2 * scale

        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2

        # Draw rod
        theta = self.state[0] + np.pi / 2
        coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-theta)
            coord = (coord[0] + offset, coord[1] + offset)
            coords.append(coord)

        gfxdraw.aapolygon(self.surf, coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, coords, (204, 77, 77))

        # Draw axle
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        # Draw torque indicator
        if self.last_u is not None:
            # Simple torque indicator
            pass

        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.isopen

    def render_goal(self, goal, end_goal, mode='human'):
        """Render with goal visualization for HAC algorithm"""
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise ImportError("pygame is not installed, run `pip install pygame`")

        screen_dim = 500

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((screen_dim, screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((screen_dim, screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 2.2
        scale = screen_dim / (bound * 2)
        offset = screen_dim // 2

        rod_length = 1 * scale
        rod_width = 0.2 * scale
        goal_radius = 0.1 * scale

        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2

        # Draw main rod (red)
        theta = self.state[0] + np.pi / 2
        coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-theta)
            coord = (coord[0] + offset, coord[1] + offset)
            coords.append(coord)
        gfxdraw.aapolygon(self.surf, coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, coords, (204, 77, 77))

        # Draw goal indicator (yellow circle at end)
        theta_goal = goal[0] + np.pi / 2
        goal_pos = pygame.math.Vector2(rod_length, 0).rotate_rad(-theta_goal)
        goal_x = int(goal_pos[0] + offset)
        goal_y = int(goal_pos[1] + offset)
        gfxdraw.aacircle(self.surf, goal_x, goal_y, int(goal_radius), (204, 204, 77))
        gfxdraw.filled_circle(self.surf, goal_x, goal_y, int(goal_radius), (204, 204, 77))

        # Draw end goal indicator (blue circle at end)
        theta_end = end_goal[0] + np.pi / 2
        end_pos = pygame.math.Vector2(rod_length, 0).rotate_rad(-theta_end)
        end_x = int(end_pos[0] + offset)
        end_y = int(end_pos[1] + offset)
        gfxdraw.aacircle(self.surf, end_x, end_y, int(goal_radius), (77, 77, 204))
        gfxdraw.filled_circle(self.surf, end_x, end_y, int(goal_radius), (77, 77, 204))

        # Draw axle
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        self.screen.blit(self.surf, (0, 0))
        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()
        return self.isopen

    def render_goal_2(self, goal1, goal2, end_goal, mode='human'):
        """Render with multiple goal visualization for HAC algorithm (3 levels)"""
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise ImportError("pygame is not installed, run `pip install pygame`")

        screen_dim = 500

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((screen_dim, screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((screen_dim, screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 2.2
        scale = screen_dim / (bound * 2)
        offset = screen_dim // 2

        rod_length = 1 * scale
        rod_width = 0.2 * scale
        goal_radius = 0.1 * scale

        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2

        # Draw main rod (red)
        theta = self.state[0] + np.pi / 2
        coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-theta)
            coord = (coord[0] + offset, coord[1] + offset)
            coords.append(coord)
        gfxdraw.aapolygon(self.surf, coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, coords, (204, 77, 77))

        # Draw goal1 indicator (yellow)
        theta_g1 = goal1[0] + np.pi / 2
        g1_pos = pygame.math.Vector2(rod_length, 0).rotate_rad(-theta_g1)
        g1_x = int(g1_pos[0] + offset)
        g1_y = int(g1_pos[1] + offset)
        gfxdraw.aacircle(self.surf, g1_x, g1_y, int(goal_radius), (204, 204, 77))
        gfxdraw.filled_circle(self.surf, g1_x, g1_y, int(goal_radius), (204, 204, 77))

        # Draw goal2 indicator (green)
        theta_g2 = goal2[0] + np.pi / 2
        g2_pos = pygame.math.Vector2(rod_length, 0).rotate_rad(-theta_g2)
        g2_x = int(g2_pos[0] + offset)
        g2_y = int(g2_pos[1] + offset)
        gfxdraw.aacircle(self.surf, g2_x, g2_y, int(goal_radius), (77, 204, 77))
        gfxdraw.filled_circle(self.surf, g2_x, g2_y, int(goal_radius), (77, 204, 77))

        # Draw end goal indicator (blue)
        theta_end = end_goal[0] + np.pi / 2
        end_pos = pygame.math.Vector2(rod_length, 0).rotate_rad(-theta_end)
        end_x = int(end_pos[0] + offset)
        end_y = int(end_pos[1] + offset)
        gfxdraw.aacircle(self.surf, end_x, end_y, int(goal_radius), (77, 77, 204))
        gfxdraw.filled_circle(self.surf, end_x, end_y, int(goal_radius), (77, 77, 204))

        # Draw axle
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        self.screen.blit(self.surf, (0, 0))
        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()
        return self.isopen

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False
            self.screen = None


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
