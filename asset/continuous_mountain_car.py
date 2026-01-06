# -*- coding: utf-8 -*-
"""
@author: Olivier Sigaud

A merge between two sources:

* Adaptation of the MountainCar Environment from the "FAReinforcement" library
of Jose Antonio Martin H. (version 1.0), adapted by  'Tom Schaul, tom@idsia.ch'
and then modified by Arnaud de Broissia

* the OpenAI/gym MountainCar environment
itself from
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC

Updated for gymnasium API (2024)
"""

import math

import numpy as np

import gymnasium as gym
from gymnasium import spaces

class Continuous_MountainCarEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 30
    }

    def __init__(self, render_mode=None):
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.45 # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        self.power = 0.0015

        self.low_state = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high_state = np.array([self.max_position, self.max_speed], dtype=np.float32)

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.isopen = True

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action,
                                       shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state,
                                            dtype=np.float32)

        self.state = None

    def step(self, action):
        position = self.state[0]
        velocity = self.state[1]
        force = min(max(action[0], -1.0), 1.0)

        velocity += force*self.power -0.0025 * math.cos(3*position)
        if (velocity > self.max_speed): velocity = self.max_speed
        if (velocity < -self.max_speed): velocity = -self.max_speed
        position += velocity
        if (position > self.max_position): position = self.max_position
        if (position < self.min_position): position = self.min_position
        if (position==self.min_position and velocity<0): velocity = 0

        terminated = bool(position >= self.goal_position)
        truncated = False

        reward = 0
        if terminated:
            reward = 100.0
        reward -= math.pow(action[0],2)*0.1

        self.state = np.array([position, velocity], dtype=np.float32)
            
        return self.state, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0], dtype=np.float32)
            
        return np.array(self.state, dtype=np.float32), {}

    def _height(self, xs):
        return np.sin(3 * xs)*.45+.55

    def render(self):
        if self.render_mode is None:
            return None
            
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise ImportError("pygame is not installed, run `pip install pygame`")

        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            else:  # rgb_array
                self.screen = pygame.Surface((screen_width, screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))

        pos = self.state[0]

        xs = np.linspace(self.min_position, self.max_position, 100)
        ys = self._height(xs)
        xys = list(zip((xs - self.min_position) * scale, ys * scale))

        pygame.draw.aalines(self.surf, points=xys, closed=False, color=(0, 0, 0))

        clearance = 10

        l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
        coords = []
        for c in [(l, b), (l, t), (r, t), (r, b)]:
            c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
            coords.append(
                (
                    c[0] + (pos - self.min_position) * scale,
                    c[1] + clearance + self._height(pos) * scale,
                )
            )

        gfxdraw.aapolygon(self.surf, coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, coords, (0, 0, 0))

        for c in [(carwidth / 4, 0), (-carwidth / 4, 0)]:
            c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
            wheel = (
                int(c[0] + (pos - self.min_position) * scale),
                int(c[1] + clearance + self._height(pos) * scale),
            )

            gfxdraw.aacircle(
                self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
            )
            gfxdraw.filled_circle(
                self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
            )

        flagx = int((self.goal_position - self.min_position) * scale)
        flagy1 = int(self._height(self.goal_position) * scale)
        flagy2 = flagy1 + 50
        gfxdraw.vline(self.surf, flagx, flagy1, flagy2, (0, 0, 0))

        gfxdraw.aapolygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )
        gfxdraw.filled_polygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )

        self.surf = pygame.transform.flip(self.surf, False, True)
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

        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))

        pos = self.state[0]

        xs = np.linspace(self.min_position, self.max_position, 100)
        ys = self._height(xs)
        xys = list(zip((xs - self.min_position) * scale, ys * scale))

        pygame.draw.aalines(self.surf, points=xys, closed=False, color=(0, 0, 0))

        clearance = 10
        l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0

        # Draw main car (black)
        coords = []
        for c in [(l, b), (l, t), (r, t), (r, b)]:
            c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
            coords.append(
                (
                    c[0] + (pos - self.min_position) * scale,
                    c[1] + clearance + self._height(pos) * scale,
                )
            )
        gfxdraw.aapolygon(self.surf, coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, coords, (0, 0, 0))

        # Draw goal car (red)
        pos1 = goal[0]
        coords1 = []
        for c in [(l, b), (l, t), (r, t), (r, b)]:
            c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos1))
            coords1.append(
                (
                    c[0] + (pos1 - self.min_position) * scale,
                    c[1] + clearance + self._height(pos1) * scale,
                )
            )
        gfxdraw.aapolygon(self.surf, coords1, (255, 50, 50))
        gfxdraw.filled_polygon(self.surf, coords1, (255, 50, 50))

        # Draw end goal car (blue)
        pos2 = end_goal[0]
        coords2 = []
        for c in [(l, b), (l, t), (r, t), (r, b)]:
            c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos2))
            coords2.append(
                (
                    c[0] + (pos2 - self.min_position) * scale,
                    c[1] + clearance + self._height(pos2) * scale,
                )
            )
        gfxdraw.aapolygon(self.surf, coords2, (50, 50, 255))
        gfxdraw.filled_polygon(self.surf, coords2, (50, 50, 255))

        # Draw flag
        flagx = int((self.goal_position - self.min_position) * scale)
        flagy1 = int(self._height(self.goal_position) * scale)
        flagy2 = flagy1 + 50
        gfxdraw.vline(self.surf, flagx, flagy1, flagy2, (0, 0, 0))
        gfxdraw.aapolygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )
        gfxdraw.filled_polygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )

        self.surf = pygame.transform.flip(self.surf, False, True)
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

        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))

        pos = self.state[0]

        xs = np.linspace(self.min_position, self.max_position, 100)
        ys = self._height(xs)
        xys = list(zip((xs - self.min_position) * scale, ys * scale))

        pygame.draw.aalines(self.surf, points=xys, closed=False, color=(0, 0, 0))

        clearance = 10
        l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0

        # Draw main car (black)
        coords = []
        for c in [(l, b), (l, t), (r, t), (r, b)]:
            c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
            coords.append(
                (
                    c[0] + (pos - self.min_position) * scale,
                    c[1] + clearance + self._height(pos) * scale,
                )
            )
        gfxdraw.aapolygon(self.surf, coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, coords, (0, 0, 0))

        # Draw goal1 car (red)
        pos1 = goal1[0]
        coords1 = []
        for c in [(l, b), (l, t), (r, t), (r, b)]:
            c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos1))
            coords1.append(
                (
                    c[0] + (pos1 - self.min_position) * scale,
                    c[1] + clearance + self._height(pos1) * scale,
                )
            )
        gfxdraw.aapolygon(self.surf, coords1, (255, 50, 50))
        gfxdraw.filled_polygon(self.surf, coords1, (255, 50, 50))

        # Draw goal2 car (green)
        pos2 = goal2[0]
        coords2 = []
        for c in [(l, b), (l, t), (r, t), (r, b)]:
            c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos2))
            coords2.append(
                (
                    c[0] + (pos2 - self.min_position) * scale,
                    c[1] + clearance + self._height(pos2) * scale,
                )
            )
        gfxdraw.aapolygon(self.surf, coords2, (50, 255, 50))
        gfxdraw.filled_polygon(self.surf, coords2, (50, 255, 50))

        # Draw end goal car (blue)
        pos3 = end_goal[0]
        coords3 = []
        for c in [(l, b), (l, t), (r, t), (r, b)]:
            c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos3))
            coords3.append(
                (
                    c[0] + (pos3 - self.min_position) * scale,
                    c[1] + clearance + self._height(pos3) * scale,
                )
            )
        gfxdraw.aapolygon(self.surf, coords3, (50, 50, 255))
        gfxdraw.filled_polygon(self.surf, coords3, (50, 50, 255))

        # Draw flag
        flagx = int((self.goal_position - self.min_position) * scale)
        flagy1 = int(self._height(self.goal_position) * scale)
        flagy2 = flagy1 + 50
        gfxdraw.vline(self.surf, flagx, flagy1, flagy2, (0, 0, 0))
        gfxdraw.aapolygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )
        gfxdraw.filled_polygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )

        self.surf = pygame.transform.flip(self.surf, False, True)
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
