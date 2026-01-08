"""坐标转换工具

提供世界坐标系与本体坐标系之间的转换函数。

本体坐标系 (极坐标):
    - r: 到目标的距离
    - theta_rel: 相对于智能体朝向的角度 ∈ [-π, π]

世界坐标系 (笛卡尔):
    - x, y: 全局位置
"""

import numpy as np
import torch
from typing import Tuple, Union

# 类型别名
ArrayLike = Union[np.ndarray, torch.Tensor]
Scalar = Union[float, np.floating, torch.Tensor]


def normalize_angle(angle: ArrayLike) -> ArrayLike:
    """
    归一化角度到 [-π, π]
    
    Args:
        angle: 输入角度 (标量或数组)
        
    Returns:
        归一化后的角度
    """
    if isinstance(angle, torch.Tensor):
        return torch.atan2(torch.sin(angle), torch.cos(angle))
    return np.arctan2(np.sin(angle), np.cos(angle))


def polar_to_world(
    agent_pos: ArrayLike,
    agent_theta: Scalar,
    r: Scalar,
    theta_rel: Scalar
) -> ArrayLike:
    """
    极坐标 (本体系) → 世界坐标
    
    Args:
        agent_pos: 智能体位置 [x, y] 或 [batch, 2]
        agent_theta: 智能体朝向 θ 或 [batch]
        r: 目标距离 或 [batch]
        theta_rel: 相对角度 或 [batch]
        
    Returns:
        世界坐标 [x, y] 或 [batch, 2]
    """
    theta_world = agent_theta + theta_rel
    
    if isinstance(agent_pos, torch.Tensor):
        dx = r * torch.cos(theta_world)
        dy = r * torch.sin(theta_world)
        if agent_pos.dim() == 1:
            return agent_pos + torch.stack([dx, dy])
        return agent_pos + torch.stack([dx, dy], dim=1)
    else:
        dx = r * np.cos(theta_world)
        dy = r * np.sin(theta_world)
        if np.ndim(agent_pos) == 1:
            return agent_pos + np.array([dx, dy])
        return agent_pos + np.stack([dx, dy], axis=1)


def world_to_polar(
    agent_pos: ArrayLike,
    agent_theta: Scalar,
    target_pos: ArrayLike
) -> Tuple[Scalar, Scalar]:
    """
    世界坐标 → 极坐标 (本体系)
    
    Args:
        agent_pos: 智能体位置 [x, y] 或 [batch, 2]
        agent_theta: 智能体朝向 θ 或 [batch]
        target_pos: 目标位置 [x, y] 或 [batch, 2]
        
    Returns:
        (r, theta_rel): 距离和相对角度
    """
    if isinstance(agent_pos, torch.Tensor):
        if agent_pos.dim() == 1:
            dx = target_pos[0] - agent_pos[0]
            dy = target_pos[1] - agent_pos[1]
        else:
            dx = target_pos[:, 0] - agent_pos[:, 0]
            dy = target_pos[:, 1] - agent_pos[:, 1]
        
        r = torch.sqrt(dx**2 + dy**2)
        theta_world = torch.atan2(dy, dx)
        theta_rel = normalize_angle(theta_world - agent_theta)
        return r, theta_rel
    else:
        if np.ndim(agent_pos) == 1:
            dx = target_pos[0] - agent_pos[0]
            dy = target_pos[1] - agent_pos[1]
        else:
            dx = target_pos[:, 0] - agent_pos[:, 0]
            dy = target_pos[:, 1] - agent_pos[:, 1]
        
        r = np.sqrt(dx**2 + dy**2)
        theta_world = np.arctan2(dy, dx)
        theta_rel = normalize_angle(theta_world - agent_theta)
        return r, theta_rel
