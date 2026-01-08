"""通用工具函数"""

import torch
import numpy as np
import random
from typing import Optional

# 全局设备 (单例模式，避免多处定义不一致)
_DEVICE: Optional[torch.device] = None


def get_device() -> torch.device:
    """获取计算设备 (单例)"""
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return _DEVICE


def set_seed(seed: int) -> None:
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def to_tensor(
    x: np.ndarray,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """将 numpy 数组转换为 tensor"""
    if device is None:
        device = get_device()
    return torch.as_tensor(x, dtype=dtype, device=device)


def to_numpy(x: torch.Tensor) -> np.ndarray:
    """将 tensor 转换为 numpy 数组"""
    return x.detach().cpu().numpy()
