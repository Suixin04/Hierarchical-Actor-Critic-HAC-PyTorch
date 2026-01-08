"""深度编码器

将高维深度传感器数据编码为低维嵌入表示。

设计选择:
- 使用 LeakyReLU: 避免 ReLU 的死神经元和 Tanh 的饱和问题
- 保留原始深度值: 不进行归一化，让网络学习如何利用绝对距离信息
- 输出无范围限制: 让后续网络自己学习如何利用 embedding
"""

import torch
import torch.nn as nn


class DepthEncoder(nn.Module):
    """
    深度信息编码器
    
    将原始深度读数 [depth_dim] 编码为低维嵌入 [embedding_dim]。
    
    Args:
        depth_dim: 输入深度维度 (射线数量)
        embedding_dim: 输出嵌入维度
        hidden_dim: 隐藏层维度
    """
    
    def __init__(
        self, 
        depth_dim: int = 16, 
        embedding_dim: int = 8,
        hidden_dim: int = 32
    ):
        super().__init__()
        
        self.depth_dim = depth_dim
        self.embedding_dim = embedding_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(depth_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LeakyReLU(0.1)  # 保持梯度流动
        )
    
    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """
        编码深度信息
        
        Args:
            depth: [batch, depth_dim] 原始深度读数
            
        Returns:
            embedding: [batch, embedding_dim] 深度嵌入
        """
        return self.encoder(depth)


def create_depth_encoder(
    depth_dim: int = 16,
    embedding_dim: int = 8,
    hidden_dim: int = 32,
    device: torch.device = None
) -> DepthEncoder:
    """
    工厂函数: 创建深度编码器
    
    Args:
        depth_dim: 输入深度维度
        embedding_dim: 输出嵌入维度
        hidden_dim: 隐藏层维度
        device: 设备
        
    Returns:
        DepthEncoder 实例
    """
    encoder = DepthEncoder(depth_dim, embedding_dim, hidden_dim)
    if device is not None:
        encoder = encoder.to(device)
    return encoder
