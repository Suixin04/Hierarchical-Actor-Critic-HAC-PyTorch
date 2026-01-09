"""
HAC-MPC: Hierarchical Actor-Critic with Model Predictive Control

分层强化学习框架。

目录结构:
    src/
    ├── agents/          # 强化学习智能体
    │   ├── sac.py       # SAC 算法
    │   └── hac.py       # HAC 分层算法
    ├── models/          # 神经网络模型
    │   ├── actor.py     # Actor 网络
    │   ├── critic.py    # Critic 网络
    │   └── encoder.py   # 深度编码器
    ├── control/         # 控制器
    │   ├── dynamics.py  # 动力学模型
    │   ├── mpc.py       # MPC 控制器
    │   └── cost.py      # 代价函数
    ├── buffers/         # 经验回放
    │   └── replay_buffer.py
    ├── utils/           # 工具函数
    │   ├── coordinate.py    # 坐标转换
    │   └── common.py        # 通用工具
    ├── configs/         # 配置管理
    │   ├── base.py
    │   └── navigation.py
    └── evaluation/      # 评估和可视化
        ├── metrics.py
        └── visualize.py
"""

__version__ = "2.0.0"

