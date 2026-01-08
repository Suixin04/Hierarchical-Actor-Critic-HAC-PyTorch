# HAC-MPC: Hierarchical Actor-Critic with Model Predictive Control

一个清晰、模块化的分层强化学习框架。

## 项目结构

```
├── src/                    # 源代码
│   ├── agents/             # 强化学习智能体
│   │   ├── sac.py          # SAC 算法
│   │   └── hac.py          # HAC 分层算法
│   ├── models/             # 神经网络模型
│   │   ├── encoder.py      # 深度编码器
│   │   ├── actor.py        # Actor 网络
│   │   └── critic.py       # Critic 网络
│   ├── control/            # 控制器
│   │   ├── dynamics.py     # 动力学模型
│   │   ├── mpc.py          # MPC 控制器
│   │   └── cost.py         # 代价函数
│   ├── buffers/            # 经验回放
│   │   └── replay_buffer.py
│   ├── utils/              # 工具函数
│   │   ├── coordinate.py   # 坐标转换
│   │   └── common.py       # 通用工具
│   ├── configs/            # 配置管理
│   │   ├── base.py
│   │   └── navigation.py
│   └── train.py            # 训练逻辑
├── asset/                  # 自定义 Gymnasium 环境
├── train.py                # 训练入口脚本
├── test.py                 # 单元测试
└── requirements.txt        # 依赖
```

## 快速开始

### 训练

```bash
# 默认训练
python train.py

# 开启渲染
python train.py --render

# 自定义参数
python train.py --e2e_episodes 500 --max_episodes 2000
```

### 测试

```bash
python test.py
```

### 在代码中使用

```python
from src.configs import get_config
from src.agents import HACAgent

# 获取配置
config = get_config("Navigation2DObstacle-v1")

# 创建智能体
agent = HACAgent(config)

# 训练循环
for episode in range(config.max_episodes):
    state, _ = env.reset()
    last_state, done = agent.run_HAC(
        env, config.k_level - 1, state, goal, is_subgoal_test=False
    )
    agent.update(config.n_iter, config.batch_size)
```

## 架构设计

### 分层结构

- **Level 0 (底层)**: MPC 控制器
  - 使用已知动力学模型
  - 负责轨迹追踪
  - 不需要学习

- **Level 1+ (高层)**: SAC 策略
  - 输出子目标（世界坐标）
  - 使用深度编码器处理传感器信息
  - 通过 RL 和 E2E 训练学习

### 训练阶段

1. **Phase 1 (E2E 预热)**
   - Level 1: E2E 更新 Encoder，RL 更新 Actor
   - Level 2+: RL 更新

2. **Phase 2 (RL 微调)**
   - Level 1: Encoder 微调或冻结，RL 更新 Actor
   - Level 2+: RL 继续更新

## 配置系统

配置使用 dataclass 定义，支持类型检查和默认值：

```python
from src.configs import Navigation2DConfig

config = Navigation2DConfig()
print(config.state_dim)      # 21
print(config.k_level)        # 3
print(config.use_depth_encoder)  # True
```

## 主要特性

- **清晰的模块划分**: 每个模块职责单一
- **类型安全**: 使用类型注解
- **可测试**: 模块化设计便于单元测试
- **统一的设备管理**: 单例模式管理 CUDA/CPU
- **dataclass 配置**: 类型安全的配置系统

## 依赖

```
torch>=2.0
numpy
gymnasium
```

## License

MIT
