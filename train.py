#!/usr/bin/env python3
"""
HAC 训练和推理入口脚本

用法:
    python train.py                     # 默认训练
    python train.py --e2e_episodes 300  # E2E预热300回合
    python train.py --render            # 开启渲染
    python train.py --test --load_dir ./preTrained/xxx  # 推理模式
    python train.py --test --render     # 推理模式 + 渲染
"""

from src.train import train, test, parse_args

if __name__ == '__main__':
    args = parse_args()
    
    if args.test:
        test(args)
    else:
        train(args)
