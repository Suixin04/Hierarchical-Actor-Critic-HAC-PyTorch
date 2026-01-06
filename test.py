"""
HAC 测试脚本

用法:
    python test.py                                     # 默认 DDPG
    python test.py --algorithm sac                     # 使用 SAC
    python test.py --env Pendulum-h-v1                # Pendulum
    python test.py --render                            # 开启渲染
    python test.py --episodes 10                       # 运行10个episode
"""
import argparse
import torch
import gymnasium as gym
import numpy as np

import asset  # 注册自定义环境
from configs import get_config
from HAC import HAC


def parse_args():
    parser = argparse.ArgumentParser(description='HAC Testing')
    parser.add_argument('--env', type=str, default='Navigation2DObstacle-v1',
                        help='Environment name')
    parser.add_argument('--algorithm', type=str, default='sac',
                        choices=['ddpg', 'sac'],
                        help='Base algorithm (ddpg or sac), overrides config')
    parser.add_argument('--k_level', type=int, default=None,
                        help='Number of hierarchy levels (overrides config)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of test episodes')
    parser.add_argument('--render', action='store_true',
                        help='Enable rendering')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--model', type=str, default=None,
                        help='Model filename (without _level_X suffix)')
    return parser.parse_args()


def test(args):
    # 获取环境配置
    config = get_config(args.env)
    
    # 命令行参数覆盖配置
    if args.algorithm is not None:
        config.algorithm = args.algorithm
    if args.k_level is not None:
        config.k_level = args.k_level
    if args.seed is not None:
        config.random_seed = args.seed
    
    print("=" * 60)
    print(f"HAC Testing with {config.algorithm.upper()}")
    print("=" * 60)
    print(config)
    print("=" * 60)
    
    # 计算 HAC 需要的最大步数: H^k_level
    max_steps = config.H ** config.k_level
    print(f"Max steps per episode: H^k_level = {config.H}^{config.k_level} = {max_steps}")
    
    # 创建环境
    if args.render:
        env = gym.make(config.env_name, render_mode="human", max_steps=max_steps)
    else:
        env = gym.make(config.env_name, max_steps=max_steps)
    
    # 设置随机种子
    if config.random_seed:
        print(f"Random Seed: {config.random_seed}")
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
    
    # 创建 HAC 智能体
    agent = HAC(config, render=args.render, algorithm=config.algorithm)
    
    # 加载模型 (默认优先加载 solved 版本)
    model_dir = config.get_save_directory()
    if args.model:
        model_name = args.model
    else:
        # 优先尝试 solved 版本
        solved_name = config.get_filename() + '_solved'
        import os
        solved_path = os.path.join(model_dir, f"{solved_name}_level_0_actor.pth")
        if os.path.exists(solved_path):
            model_name = solved_name
            print("Found solved model, loading it...")
        else:
            model_name = config.get_filename()
    
    print(f"Loading model from: {model_dir}{model_name}")
    agent.load(model_dir, model_name)
    
    # 测试统计
    total_rewards = []
    total_steps = []
    successes = 0
    
    # 测试循环
    for episode in range(1, args.episodes + 1):
        agent.reset_episode()
        
        # 重置环境
        state, _ = env.reset(seed=config.random_seed if episode == 1 else None)
        
        # 运行 HAC (测试模式: is_subgoal_test=True, 不添加探索噪声)
        last_state, done = agent.run_HAC(
            env, 
            config.k_level - 1, 
            state, 
            config.goal_state, 
            is_subgoal_test=True  # 测试时不添加噪声
        )
        
        # 检查是否成功
        success = agent.check_goal(last_state, config.goal_state, config.goal_threshold)
        if success:
            successes += 1
        
        total_rewards.append(agent.reward)
        total_steps.append(agent.timestep)
        
        status = "✓" if success else "✗"
        print(f"Episode: {episode}\t Reward: {agent.reward:.2f}\t Steps: {agent.timestep}\t {status}")
    
    # 打印统计信息
    print("=" * 60)
    print(f"Results over {args.episodes} episodes:")
    print(f"  Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"  Average Steps:  {np.mean(total_steps):.1f} ± {np.std(total_steps):.1f}")
    print(f"  Success Rate:   {successes}/{args.episodes} ({100*successes/args.episodes:.1f}%)")
    print("=" * 60)
    
    env.close()


if __name__ == '__main__':
    args = parse_args()
    test(args)
