"""
HAC 测试脚本

架构: SAC + HAC + MPC (确定)

用法:
    python test.py                                     # 默认测试
    python test.py --render                            # 开启渲染
    python test.py --episodes 10                       # 运行10个episode
"""
import argparse
import torch
import gymnasium as gym
import numpy as np
import os

import asset  # 注册自定义环境
from configs import get_config
from HAC import HAC


def parse_args():
    parser = argparse.ArgumentParser(description='HAC Testing')
    parser.add_argument('--env', type=str, default='Navigation2DObstacle-v1',
                        help='Environment name')
    parser.add_argument('--k_level', type=int, default=None,
                        help='Number of hierarchy levels')
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
    if args.k_level is not None:
        config.k_level = args.k_level
    if args.seed is not None:
        config.random_seed = args.seed
    
    print("=" * 60)
    print("  HAC + SAC + MPC 测试")
    print("=" * 60)
    print(config)
    print("=" * 60)
    
    # 计算最大步数
    max_steps = config.H ** config.k_level
    print(f"Max steps: H^k_level = {config.H}^{config.k_level} = {max_steps}")
    
    # 创建环境
    if args.render:
        env = gym.make(config.env_name, render_mode="human", max_steps=max_steps)
    else:
        env = gym.make(config.env_name, max_steps=max_steps)
    
    # 随机种子
    if config.random_seed:
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
    
    # 创建 HAC 智能体
    agent = HAC(config, render=args.render)
    
    # 加载模型
    model_dir = config.get_save_directory()
    if args.model:
        model_name = args.model
    else:
        base_name = config.get_filename()
        best_name = base_name + '_best'
        
        best_path = os.path.join(model_dir, f"{best_name}_level_1_actor.pth")
        if os.path.exists(best_path):
            model_name = best_name
            print("Loading best model...")
        else:
            model_name = base_name
    
    print(f"Model: {model_dir}/{model_name}")
    agent.load(model_dir, model_name)
    
    # 测试统计
    total_rewards = []
    total_steps = []
    successes = 0
    
    # 测试循环
    for episode in range(1, args.episodes + 1):
        agent.reset_episode()
        state, _ = env.reset(seed=config.random_seed if episode == 1 else None)
        
        if hasattr(env.unwrapped, 'obstacles'):
            agent.set_obstacles(env.unwrapped.obstacles)
        
        last_state, done = agent.run_HAC(
            env, config.k_level - 1, state, config.goal_state, is_subgoal_test=True
        )
        
        success = agent.check_goal(last_state, config.goal_state, config.goal_threshold)
        if success:
            successes += 1
        
        total_rewards.append(agent.reward)
        total_steps.append(agent.timestep)
        
        status = "✓" if success else "✗"
        print(f"Episode: {episode}\t Reward: {agent.reward:.2f}\t Steps: {agent.timestep}\t {status}")
    
    # 统计
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
