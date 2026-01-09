#!/usr/bin/env python3
"""
HAC + MPC 训练和推理入口

用法:
    训练: python train.py
    推理: python train.py --test --run_dir ./runs/xxx
"""

import argparse
import os
import sys
import json
from datetime import datetime

import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import asset
from src.configs import get_config
from src.agents import HACAgent
from src.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='HAC + MPC')
    parser.add_argument('--test', action='store_true', help='推理模式')
    parser.add_argument('--run_dir', type=str, default=None, help='加载目录')
    parser.add_argument('--model_name', type=str, default='best', help='模型名: best/final/ep{N}')
    parser.add_argument('--env', type=str, default='Navigation2DObstacle-v1')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_episodes', type=int, default=None)
    parser.add_argument('--test_episodes', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--runs_dir', type=str, default='./runs')
    return parser.parse_args()


def train(args):
    config = get_config(args.env)
    if args.max_episodes:
        config.max_episodes = args.max_episodes
    set_seed(args.seed)

    # 创建运行目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.runs_dir, f"{timestamp}")
    os.makedirs(os.path.join(run_dir, 'models'), exist_ok=True)
    
    # 创建TensorBoard writer
    tb_log_dir = os.path.join(run_dir, 'tensorboard')
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"  TensorBoard: {tb_log_dir}")

    # 保存配置
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump({'env': config.env_name, 'k_level': config.k_level, 
                   'H': config.H, 'seed': args.seed}, f, indent=2)

    print(f"{'='*50}\n  HAC Training\n  Run: {run_dir}\n  Env: {config.env_name}\n{'='*50}")

    # 创建环境和智能体
    render_mode = "human" if args.render else None
    env = gym.make(config.env_name, render_mode=render_mode, 
                   max_steps=config.H ** config.k_level)
    agent = HACAgent(config, render=args.render)

    best_reward = float('-inf')
    log_file = open(os.path.join(run_dir, "log.csv"), "w")
    log_file.write("episode,reward,steps,success\n")

    for ep in range(1, config.max_episodes + 1):
        agent.reset_episode()
        state, _ = env.reset(seed=args.seed if ep == 1 else None)
        goal = env.unwrapped.goal_pos if hasattr(env.unwrapped, 'goal_pos') else config.goal_state

        last_state, _ = agent.run_HAC(env, config.k_level - 1, state, goal, False)
        success = agent.check_goal(last_state, goal, config.goal_threshold)
        train_stats = agent.update(config.n_iter, config.batch_size)

        # TensorBoard 日志记录
        # 1. 记录每episode步数
        writer.add_scalar('Episode/Steps', agent.timestep, ep)
        writer.add_scalar('Episode/Reward', agent.reward, ep)
        writer.add_scalar('Episode/Success', int(success), ep)
        
        # 2. 记录各层critic loss和梯度
        for level_name, level_stats in train_stats.items():
            level_idx = level_name.split('_')[1]  # 提取层级数字
            
            # Critic Loss
            if 'critic_loss' in level_stats:
                writer.add_scalar(f'Loss/{level_name}/Critic', level_stats['critic_loss'], ep)
            
            # Actor Loss
            if 'actor_loss' in level_stats:
                writer.add_scalar(f'Loss/{level_name}/Actor', level_stats['actor_loss'], ep)
            
            # Alpha (熵系数)
            if 'alpha' in level_stats:
                writer.add_scalar(f'Alpha/{level_name}', level_stats['alpha'], ep)
            
            # Critic 梯度统计
            if 'critic_grad_norm' in level_stats:
                writer.add_scalar(f'Gradients/{level_name}/Critic_Norm', level_stats['critic_grad_norm'], ep)
            if 'critic_grad_max' in level_stats:
                writer.add_scalar(f'Gradients/{level_name}/Critic_Max', level_stats['critic_grad_max'], ep)
            if 'critic_grad_mean' in level_stats:
                writer.add_scalar(f'Gradients/{level_name}/Critic_Mean', level_stats['critic_grad_mean'], ep)
            
            # 各层网络梯度详情
            if 'critic_layer_grads' in level_stats:
                for layer_name, grad_info in level_stats['critic_layer_grads'].items():
                    writer.add_scalar(f'Gradients/{level_name}/Critic_Layers/{layer_name}_norm', grad_info['norm'], ep)
                    writer.add_scalar(f'Gradients/{level_name}/Critic_Layers/{layer_name}_mean', grad_info['mean'], ep)
                    writer.add_scalar(f'Gradients/{level_name}/Critic_Layers/{layer_name}_max', grad_info['max'], ep)

        # 日志
        log_file.write(f"{ep},{agent.reward},{agent.timestep},{int(success)}\n")
        log_file.flush()

        # 保存
        tag = ""
        if agent.reward > best_reward:
            best_reward = agent.reward
            agent.save(os.path.join(run_dir, 'models'), 'best')
            tag = " *"
        if ep % args.save_freq == 0:
            agent.save(os.path.join(run_dir, 'models'), f'ep{ep}')

        print(f"Ep:{ep} R:{agent.reward:.1f} Steps:{agent.timestep} "
              f"{'OK' if success else ''}{tag}")

    env.close()
    log_file.close()
    writer.close()
    agent.save(os.path.join(run_dir, 'models'), 'final')
    print(f"\nDone! Best: {best_reward:.1f}")
    print(f"TensorBoard logs saved to: {tb_log_dir}")
    print(f"Run 'tensorboard --logdir {tb_log_dir}' to view logs")


def test(args):
    run_dir = args.run_dir
    if not run_dir:
        # 找最新
        if os.path.exists(args.runs_dir):
            runs = sorted(os.listdir(args.runs_dir), reverse=True)
            run_dir = os.path.join(args.runs_dir, runs[0]) if runs else None
    if not run_dir or not os.path.exists(run_dir):
        print("No run directory found")
        return

    with open(os.path.join(run_dir, 'config.json')) as f:
        cfg = json.load(f)
    
    config = get_config(cfg['env'])
    config.k_level = cfg['k_level']
    set_seed(args.seed)

    render_mode = "human" if args.render else None
    env = gym.make(config.env_name, render_mode=render_mode)
    agent = HACAgent(config, render=args.render)
    agent.load(os.path.join(run_dir, 'models'), args.model_name)

    print(f"{'='*50}\n  HAC Inference\n  Run: {run_dir}\n{'='*50}")

    rewards, successes = [], []
    for ep in range(1, args.test_episodes + 1):
        state, _ = env.reset()
        agent.reset()
        goal = env.unwrapped.goal_pos if hasattr(env.unwrapped, 'goal_pos') else config.goal_state

        last_state, _ = agent.run_HAC_inference(env, agent.k_level - 1, state, goal)
        success = agent.check_goal(last_state, goal, config.goal_threshold)
        
        rewards.append(agent.reward)
        successes.append(success)
        print(f"  Ep {ep}: R={agent.reward:.1f} {'OK' if success else ''}")

    env.close()
    print(f"\nMean: {np.mean(rewards):.1f}, SR: {sum(successes)}/{len(successes)}")


if __name__ == '__main__':
    args = parse_args()
    test(args) if args.test else train(args)
