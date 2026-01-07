"""
HAC 分阶段训练脚本 (清理版)

用法:
    python train.py --env Navigation2DObstacle-v1       # 默认分阶段训练
    python train.py --e2e_episodes 300                  # E2E预热300回合
    python train.py --render                            # 开启渲染
    
训练阶段:
    Phase 1 (E2E预热): 
      - Level 1: E2E 更新 Encoder (特权学习避障), RL 更新 Actor (encoder detach)
      - Level 2+: RL 更新 Encoder + Actor (高层规划)
    Phase 2 (RL微调):  
      - Level 1: Encoder 冻结, RL 更新 Actor
      - Level 2+: RL 继续更新 Encoder + Actor
    
梯度流设计:
    - Level 1 Encoder ← E2E only (通过特权学习学会避障表征)
    - Level 2+ Encoder ← RL (学习高层规划所需的深度表征)

架构改进:
    - Level 1: 使用 MPC 预测可达性 (替代传统 Subgoal Testing)
      * MPC 直接预测 H 步后能到达的位置
      * 如果预测不可达，立即惩罚，无需实际执行
      * 预测的最终位置可用于 Hindsight Action
    - Level 2+: 保留传统 Subgoal Testing 机制
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
    parser = argparse.ArgumentParser(description='HAC Staged Training')
    parser.add_argument('--env', type=str, default='Navigation2DObstacle-v1',
                        help='Environment name')
    parser.add_argument('--algorithm', type=str, default='sac',
                        choices=['ddpg', 'sac'],
                        help='High-level algorithm')
    parser.add_argument('--k_level', type=int, default=None,
                        help='Number of hierarchy levels')
    parser.add_argument('--max_episodes', type=int, default=None,
                        help='Max training episodes')
    parser.add_argument('--render', action='store_true',
                        help='Enable rendering')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    
    # E2E 预热参数
    parser.add_argument('--e2e_episodes', type=int, default=100,
                        help='Number of episodes for E2E warmup phase')
    parser.add_argument('--e2e_freq', type=int, default=3,
                        help='E2E update frequency (every N episodes)')
    parser.add_argument('--e2e_lr', type=float, default=3e-4,
                        help='Learning rate for E2E training')
    parser.add_argument('--e2e_batch', type=int, default=128,
                        help='Batch size for E2E training')
    parser.add_argument('--e2e_updates', type=int, default=10,
                        help='Number of gradient updates per E2E call')
    
    return parser.parse_args()


def run_e2e_training(agent, config, batch_size=64, lr=3e-4, num_updates=5):
    """
    E2E 训练：通过可微 MPC 更新 Encoder + Actor
    
    梯度流: TaskLoss → MPC轨迹 → 子目标 → Actor → Encoder
    
    Returns:
        final_dist: 到目标的平均距离 (归一化指标)
    """
    if not hasattr(agent, 'mpc') or agent.mpc is None:
        return None
    
    if agent.k_level < 2:
        return None
    
    upper_level = agent.k_level - 1
    buffer = agent.replay_buffer[upper_level]
    if buffer is None or len(buffer) < batch_size:
        return None
    
    losses = []
    for _ in range(num_updates):
        if hasattr(buffer, 'sample_states'):
            states = buffer.sample_states(batch_size)
        else:
            batch = buffer.sample(batch_size)
            states = batch[0]
        
        try:
            loss_history = agent.train_end_to_end_batch(
                states=states,
                final_goal=np.array(config.goal_state),
                num_steps=5,
                lr=lr
            )
            if loss_history and 'total' in loss_history[0]:
                losses.append(loss_history[0]['total'])
        except Exception as e:
            print(f"E2E error: {e}")
            continue
    
    return losses[-1] if losses else None


def train(args):
    """分阶段训练主函数"""
    # 获取配置
    config = get_config(args.env)
    
    # 命令行覆盖
    if args.algorithm:
        config.algorithm = args.algorithm
    if args.k_level:
        config.k_level = args.k_level
    if args.max_episodes:
        config.max_episodes = args.max_episodes
    if args.seed:
        config.random_seed = args.seed
    
    print("=" * 60)
    print("  HAC STAGED TRAINING")
    print("=" * 60)
    print(f"  Phase 1 (E2E Warmup): Episodes 1-{args.e2e_episodes}")
    print(f"    - Level 1 Encoder: E2E (特权学习避障)")
    print(f"    - Level 2+ Encoder: RL (高层规划)")
    print(f"  Phase 2 (RL Finetune): Episodes {args.e2e_episodes+1}-{config.max_episodes}")
    print(f"    - Level 1 Encoder: Frozen")
    print(f"    - Level 2+ Encoder: RL (继续更新)")
    print("=" * 60)
    print(config)
    print("=" * 60)
    
    # 环境
    max_steps = config.H ** config.k_level
    if args.render:
        env = gym.make(config.env_name, render_mode="human", max_steps=max_steps)
    else:
        env = gym.make(config.env_name, max_steps=max_steps)
    
    # 随机种子
    if config.random_seed:
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
    
    # HAC 智能体
    agent = HAC(config, render=args.render, algorithm=config.algorithm)
    
    # 保存目录
    save_dir = config.get_save_directory()
    os.makedirs(save_dir, exist_ok=True)
    
    # TensorBoard
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(f"runs/{config.env_name}_staged")
    
    log_file = open("log.txt", "w+")
    
    # ==================== 训练循环 ====================
    current_phase = 1
    e2e_losses = []
    best_reward = float('-inf')
    
    for episode in range(1, config.max_episodes + 1):
        # ========== 阶段切换 ==========
        if current_phase == 1 and episode > args.e2e_episodes:
            print("\n" + "=" * 60)
            print("  PHASE TRANSITION: E2E Warmup -> RL Finetune")
            print("  Freezing all encoders...")
            print("=" * 60 + "\n")
            
            agent.freeze_all_encoders()
            current_phase = 2
            agent.save(save_dir, config.get_filename() + '_phase1')
        
        # 运行 episode
        agent.reset_episode()
        state, _ = env.reset(seed=config.random_seed if episode == 1 else None)
        
        if hasattr(env.unwrapped, 'obstacles'):
            agent.set_obstacles(env.unwrapped.obstacles)
        
        last_state, done = agent.run_HAC(
            env, config.k_level - 1, state, config.goal_state, is_subgoal_test=False
        )
        
        solved = agent.check_goal(last_state, config.goal_state, config.goal_threshold)
        if solved:
            print("################ Solved! ################")
        
        # ========== Phase 1: E2E 预热 ==========
        if current_phase == 1:
            # RL 更新策略 (encoder 输出被 detach，不会被 RL 更新)
            agent.update(config.n_iter, config.batch_size)
            
            # E2E 更新 Encoder + Actor
            e2e_loss = None
            if episode % args.e2e_freq == 0:
                e2e_loss = run_e2e_training(
                    agent, config,
                    batch_size=args.e2e_batch,
                    lr=args.e2e_lr,
                    num_updates=args.e2e_updates
                )
                if e2e_loss is not None:
                    e2e_losses.append(e2e_loss)
                    writer.add_scalar('E2E/Distance', e2e_loss, episode)
            
            phase_str = "[P1-E2E]"
            e2e_str = f" Dist:{e2e_loss:.2f}" if e2e_loss else ""
        
        # ========== Phase 2: RL 微调 ==========
        else:
            # 纯 RL 更新 (encoder 已冻结)
            agent.update(config.n_iter, config.batch_size)
            phase_str = "[P2-RL]"
            e2e_str = ""
        
        # 记录
        writer.add_scalar('Reward/Episode', agent.reward, episode)
        writer.add_scalar('Steps/Episode', agent.timestep, episode)
        writer.add_scalar('Phase', current_phase, episode)
        
        log_file.write(f'{episode},{agent.reward},{current_phase}\n')
        log_file.flush()
        
        # 保存
        if agent.reward > best_reward:
            best_reward = agent.reward
            agent.save(save_dir, config.get_filename() + '_best')
        
        if episode % config.save_episode == 0:
            agent.save(save_dir, config.get_filename())
        
        print(f"{phase_str} Ep:{episode} R:{agent.reward:.2f} Steps:{agent.timestep}{e2e_str}")
    
    writer.close()
    log_file.close()
    env.close()
    
    print("\n" + "=" * 60)
    print("  Training Completed!")
    print(f"  Best Reward: {best_reward:.2f}")
    print(f"  E2E Updates: {len(e2e_losses)}")
    if e2e_losses:
        print(f"  Final E2E Dist: {e2e_losses[-1]:.2f}")
    print("=" * 60)


if __name__ == '__main__':
    args = parse_args()
    train(args)
