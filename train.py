"""
HAC 训练脚本

用法:
    python train.py                                    # 默认 DDPG
    python train.py --algorithm sac                    # 使用 SAC
    python train.py --env Pendulum-h-v1               # Pendulum
    python train.py --env MountainCarContinuous-h-v1 --k_level 3  # 3层
    python train.py --render                           # 开启渲染
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
    parser = argparse.ArgumentParser(description='HAC Training')
    parser.add_argument('--env', type=str, default='Navigation2DObstacle-v1',
                        help='Environment name')
    parser.add_argument('--algorithm', type=str, default='sac',
                        choices=['ddpg', 'sac'],
                        help='Base algorithm (ddpg or sac), overrides config')
    parser.add_argument('--k_level', type=int, default=None,
                        help='Number of hierarchy levels (overrides config)')
    parser.add_argument('--max_episodes', type=int, default=None,
                        help='Max training episodes (overrides config)')
    parser.add_argument('--render', action='store_true',
                        help='Enable rendering')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config)')
    return parser.parse_args()


def train(args):
    # 获取环境配置
    config = get_config(args.env)
    
    # 命令行参数覆盖配置
    if args.algorithm is not None:
        config.algorithm = args.algorithm
    if args.k_level is not None:
        config.k_level = args.k_level
    if args.max_episodes is not None:
        config.max_episodes = args.max_episodes
    if args.seed is not None:
        config.random_seed = args.seed
    
    print("=" * 60)
    print(f"HAC Training with {config.algorithm.upper()}")
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
    
    # 创建保存目录
    save_dir = config.get_save_directory()
    os.makedirs(save_dir, exist_ok=True)
    
    # TensorBoard 日志
    from torch.utils.tensorboard import SummaryWriter
    log_dir = f"runs/{config.env_name}_{config.algorithm}"
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs: {log_dir}")
    print(f"Run: tensorboard --logdir=runs")
    
    # 日志文件
    log_file = open("log.txt", "w+")
    
    # 训练循环
    solved_count = 0
    for episode in range(1, config.max_episodes + 1):
        agent.reset_episode()
        
        # 重置环境
        state, _ = env.reset(seed=config.random_seed if episode == 1 else None)
        
        # 设置障碍物信息给 MPC
        if hasattr(env.unwrapped, 'obstacles'):
            agent.set_obstacles(env.unwrapped.obstacles)
        
        # 运行 HAC
        last_state, done = agent.run_HAC(
            env, 
            config.k_level - 1, 
            state, 
            config.goal_state, 
            is_subgoal_test=False
        )
        
        # 检查是否解决
        solved = agent.check_goal(last_state, config.goal_state, config.goal_threshold)
        if solved:
            print("################ Solved! ################")
            agent.save(save_dir, config.get_filename() + '_solved')
            solved_count += 1
        
        # 更新策略
        agent.update(config.n_iter, config.batch_size)
        
        # ===== TensorBoard 记录 (每 100 个 episode) =====
        if episode % 100 == 0:
            # 如果有深度编码器，绘制相关性散点图
            if agent.use_depth_encoder and agent.algorithm == 'sac':
                visualize_encoder_to_tensorboard(writer, agent, env, config, episode)
            solved_count = 0
        
        # 记录日志
        log_file.write(f'{episode},{agent.reward}\n')
        log_file.flush()
        
        # 定期保存
        if episode % config.save_episode == 0:
            agent.save(save_dir, config.get_filename())
        
        print(f"Episode: {episode}\t Reward: {agent.reward:.2f}\t Steps: {agent.timestep}")
    
    writer.close()
    log_file.close()
    env.close()
    print("Training completed!")


def visualize_encoder_to_tensorboard(writer, agent, env, config, episode):
    """将各层编码器的 embedding 相关性散点图记录到 TensorBoard"""
    import torch
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    import io
    from PIL import Image
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 收集样本
    depth_samples = []
    min_depths = []
    front_depths = []
    
    state, _ = env.reset()
    for _ in range(500):
        depth = state[config.base_state_dim:]
        depth_samples.append(depth.copy())
        min_depths.append(depth.min())
        mid = len(depth) // 2
        front_depths.append(np.mean(depth[mid-1:mid+2]))
        
        action = env.action_space.sample()
        state, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            state, _ = env.reset()
    
    depth_samples = np.array(depth_samples)
    min_depths = np.array(min_depths)
    front_depths = np.array(front_depths)
    
    # 为每层编码器生成图 (Level 0 是 MPC，从 Level 1 开始)
    for level in range(1, agent.k_level):
        policy = agent.HAC[level]
        encoder = getattr(policy, 'depth_encoder', None)
        if encoder is None:
            continue
        
        # 计算 embeddings
        with torch.no_grad():
            depth_tensor = torch.FloatTensor(depth_samples).to(device)
            embeddings = encoder(depth_tensor).cpu().numpy()
        
        embed_dim = embeddings.shape[1]
        
        # 创建散点图
        fig, axes = plt.subplots(2, embed_dim, figsize=(4*embed_dim, 8))
        
        for i in range(embed_dim):
            # 上行: vs Min Depth
            axes[0, i].scatter(min_depths, embeddings[:, i], alpha=0.3, s=5, c='steelblue')
            corr_min = np.corrcoef(embeddings[:, i], min_depths)[0, 1]
            if np.isnan(corr_min):
                corr_min = 0.0
            axes[0, i].set_xlabel('Min Depth')
            axes[0, i].set_ylabel(f'Embed Dim {i}')
            axes[0, i].set_title(f'Dim {i} vs Min Depth (r={corr_min:.2f})')
            
            # 下行: vs Front Depth
            axes[1, i].scatter(front_depths, embeddings[:, i], alpha=0.3, s=5, c='steelblue')
            corr_front = np.corrcoef(embeddings[:, i], front_depths)[0, 1]
            if np.isnan(corr_front):
                corr_front = 0.0
            axes[1, i].set_xlabel('Front Depth')
            axes[1, i].set_ylabel(f'Embed Dim {i}')
            axes[1, i].set_title(f'Dim {i} vs Front Depth (r={corr_front:.2f})')
        
        plt.suptitle(f'Level {level} Encoder (Episode {episode})', fontsize=14)
        plt.tight_layout()
        
        # 转换为 TensorBoard 图片
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        image = Image.open(buf)
        image_array = np.array(image)
        plt.close(fig)
        
        # 记录到 TensorBoard (HWC -> CHW)
        image_tensor = torch.from_numpy(image_array[:, :, :3]).permute(2, 0, 1)
        writer.add_image(f'Encoder/Level_{level}_Correlation', image_tensor, episode)


if __name__ == '__main__':
    args = parse_args()
    train(args)
