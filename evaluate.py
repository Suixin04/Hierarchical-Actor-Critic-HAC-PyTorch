#!/usr/bin/env python3
"""
训练验证脚本

用于验证训练效果和编码器特征提取能力

用法:
    python evaluate.py --load_dir preTrained/xxx     # 评估模型
    python evaluate.py --analyze_encoder             # 分析编码器
    python evaluate.py --ablation                    # 运行消融实验
    python evaluate.py --visualize_trajectories      # 可视化轨迹
"""

import argparse
import os
import sys
import numpy as np
import gymnasium as gym

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import asset
from src.configs import get_config
from src.agents import HACAgent
from src.utils import set_seed
from src.evaluation.metrics import TrainingMetrics
from src.evaluation.visualize import (
    plot_training_curves, 
    plot_encoder_analysis,
    visualize_trajectories,
    plot_reward_comparison
)


def parse_args():
    parser = argparse.ArgumentParser(description='HAC Evaluation and Analysis')
    
    # 基础参数
    parser.add_argument('--env', type=str, default='Navigation2DObstacle-v1')
    parser.add_argument('--load_dir', type=str, default=None)
    parser.add_argument('--load_name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--save_dir', type=str, default='evaluation_results')
    
    # 评估参数
    parser.add_argument('--test_episodes', type=int, default=100,
                        help='Number of test episodes')
    
    # 分析模式
    parser.add_argument('--analyze_encoder', action='store_true',
                        help='Analyze encoder feature extraction')
    parser.add_argument('--visualize_trajectories', action='store_true',
                        help='Visualize agent trajectories')
    parser.add_argument('--ablation', action='store_true',
                        help='Run ablation study')
    parser.add_argument('--analyze_metrics', type=str, default=None,
                        help='Path to metrics.json file to analyze')
    
    return parser.parse_args()


def evaluate_model(args):
    """评估模型性能"""
    config = get_config(args.env)
    set_seed(args.seed)
    
    # 创建环境
    render_mode = "human" if args.render else None
    env = gym.make(config.env_name, render_mode=render_mode)
    
    # 加载模型
    load_dir = args.load_dir or config.get_save_directory()
    load_name = args.load_name or config.get_filename() + '_best'
    
    print(f"\n{'='*60}")
    print("  Model Evaluation")
    print(f"{'='*60}")
    print(f"  Model: {load_dir}/{load_name}")
    print(f"  Test Episodes: {args.test_episodes}")
    print(f"{'='*60}\n")
    
    agent = HACAgent(config=config, render=args.render)
    
    try:
        agent.load(load_dir, load_name)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None, None
    
    # 运行评估
    rewards = []
    steps_list = []
    successes = []
    trajectories = []
    goals = []
    
    # 计算最大步数
    max_steps = config.H ** config.k_level
    
    for ep in range(1, args.test_episodes + 1):
        state, _ = env.reset()
        agent.reset()
        
        env_goal = env.unwrapped.goal_pos if hasattr(env.unwrapped, 'goal_pos') else config.goal_state
        goals.append(env_goal.copy())
        
        # 设置智能体目标
        agent.set_goal(env_goal)
        
        trajectory = [state[:2].copy()]
        total_reward = 0.0
        total_steps = 0
        done = False
        
        while not done and total_steps < max_steps:
            action = agent.act(state, deterministic=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 渲染
            if args.render:
                env.render()
            
            trajectory.append(next_state[:2].copy())
            total_reward += reward
            total_steps += 1
            state = next_state
        
        trajectories.append(np.array(trajectory))
        rewards.append(total_reward)
        steps_list.append(total_steps)
        
        success = agent.check_goal(state, env_goal, config.goal_threshold)
        successes.append(success)
        
        status = "✓" if success else "✗"
        print(f"  Episode {ep:3d}: R={total_reward:7.2f}, Steps={total_steps:4d} {status}")
    
    env.close()
    
    # 统计结果
    results = {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'max_reward': np.max(rewards),
        'min_reward': np.min(rewards),
        'success_rate': np.mean(successes),
        'mean_steps': np.mean(steps_list),
        'trajectories': trajectories,
        'goals': goals,
    }
    
    print(f"\n{'='*60}")
    print("  Evaluation Results")
    print(f"{'='*60}")
    print(f"  Mean Reward:  {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Max Reward:   {results['max_reward']:.2f}")
    print(f"  Min Reward:   {results['min_reward']:.2f}")
    print(f"  Success Rate: {results['success_rate']*100:.1f}%")
    print(f"  Mean Steps:   {results['mean_steps']:.1f}")
    print(f"{'='*60}\n")
    
    return results, agent, config


def analyze_encoder(args):
    """分析编码器特征提取能力"""
    config = get_config(args.env)
    
    if not config.use_depth_encoder:
        print("Error: This environment doesn't use depth encoder.")
        return
    
    set_seed(args.seed)
    
    # 创建环境并收集样本
    env = gym.make(config.env_name)
    
    print(f"\n{'='*60}")
    print("  Encoder Analysis")
    print(f"{'='*60}")
    print("  Collecting depth samples...")
    
    depth_samples = []
    state, _ = env.reset()
    
    for _ in range(2000):
        depth = state[config.base_state_dim:]
        depth_samples.append(depth.copy())
        
        action = env.action_space.sample()
        state, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            state, _ = env.reset()
    
    depth_samples = np.array(depth_samples)
    print(f"  Collected {len(depth_samples)} samples")
    
    # 加载模型
    load_dir = args.load_dir or config.get_save_directory()
    load_name = args.load_name or config.get_filename() + '_best'
    
    agent = HACAgent(config=config, render=False)
    
    try:
        agent.load(load_dir, load_name)
        print(f"  Loaded model from {load_dir}/{load_name}")
    except FileNotFoundError:
        print("  Warning: No pretrained model found, using random initialization")
    
    env.close()
    
    # 分析编码器
    os.makedirs(args.save_dir, exist_ok=True)
    
    try:
        # 获取编码器 (在 GaussianActor 中属性名是 depth_encoder)
        encoder = agent.policies[1].actor.depth_encoder
        if encoder is None:
            print("  Error: No depth encoder found in the model")
            return
            
        metrics = plot_encoder_analysis(
            encoder,
            depth_samples,
            os.path.join(args.save_dir, 'encoder_analysis.png'),
            depth_max_range=config.depth_max_range,
            title='Depth Encoder Feature Analysis'
        )
        
        print(f"\n  Encoder Metrics:")
        print(f"    Embedding Variance: {metrics['embedding_variance']:.4f}")
        print(f"    PCA Explained Var:  {metrics['pca_explained_variance']*100:.1f}%")
        print(f"    Max Correlation:    {metrics['max_correlation']:.3f}")
        print(f"    Mean Max Corr:      {metrics['mean_max_correlation']:.3f}")
        
        # 编码器有效性判断
        print(f"\n  Encoder Effectiveness:")
        if metrics['embedding_variance'] < 0.01:
            print("    ⚠️  Low variance - encoder may be collapsed")
        else:
            print("    ✓  Variance OK")
        
        if metrics['max_correlation'] < 0.3:
            print("    ⚠️  Low correlation - encoder may not capture depth features")
        else:
            print("    ✓  Correlation OK")
        
        print(f"\n  Saved analysis to {args.save_dir}/encoder_analysis.png")
        
    except Exception as e:
        print(f"  Error analyzing encoder: {e}")


def visualize_agent_trajectories(args):
    """可视化智能体轨迹"""
    results, agent, config = evaluate_model(args)
    
    if results is None:
        return
    
    # 获取障碍物信息
    env = gym.make(config.env_name)
    env.reset()
    obstacles = env.unwrapped.obstacles if hasattr(env.unwrapped, 'obstacles') else None
    env.close()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 可视化前 10 条轨迹
    n_traj = min(10, len(results['trajectories']))
    visualize_trajectories(
        trajectories=results['trajectories'][:n_traj],
        goals=results['goals'][:n_traj],
        obstacles=obstacles,
        world_size=config.world_size,
        save_path=os.path.join(args.save_dir, 'trajectories.png'),
        title=f'Agent Trajectories (n={n_traj})'
    )
    
    print(f"  Saved trajectories to {args.save_dir}/trajectories.png")


def analyze_training_metrics(args):
    """分析训练指标"""
    if args.analyze_metrics is None:
        # 尝试默认路径
        config = get_config(args.env)
        metrics_path = os.path.join(config.get_save_directory(), 'metrics.json')
    else:
        metrics_path = args.analyze_metrics
    
    if not os.path.exists(metrics_path):
        print(f"Error: Metrics file not found at {metrics_path}")
        return
    
    print(f"\n{'='*60}")
    print("  Training Metrics Analysis")
    print(f"{'='*60}")
    print(f"  Loading: {metrics_path}")
    
    metrics = TrainingMetrics.load(metrics_path)
    summary = metrics.get_summary()
    
    print(f"\n  Summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.4f}")
        else:
            print(f"    {key}: {value}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    plot_training_curves(
        metrics,
        os.path.join(args.save_dir, 'training_curves.png'),
        title='Training Progress'
    )
    
    print(f"\n  Saved curves to {args.save_dir}/training_curves.png")


def run_ablation_study(args):
    """运行消融实验提示"""
    print(f"\n{'='*60}")
    print("  Ablation Study Guide")
    print(f"{'='*60}")
    print("""
  要验证各组件的有效性，建议进行以下消融实验:

  1. 编码器消融:
     - 对比: 有深度编码器 vs 直接使用原始深度
     - 命令: python train.py --env Navigation2DObstacle-v1
             修改配置: use_depth_encoder = False

  2. E2E 预热消融:
     - 对比: 不同 E2E 预热回合数 (0, 500, 1000, 2000)
     - 命令: python train.py --e2e_episodes 0
             python train.py --e2e_episodes 500
             python train.py --e2e_episodes 1000

  3. MPC 可达性预测消融:
     - 对比: 使用 vs 不使用 MPC 可达性检查
     - 修改 hac.py: 注释掉 predict_reachability 相关代码

  4. 层级数量消融:
     - 对比: 2层 vs 3层 vs 4层
     - 命令: python train.py --k_level 2
             python train.py --k_level 3
             python train.py --k_level 4

  实验后使用以下命令对比结果:
     python evaluate.py --analyze_metrics preTrained/exp1/metrics.json
     
  或收集多组 metrics.json 使用 plot_reward_comparison 对比
""")


def main():
    args = parse_args()
    
    if args.analyze_metrics:
        analyze_training_metrics(args)
    elif args.analyze_encoder:
        analyze_encoder(args)
    elif args.visualize_trajectories:
        visualize_agent_trajectories(args)
    elif args.ablation:
        run_ablation_study(args)
    else:
        # 默认: 评估模型
        evaluate_model(args)


if __name__ == '__main__':
    main()
