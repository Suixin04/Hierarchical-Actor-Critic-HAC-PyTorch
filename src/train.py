"""
HAC 训练和推理脚本

用法:
    python train.py                     # 默认训练
    python train.py --e2e_episodes 300  # E2E预热300回合
    python train.py --render            # 开启渲染
    python train.py --test --load_dir ./preTrained/xxx  # 推理模式

训练阶段:
    Phase 1 (E2E 预热): 
      - Level 1: E2E 更新 Encoder, RL 更新 Actor
      - Level 2+: RL 更新
      
    Phase 2 (RL 微调):  
      - Level 1: Encoder 微调或冻结, RL 更新 Actor
      - Level 2+: RL 继续更新
"""

import argparse
import os
import sys

import numpy as np
import gymnasium as gym

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asset  # 注册自定义环境
from src.configs import get_config
from src.agents import HACAgent
from src.utils import set_seed
from src.evaluation.metrics import TrainingMetrics
from src.evaluation.visualize import plot_training_curves, plot_encoder_analysis


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='HAC Training and Inference')
    
    # 模式选择
    parser.add_argument('--test', action='store_true',
                        help='Test/inference mode (no training)')
    parser.add_argument('--load_dir', type=str, default=None,
                        help='Directory to load model from')
    parser.add_argument('--load_name', type=str, default=None,
                        help='Model name to load (default: HAC_{env}_best)')
    parser.add_argument('--test_episodes', type=int, default=10,
                        help='Number of test episodes')
    
    # 环境参数
    parser.add_argument('--env', type=str, default='Navigation2DObstacle-v1',
                        help='Environment name')
    parser.add_argument('--k_level', type=int, default=None,
                        help='Number of hierarchy levels')
    parser.add_argument('--max_episodes', type=int, default=None,
                        help='Max training episodes')
    parser.add_argument('--render', action='store_true',
                        help='Enable rendering')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    
    # E2E 训练参数
    parser.add_argument('--e2e_episodes', type=int, default=1000,
                        help='Episodes for E2E warmup phase')
    parser.add_argument('--e2e_freq', type=int, default=3,
                        help='E2E update frequency')
    parser.add_argument('--e2e_lr', type=float, default=3e-4,
                        help='E2E learning rate')
    parser.add_argument('--e2e_batch', type=int, default=128,
                        help='E2E batch size')
    parser.add_argument('--e2e_updates', type=int, default=10,
                        help='E2E gradient updates per call')
    
    return parser.parse_args()


def run_e2e_training(
    agent: HACAgent,
    env_goal: np.ndarray,
    batch_size: int = 64,
    lr: float = 3e-4,
    num_updates: int = 5
) -> float | None:
    """
    E2E 训练: 通过可微 MPC 更新 Encoder + Actor
    
    Returns:
        最后一次更新的距离损失，或 None
    """
    if agent.k_level < 2:
        return None
    
    buffer = agent.buffers[agent.k_level - 1]
    if buffer is None or len(buffer) < batch_size:
        return None
    
    losses = []
    for _ in range(num_updates):
        states = buffer.sample_states(batch_size)
        
        try:
            loss_history = agent.train_end_to_end_batch(
                states=states,
                final_goal=np.array(env_goal),
                num_steps=5,
                lr=lr
            )
            if loss_history and 'total' in loss_history[0]:
                losses.append(loss_history[0]['total'])
        except Exception as e:
            print(f"E2E error: {e}")
            continue
    
    return losses[-1] if losses else None


def train(args: argparse.Namespace) -> None:
    """训练主函数"""
    # 获取配置
    config = get_config(args.env)
    
    # 命令行覆盖
    if args.k_level is not None:
        config.k_level = args.k_level
    if args.max_episodes is not None:
        config.max_episodes = args.max_episodes
    if args.seed is not None:
        config.random_seed = args.seed
    
    # 打印配置
    _print_header(config, args)
    
    # 设置随机种子
    if config.random_seed:
        set_seed(config.random_seed)
    
    # 创建环境
    max_steps = config.H ** config.k_level
    render_mode = "human" if args.render else None
    env = gym.make(config.env_name, render_mode=render_mode, max_steps=max_steps)
    
    # 创建智能体
    agent = HACAgent(config, render=args.render)
    
    # 保存目录
    save_dir = config.get_save_directory()
    os.makedirs(save_dir, exist_ok=True)
    
    # 训练指标收集器
    metrics = TrainingMetrics(window_size=100)
    
    # 训练循环
    current_phase = 1
    best_reward = float('-inf')
    
    # 用于编码器分析的深度样本
    depth_samples_for_analysis = []
    
    with open(os.path.join(save_dir, "log.txt"), "w") as log_file:
        log_file.write("episode,reward,phase,success,steps\n")
        
        for episode in range(1, config.max_episodes + 1):
            # 阶段切换
            if current_phase == 1 and episode > args.e2e_episodes:
                _switch_phase(agent, config, save_dir)
                current_phase = 2
                metrics.log_phase_switch(episode)
            
            # 运行 episode
            agent.reset_episode()
            state, _ = env.reset(seed=config.random_seed if episode == 1 else None)
            
            # 收集深度样本用于后续分析
            if config.use_depth_encoder and len(depth_samples_for_analysis) < 2000:
                depth = state[config.base_state_dim:]
                depth_samples_for_analysis.append(depth.copy())
            
            if hasattr(env.unwrapped, 'obstacles'):
                agent.set_obstacles(env.unwrapped.obstacles)
            
            env_goal = (
                env.unwrapped.goal_pos 
                if hasattr(env.unwrapped, 'goal_pos') 
                else config.goal_state
            )
            
            last_state, done = agent.run_HAC(
                env, config.k_level - 1, state, env_goal, is_subgoal_test=False
            )
            
            solved = agent.check_goal(last_state, env_goal, config.goal_threshold)
            if solved:
                print("################ Solved! ################")
            
            # Phase 1: E2E 预热
            if current_phase == 1:
                agent.update(config.n_iter, config.batch_size)
                
                e2e_loss = None
                if episode % args.e2e_freq == 0:
                    e2e_loss = run_e2e_training(
                        agent, env_goal,
                        batch_size=args.e2e_batch,
                        lr=args.e2e_lr,
                        num_updates=args.e2e_updates
                    )
                
                phase_str = "[P1-E2E]"
                e2e_str = f" Dist:{e2e_loss:.2f}" if e2e_loss else ""
            
            # Phase 2: RL 微调
            else:
                agent.update(config.n_iter, config.batch_size)
                phase_str = "[P2-RL]"
                e2e_str = ""
                e2e_loss = None
            
            # 记录指标
            metrics.log_episode(
                episode=episode,
                reward=agent.reward,
                steps=agent.timestep,
                success=solved,
                phase=current_phase,
                e2e_loss=e2e_loss
            )
            
            # 日志
            log_file.write(f'{episode},{agent.reward},{current_phase},{int(solved)},{agent.timestep}\n')
            log_file.flush()
            
            # 保存逻辑
            saved_str = ""
            if agent.reward > best_reward:
                best_reward = agent.reward
                agent.save(save_dir, config.get_filename() + '_best', verbose=False)
                saved_str = " [BEST]"
            
            if episode % config.save_episode == 0:
                agent.save(save_dir, f"{config.get_filename()}_ep{episode}", verbose=False)
                saved_str += f" [SAVE]"
            
            # 定期生成可视化
            if episode % 500 == 0 or episode == config.max_episodes:
                _save_visualizations(
                    metrics, agent, config, save_dir, 
                    depth_samples_for_analysis, episode
                )
            
            print(f"{phase_str} Ep:{episode} R:{agent.reward:.2f} "
                  f"Steps:{agent.timestep} SR:{metrics.get_success_rate()*100:.1f}%"
                  f"{e2e_str}{saved_str}")
    
    env.close()
    
    # 最终可视化
    _save_visualizations(
        metrics, agent, config, save_dir, 
        depth_samples_for_analysis, config.max_episodes
    )
    
    # 保存指标
    metrics.save(os.path.join(save_dir, 'metrics.json'))
    
    _print_summary(metrics)


def _print_header(config, args) -> None:
    """打印训练头信息"""
    print("=" * 60)
    print("  HAC + SAC + MPC Training")
    print("=" * 60)
    print(f"  Phase 1 (E2E Warmup): Episodes 1-{args.e2e_episodes}")
    print(f"  Phase 2 (RL Finetune): Episodes {args.e2e_episodes+1}-{config.max_episodes}")
    print("=" * 60)
    print(config)
    print("=" * 60)


def _switch_phase(agent, config, save_dir) -> None:
    """切换到 Phase 2"""
    print("\n" + "=" * 60)
    print("  Phase Switch: E2E Warmup -> RL Finetune")
    print("=" * 60)
    agent.enable_level1_encoder_finetune()
    agent.save(save_dir, config.get_filename() + '_phase1')
    print("=" * 60 + "\n")


def _save_visualizations(
    metrics: TrainingMetrics,
    agent: HACAgent,
    config,
    save_dir: str,
    depth_samples: list,
    episode: int
) -> None:
    """保存可视化结果"""
    fig_dir = os.path.join(save_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    # 训练曲线
    plot_training_curves(
        metrics,
        os.path.join(fig_dir, f'training_curves_ep{episode}.png'),
        title=f'Training Progress (Episode {episode})'
    )
    
    # 编码器分析 (如果使用深度编码器)
    if config.use_depth_encoder and len(depth_samples) >= 500:
        try:
            encoder = agent.policies[1].actor.depth_encoder
            if encoder is None:
                print("  Warning: No depth encoder found")
                return
            depth_arr = np.array(depth_samples)
            encoder_metrics = plot_encoder_analysis(
                encoder,
                depth_arr,
                os.path.join(fig_dir, f'encoder_analysis_ep{episode}.png'),
                depth_max_range=config.depth_max_range,
                title=f'Encoder Analysis (Episode {episode})'
            )
            print(f"  Encoder metrics: var={encoder_metrics['embedding_variance']:.4f}, "
                  f"max_corr={encoder_metrics['max_correlation']:.3f}")
        except Exception as e:
            print(f"  Warning: Encoder analysis failed: {e}")


def _print_summary(metrics: TrainingMetrics) -> None:
    """打印训练总结"""
    summary = metrics.get_summary()
    
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  Total Episodes: {summary['total_episodes']}")
    print(f"  Best Reward: {summary['best_reward']:.2f} @ Episode {summary['best_episode']}")
    print(f"  Final Mean Reward: {summary['final_100_mean_reward']:.2f}")
    print(f"  Final Success Rate: {summary['final_100_success_rate']*100:.1f}%")
    print(f"  Phase 1 Mean Reward: {summary['phase1_mean_reward']:.2f}")
    print(f"  Phase 2 Mean Reward: {summary['phase2_mean_reward']:.2f}")
    print("=" * 60)


def test(args: argparse.Namespace) -> None:
    """
    推理/测试模式
    
    加载预训练模型并运行评估回合，不进行任何训练更新。
    """
    # 获取配置
    config = get_config(args.env)
    
    # 设置随机种子
    if args.seed is not None:
        set_seed(args.seed)
    elif config.random_seed is not None:
        set_seed(config.random_seed)
    
    # 创建环境
    render_mode = "human" if args.render else None
    env = gym.make(config.env_name, render_mode=render_mode)
    
    # 构建模型加载路径
    if args.load_dir is None:
        load_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'preTrained'
        )
    else:
        load_dir = args.load_dir
    
    if args.load_name is None:
        load_name = config.get_filename() + '_best'
    else:
        load_name = args.load_name
    
    # 创建 Agent
    print("=" * 60)
    print("  HAC Inference Mode")
    print("=" * 60)
    print(f"  Environment: {config.env_name}")
    print(f"  Load Dir: {load_dir}")
    print(f"  Model Name: {load_name}")
    print(f"  Test Episodes: {args.test_episodes}")
    print(f"  Render: {args.render}")
    print("=" * 60)
    
    agent = HACAgent(config=config, render=args.render)
    
    # 加载模型
    try:
        agent.load(load_dir, load_name)
        print(f"Successfully loaded model from {load_dir}/{load_name}")
    except FileNotFoundError as e:
        print(f"Error: Could not find model file: {e}")
        print("\nAvailable models in directory:")
        if os.path.exists(load_dir):
            for f in os.listdir(load_dir):
                if f.endswith('.pth'):
                    print(f"  - {f}")
        else:
            print(f"  Directory does not exist: {load_dir}")
        env.close()
        return
    
    # 运行测试回合
    rewards = []
    steps_list = []
    success_count = 0
    
    # 计算最大步数
    max_steps = config.H ** config.k_level
    
    print("\n" + "-" * 40)
    print("Running test episodes...")
    print("-" * 40)
    
    for episode in range(1, args.test_episodes + 1):
        state, _ = env.reset()
        agent.reset()
        
        # 获取并设置目标
        env_goal = (
            env.unwrapped.goal_pos 
            if hasattr(env.unwrapped, 'goal_pos') 
            else config.goal_state
        )
        agent.set_goal(env_goal)
        
        done = False
        total_reward = 0.0
        total_steps = 0
        
        while not done and total_steps < max_steps:
            action = agent.act(state, deterministic=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            total_steps += 1
            state = next_state
        
        rewards.append(total_reward)
        steps_list.append(total_steps)
        
        # 检查是否成功
        success = agent.check_goal(state, env_goal, config.goal_threshold)
        if success:
            success_count += 1
            success_str = " [SUCCESS]"
        else:
            success_str = ""
        
        print(f"  Episode {episode}: Reward={total_reward:.2f}, "
              f"Steps={total_steps}{success_str}")
    
    # 打印统计结果
    print("\n" + "=" * 60)
    print("  Test Results Summary")
    print("=" * 60)
    print(f"  Episodes: {args.test_episodes}")
    print(f"  Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Max Reward: {np.max(rewards):.2f}")
    print(f"  Min Reward: {np.min(rewards):.2f}")
    print(f"  Mean Steps: {np.mean(steps_list):.1f}")
    print(f"  Success Rate: {success_count}/{args.test_episodes} "
          f"({100*success_count/args.test_episodes:.1f}%)")
    print("=" * 60)
    
    env.close()


if __name__ == '__main__':
    args = parse_args()
    
    if args.test:
        test(args)
    else:
        train(args)
