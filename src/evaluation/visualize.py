"""
训练可视化工具

提供训练曲线、编码器分析、轨迹可视化等功能
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import os

if TYPE_CHECKING:
    from src.evaluation.metrics import TrainingMetrics


def plot_training_curves(
    metrics: 'TrainingMetrics',
    save_path: str,
    title: str = "Training Curves"
) -> None:
    """
    绘制训练曲线
    
    包括：奖励曲线、成功率、E2E损失、步数
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    episodes = [e.episode for e in metrics.episodes]
    rewards = [e.reward for e in metrics.episodes]
    steps = [e.steps for e in metrics.episodes]
    successes = [float(e.success) for e in metrics.episodes]
    phases = [e.phase for e in metrics.episodes]
    
    # 1. 奖励曲线
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(episodes, rewards, alpha=0.3, color='steelblue', label='Episode Reward')
    
    # 滑动平均
    window = 50
    if len(rewards) >= window:
        smooth_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(episodes[window-1:], smooth_rewards, color='darkblue', 
                linewidth=2, label=f'{window}-Episode Avg')
    
    # 标记阶段切换
    if metrics.phase_switch_episode is not None:
        ax1.axvline(x=metrics.phase_switch_episode, color='red', linestyle='--',
                   label=f'Phase Switch @{metrics.phase_switch_episode}')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Episode Rewards')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # 2. 成功率
    ax2 = fig.add_subplot(gs[0, 1])
    window_sr = 100
    if len(successes) >= window_sr:
        success_rate = np.convolve(successes, np.ones(window_sr)/window_sr, mode='valid')
        ax2.plot(episodes[window_sr-1:], success_rate * 100, color='green', linewidth=2)
    else:
        cumsum = np.cumsum(successes)
        success_rate = cumsum / (np.arange(len(successes)) + 1)
        ax2.plot(episodes, success_rate * 100, color='green', linewidth=2)
    
    if metrics.phase_switch_episode is not None:
        ax2.axvline(x=metrics.phase_switch_episode, color='red', linestyle='--')
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title(f'Success Rate ({window_sr}-Episode Window)')
    ax2.set_ylim([0, 105])
    ax2.grid(True, alpha=0.3)
    
    # 3. E2E 损失
    ax3 = fig.add_subplot(gs[1, 0])
    e2e_episodes = [e.episode for e in metrics.episodes if e.e2e_loss is not None]
    e2e_losses = [e.e2e_loss for e in metrics.episodes if e.e2e_loss is not None]
    
    if e2e_losses:
        ax3.plot(e2e_episodes, e2e_losses, color='orange', alpha=0.5)
        if len(e2e_losses) >= 20:
            smooth_e2e = np.convolve(e2e_losses, np.ones(20)/20, mode='valid')
            ax3.plot(e2e_episodes[19:], smooth_e2e, color='darkorange', linewidth=2)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('E2E Distance Loss')
        ax3.set_title('End-to-End Training Loss (Goal Distance)')
    else:
        ax3.text(0.5, 0.5, 'No E2E Loss Data', ha='center', va='center', 
                transform=ax3.transAxes, fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # 4. 步数分布
    ax4 = fig.add_subplot(gs[1, 1])
    
    # 按阶段分色
    p1_steps = [s for s, p in zip(steps, phases) if p == 1]
    p2_steps = [s for s, p in zip(steps, phases) if p == 2]
    
    if p1_steps:
        ax4.hist(p1_steps, bins=30, alpha=0.5, color='blue', label='Phase 1 (E2E)')
    if p2_steps:
        ax4.hist(p2_steps, bins=30, alpha=0.5, color='green', label='Phase 2 (RL)')
    
    ax4.set_xlabel('Episode Steps')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Episode Length Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 添加总结信息
    summary = metrics.get_summary()
    summary_text = (
        f"Total Episodes: {summary['total_episodes']}\n"
        f"Best Reward: {summary['best_reward']:.2f} @ Ep {summary['best_episode']}\n"
        f"Final Success Rate: {summary['final_100_success_rate']*100:.1f}%\n"
        f"Phase 1 Mean: {summary['phase1_mean_reward']:.2f}\n"
        f"Phase 2 Mean: {summary['phase2_mean_reward']:.2f}"
    )
    fig.text(0.02, 0.02, summary_text, fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {save_path}")


def plot_encoder_analysis(
    encoder,  # DepthEncoder
    depth_samples: np.ndarray,
    save_path: str,
    depth_max_range: float = 5.0,
    title: str = "Encoder Analysis"
) -> Dict[str, float]:
    """
    分析编码器特征提取能力
    
    返回各项指标的数值
    """
    import torch
    from sklearn.decomposition import PCA
    
    device = next(encoder.parameters()).device
    
    # 1. 计算 embeddings
    encoder.eval()
    with torch.no_grad():
        depth_tensor = torch.FloatTensor(depth_samples).to(device)
        embeddings = encoder(depth_tensor).cpu().numpy()
    
    # 2. 计算深度特征
    features = _compute_depth_features(depth_samples, depth_max_range)
    
    # 3. 计算相关性
    correlations = _compute_correlations(embeddings, features)
    
    # 4. 计算 embedding 方差
    embedding_variance = embeddings.var(axis=0).mean()
    
    # 5. PCA 分析
    pca = PCA(n_components=min(embeddings.shape[1], 3))
    pca_result = pca.fit_transform(embeddings)
    explained_var = pca.explained_variance_ratio_.sum()
    
    # 绘图
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. 相关性热力图
    ax1 = fig.add_subplot(gs[0, 0])
    feature_names = list(features.keys())
    corr_matrix = np.array([[correlations.get(f'{fname}_dim{i}', 0) 
                            for fname in feature_names]
                           for i in range(embeddings.shape[1])])
    
    im = ax1.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax1.set_xticks(range(len(feature_names)))
    ax1.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=8)
    ax1.set_yticks(range(embeddings.shape[1]))
    ax1.set_yticklabels([f'Dim {i}' for i in range(embeddings.shape[1])], fontsize=8)
    ax1.set_title('Embedding-Feature Correlation')
    plt.colorbar(im, ax=ax1, shrink=0.8)
    
    # 2. Embedding 分布 (PCA)
    ax2 = fig.add_subplot(gs[0, 1])
    danger_levels = features['danger_level']
    scatter = ax2.scatter(pca_result[:, 0], pca_result[:, 1], 
                         c=danger_levels, cmap='RdYlGn_r', 
                         s=5, alpha=0.5)
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_title(f'Embedding Space (PCA, Var={explained_var*100:.1f}%)')
    plt.colorbar(scatter, ax=ax2, label='Danger Level')
    
    # 3. 各维度方差
    ax3 = fig.add_subplot(gs[0, 2])
    dim_vars = embeddings.var(axis=0)
    ax3.bar(range(len(dim_vars)), dim_vars, color='steelblue')
    ax3.set_xlabel('Embedding Dimension')
    ax3.set_ylabel('Variance')
    ax3.set_title(f'Dimension Variance (Mean={embedding_variance:.4f})')
    ax3.axhline(y=embedding_variance, color='red', linestyle='--', label='Mean')
    ax3.legend()
    
    # 4. 最强相关散点图
    ax4 = fig.add_subplot(gs[1, 0])
    max_corr_key = max(correlations, key=lambda k: abs(correlations[k]))
    fname, dim = max_corr_key.rsplit('_dim', 1)
    dim = int(dim)
    
    ax4.scatter(features[fname], embeddings[:, dim], s=3, alpha=0.3, c='steelblue')
    ax4.set_xlabel(fname)
    ax4.set_ylabel(f'Embedding Dim {dim}')
    ax4.set_title(f'Strongest Corr: {fname} (r={correlations[max_corr_key]:.3f})')
    
    # 添加回归线
    z = np.polyfit(features[fname], embeddings[:, dim], 1)
    p = np.poly1d(z)
    x_line = np.linspace(features[fname].min(), features[fname].max(), 100)
    ax4.plot(x_line, p(x_line), 'r-', linewidth=2)
    
    # 5. 深度-Embedding 响应曲线
    ax5 = fig.add_subplot(gs[1, 1])
    # 按最小深度排序
    sorted_idx = np.argsort(features['min_depth'])
    for i in range(min(3, embeddings.shape[1])):
        ax5.plot(features['min_depth'][sorted_idx], 
                embeddings[sorted_idx, i], 
                alpha=0.5, label=f'Dim {i}')
    ax5.set_xlabel('Min Depth')
    ax5.set_ylabel('Embedding Value')
    ax5.set_title('Embedding Response to Min Depth')
    ax5.legend()
    
    # 6. 各特征相关性条形图
    ax6 = fig.add_subplot(gs[1, 2])
    max_corrs = {}
    for fname in feature_names:
        corrs = [abs(correlations.get(f'{fname}_dim{i}', 0)) 
                for i in range(embeddings.shape[1])]
        max_corrs[fname] = max(corrs)
    
    sorted_features = sorted(max_corrs.items(), key=lambda x: x[1], reverse=True)
    names, values = zip(*sorted_features)
    colors = plt.cm.RdYlGn_r(np.array(values))
    ax6.barh(range(len(names)), values, color=colors)
    ax6.set_yticks(range(len(names)))
    ax6.set_yticklabels(names, fontsize=8)
    ax6.set_xlabel('Max |Correlation|')
    ax6.set_title('Feature-Embedding Max Correlation')
    ax6.axvline(x=0.3, color='gray', linestyle='--', alpha=0.5)
    ax6.set_xlim([0, 1])
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved encoder analysis to {save_path}")
    
    # 返回指标
    return {
        'embedding_variance': embedding_variance,
        'pca_explained_variance': explained_var,
        'max_correlation': max(abs(v) for v in correlations.values()),
        'mean_max_correlation': np.mean(list(max_corrs.values())),
    }


def _compute_depth_features(depth_samples: np.ndarray, max_range: float) -> Dict[str, np.ndarray]:
    """计算深度特征"""
    n_rays = depth_samples.shape[1]
    front_idx = n_rays // 4
    
    features = {
        'min_depth': depth_samples.min(axis=1),
        'max_depth': depth_samples.max(axis=1),
        'mean_depth': depth_samples.mean(axis=1),
        'std_depth': depth_samples.std(axis=1),
        'front_depth': depth_samples[:, max(0, front_idx-1):front_idx+2].mean(axis=1),
        'left_depth': depth_samples[:, :front_idx].mean(axis=1),
        'right_depth': depth_samples[:, front_idx:2*front_idx].mean(axis=1),
        'left_right_diff': (depth_samples[:, :front_idx].mean(axis=1) - 
                           depth_samples[:, front_idx:2*front_idx].mean(axis=1)),
        'danger_level': 1.0 / (depth_samples.min(axis=1) + 0.1),
    }
    return features


def _compute_correlations(
    embeddings: np.ndarray, 
    features: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """计算 embedding 各维度与特征的相关系数"""
    correlations = {}
    for fname, fvalues in features.items():
        for dim in range(embeddings.shape[1]):
            corr = np.corrcoef(embeddings[:, dim], fvalues)[0, 1]
            if not np.isnan(corr):
                correlations[f'{fname}_dim{dim}'] = corr
    return correlations


def visualize_trajectories(
    trajectories: List[np.ndarray],
    goals: List[np.ndarray],
    subgoals_history: Optional[List[List[np.ndarray]]] = None,
    obstacles: Optional[List[Tuple[float, float, float]]] = None,
    world_size: float = 10.0,
    save_path: str = "trajectories.png",
    title: str = "Agent Trajectories"
) -> None:
    """
    可视化智能体轨迹
    
    Args:
        trajectories: 轨迹列表，每个轨迹是 (T, 2) 数组
        goals: 目标位置列表
        subgoals_history: 子目标历史 (可选)
        obstacles: 障碍物列表 [(x, y, r), ...]
        world_size: 世界尺寸
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 绘制障碍物
    if obstacles:
        for ox, oy, r in obstacles:
            circle = plt.Circle((ox, oy), r, color='gray', alpha=0.5)
            ax.add_patch(circle)
    
    # 绘制轨迹
    cmap = plt.cm.viridis
    for i, (traj, goal) in enumerate(zip(trajectories, goals)):
        color = cmap(i / max(1, len(trajectories) - 1))
        
        # 轨迹线
        ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.7, linewidth=1.5)
        
        # 起点
        ax.scatter(traj[0, 0], traj[0, 1], c=[color], marker='o', s=50, 
                  edgecolors='black', linewidths=1, zorder=5)
        
        # 终点
        ax.scatter(traj[-1, 0], traj[-1, 1], c=[color], marker='s', s=50,
                  edgecolors='black', linewidths=1, zorder=5)
        
        # 目标
        ax.scatter(goal[0], goal[1], c='red', marker='*', s=200, 
                  edgecolors='darkred', linewidths=1, zorder=10)
        
        # 子目标 (如果有)
        if subgoals_history and i < len(subgoals_history):
            for sg in subgoals_history[i]:
                ax.scatter(sg[0], sg[1], c='yellow', marker='x', s=30, alpha=0.5)
    
    ax.set_xlim([0, world_size])
    ax.set_ylim([0, world_size])
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # 图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=10, label='Start'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
               markersize=10, label='End'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
               markersize=15, label='Goal'),
    ]
    if subgoals_history:
        legend_elements.append(
            Line2D([0], [0], marker='x', color='w', markerfacecolor='yellow',
                   markersize=10, label='Subgoal')
        )
    ax.legend(handles=legend_elements, loc='upper right')
    
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved trajectories to {save_path}")


def plot_reward_comparison(
    metrics_list: List['TrainingMetrics'],
    labels: List[str],
    save_path: str,
    title: str = "Training Comparison"
) -> None:
    """
    对比多组训练结果
    
    用于消融实验对比
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_list)))
    
    # 奖励曲线对比
    ax1 = axes[0]
    for metrics, label, color in zip(metrics_list, labels, colors):
        rewards = metrics.get_rewards_array()
        if len(rewards) >= 50:
            smooth = np.convolve(rewards, np.ones(50)/50, mode='valid')
            ax1.plot(range(49, len(rewards)), smooth, label=label, color=color, linewidth=2)
        else:
            ax1.plot(rewards, label=label, color=color, linewidth=2)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward (50-ep avg)')
    ax1.set_title('Reward Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 最终性能对比
    ax2 = axes[1]
    final_rewards = []
    final_srs = []
    for metrics in metrics_list:
        summary = metrics.get_summary()
        final_rewards.append(summary['final_100_mean_reward'])
        final_srs.append(summary['final_100_success_rate'] * 100)
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, final_rewards, width, label='Mean Reward', color='steelblue')
    ax2_twin = ax2.twinx()
    bars2 = ax2_twin.bar(x + width/2, final_srs, width, label='Success Rate', color='green')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('Mean Reward')
    ax2_twin.set_ylabel('Success Rate (%)')
    ax2.set_title('Final Performance (last 100 episodes)')
    
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison to {save_path}")
