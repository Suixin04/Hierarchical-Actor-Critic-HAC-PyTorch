"""
深度编码器可视化脚本

功能:
1. Embedding 与深度特征的相关性分析
2. 编码器雅可比矩阵可视化 (敏感度分析)
3. Embedding 空间结构 (PCA/t-SNE)
4. 不同危险程度下的 embedding 分布

用法:
    python visualize_encoder.py                           # 使用默认路径
    python visualize_encoder.py --model_dir preTrained/Navigation2DObstacle-v1/4level
    python visualize_encoder.py --level 1                 # 可视化 Level 1 编码器
    python visualize_encoder.py --save_dir figures        # 保存图片到指定目录
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
from matplotlib.gridspec import GridSpec
import gymnasium as gym
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import asset  # 注册自定义环境
from configs import get_config
from SAC import DepthEncoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_encoder(model_dir: str, level: int, config) -> DepthEncoder:
    """加载指定层级的编码器"""
    encoder = DepthEncoder(
        depth_dim=config.depth_dim,
        embedding_dim=config.embedding_dim,
        depth_max_range=config.depth_max_range
    ).to(device)
    
    encoder_path = os.path.join(model_dir, f'HAC_{config.env_name}_level_{level}_encoder.pth')
    if os.path.exists(encoder_path):
        encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
        print(f"Loaded encoder from {encoder_path}")
    else:
        print(f"Warning: Encoder not found at {encoder_path}, using random initialization")
    
    encoder.eval()
    return encoder


def collect_samples(env, config, num_samples=2000):
    """从环境中收集深度样本"""
    depth_samples = []
    states = []
    
    state, _ = env.reset()
    for _ in range(num_samples):
        depth = state[config.base_state_dim:]
        depth_samples.append(depth.copy())
        states.append(state.copy())
        
        # 随机动作探索
        action = env.action_space.sample()
        state, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            state, _ = env.reset()
    
    return np.array(depth_samples), np.array(states)


def compute_depth_features(depth_samples: np.ndarray, config) -> dict:
    """计算各种深度特征"""
    n_rays = depth_samples.shape[1]
    mid = n_rays // 2
    
    features = {
        'min_depth': depth_samples.min(axis=1),
        'max_depth': depth_samples.max(axis=1),
        'mean_depth': depth_samples.mean(axis=1),
        'std_depth': depth_samples.std(axis=1),
        'front_depth': depth_samples[:, mid-1:mid+2].mean(axis=1),  # 前方
        'left_depth': depth_samples[:, :n_rays//4].mean(axis=1),     # 左侧
        'right_depth': depth_samples[:, -n_rays//4:].mean(axis=1),   # 右侧
        'back_depth': depth_samples[:, n_rays//2-2:n_rays//2+3].mean(axis=1) if n_rays > 8 else depth_samples.mean(axis=1),
        'left_right_diff': depth_samples[:, :n_rays//4].mean(axis=1) - depth_samples[:, -n_rays//4:].mean(axis=1),
        'danger_level': 1.0 / (depth_samples.min(axis=1) + 0.1),  # 危险程度
    }
    return features


def compute_embeddings(encoder: DepthEncoder, depth_samples: np.ndarray) -> np.ndarray:
    """计算 embedding"""
    with torch.no_grad():
        depth_tensor = torch.FloatTensor(depth_samples).to(device)
        embeddings = encoder(depth_tensor).cpu().numpy()
    return embeddings


def compute_jacobian(encoder: DepthEncoder, depth_sample: np.ndarray) -> np.ndarray:
    """计算编码器雅可比矩阵 ∂z/∂d"""
    depth = torch.FloatTensor(depth_sample).to(device).requires_grad_(True)
    z = encoder(depth.unsqueeze(0)).squeeze(0)
    
    jacobian = torch.zeros(z.shape[0], depth.shape[0])
    for i in range(z.shape[0]):
        if depth.grad is not None:
            depth.grad.zero_()
        z[i].backward(retain_graph=True)
        jacobian[i] = depth.grad.clone()
    
    return jacobian.cpu().numpy()


def plot_correlation_analysis(embeddings: np.ndarray, features: dict, 
                              save_path: str, level: int):
    """绘制 embedding 与深度特征的相关性分析"""
    embed_dim = embeddings.shape[1]
    feature_names = list(features.keys())
    n_features = len(feature_names)
    
    # 计算相关性矩阵
    corr_matrix = np.zeros((embed_dim, n_features))
    for i in range(embed_dim):
        for j, fname in enumerate(feature_names):
            corr = np.corrcoef(embeddings[:, i], features[fname])[0, 1]
            corr_matrix[i, j] = corr if not np.isnan(corr) else 0
    
    # 绘制热力图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 热力图
    im = axes[0].imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    axes[0].set_xticks(range(n_features))
    axes[0].set_xticklabels(feature_names, rotation=45, ha='right')
    axes[0].set_yticks(range(embed_dim))
    axes[0].set_yticklabels([f'Dim {i}' for i in range(embed_dim)])
    axes[0].set_title(f'Level {level} Encoder: Correlation Matrix')
    axes[0].set_xlabel('Depth Features')
    axes[0].set_ylabel('Embedding Dimensions')
    
    # 添加数值标签
    for i in range(embed_dim):
        for j in range(n_features):
            text = axes[0].text(j, i, f'{corr_matrix[i, j]:.2f}',
                               ha='center', va='center', fontsize=8,
                               color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')
    
    plt.colorbar(im, ax=axes[0], label='Correlation')
    
    # 散点图: 最强相关的特征
    max_corr_idx = np.unravel_index(np.abs(corr_matrix).argmax(), corr_matrix.shape)
    best_embed_dim, best_feature_idx = max_corr_idx
    best_feature = feature_names[best_feature_idx]
    
    axes[1].scatter(features[best_feature], embeddings[:, best_embed_dim], 
                   alpha=0.3, s=5, c='steelblue')
    axes[1].set_xlabel(best_feature)
    axes[1].set_ylabel(f'Embedding Dim {best_embed_dim}')
    corr_val = corr_matrix[best_embed_dim, best_feature_idx]
    axes[1].set_title(f'Strongest Correlation: Dim {best_embed_dim} vs {best_feature}\n(r = {corr_val:.3f})')
    
    # 添加回归线
    z = np.polyfit(features[best_feature], embeddings[:, best_embed_dim], 1)
    p = np.poly1d(z)
    x_line = np.linspace(features[best_feature].min(), features[best_feature].max(), 100)
    axes[1].plot(x_line, p(x_line), 'r-', linewidth=2, label='Linear fit')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved correlation analysis to {save_path}")


def plot_jacobian_analysis(encoder: DepthEncoder, depth_samples: np.ndarray,
                           save_path: str, level: int, num_samples=100):
    """绘制雅可比矩阵分析 (编码器敏感度)"""
    # 随机选择样本计算雅可比
    indices = np.random.choice(len(depth_samples), min(num_samples, len(depth_samples)), replace=False)
    
    jacobians = []
    for idx in indices:
        jac = compute_jacobian(encoder, depth_samples[idx])
        jacobians.append(np.abs(jac))
    
    # 平均雅可比
    avg_jacobian = np.mean(jacobians, axis=0)
    std_jacobian = np.std(jacobians, axis=0)
    
    embed_dim, depth_dim = avg_jacobian.shape
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 热力图: 平均敏感度
    im = axes[0].imshow(avg_jacobian, aspect='auto', cmap='hot')
    axes[0].set_xlabel('Depth Ray Index (direction)')
    axes[0].set_ylabel('Embedding Dimension')
    axes[0].set_title(f'Level {level} Encoder Jacobian: |∂z/∂d| (Average over {num_samples} samples)')
    axes[0].set_xticks(range(depth_dim))
    axes[0].set_yticks(range(embed_dim))
    axes[0].set_yticklabels([f'Dim {i}' for i in range(embed_dim)])
    
    # 添加方向标签
    direction_labels = []
    for i in range(depth_dim):
        angle = (i / depth_dim - 0.5) * 360
        direction_labels.append(f'{int(angle)}°')
    axes[0].set_xticklabels(direction_labels, rotation=45, fontsize=8)
    
    plt.colorbar(im, ax=axes[0], label='Sensitivity |∂z/∂d|')
    
    # 条形图: 每个 embedding 维度对各方向的总敏感度
    x = np.arange(depth_dim)
    width = 0.8 / embed_dim
    colors = plt.cm.tab10(np.linspace(0, 1, embed_dim))
    
    for i in range(embed_dim):
        axes[1].bar(x + i * width - 0.4 + width/2, avg_jacobian[i], width, 
                   label=f'Dim {i}', color=colors[i], alpha=0.8)
    
    axes[1].set_xlabel('Depth Ray Index')
    axes[1].set_ylabel('Sensitivity |∂z/∂d|')
    axes[1].set_title('Sensitivity by Embedding Dimension')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(direction_labels, rotation=45, fontsize=8)
    axes[1].legend(loc='upper right')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Jacobian analysis to {save_path}")


def plot_embedding_space(embeddings: np.ndarray, features: dict,
                         save_path: str, level: int):
    """绘制 embedding 空间结构 (PCA + t-SNE)"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 根据危险程度着色
    danger = features['danger_level']
    danger_norm = (danger - danger.min()) / (danger.max() - danger.min() + 1e-6)
    
    # PCA
    if embeddings.shape[1] >= 2:
        pca = PCA(n_components=2)
        embed_pca = pca.fit_transform(embeddings)
        
        sc = axes[0, 0].scatter(embed_pca[:, 0], embed_pca[:, 1], 
                               c=danger_norm, cmap='RdYlGn_r', alpha=0.5, s=10)
        axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        axes[0, 0].set_title('PCA of Embeddings (colored by danger level)')
        plt.colorbar(sc, ax=axes[0, 0], label='Danger Level')
    
    # t-SNE (如果样本够多)
    if len(embeddings) >= 50:
        # 对大数据集采样
        n_tsne = min(1000, len(embeddings))
        indices = np.random.choice(len(embeddings), n_tsne, replace=False)
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_tsne-1))
        embed_tsne = tsne.fit_transform(embeddings[indices])
        
        sc = axes[0, 1].scatter(embed_tsne[:, 0], embed_tsne[:, 1],
                               c=danger_norm[indices], cmap='RdYlGn_r', alpha=0.5, s=10)
        axes[0, 1].set_xlabel('t-SNE 1')
        axes[0, 1].set_ylabel('t-SNE 2')
        axes[0, 1].set_title('t-SNE of Embeddings (colored by danger level)')
        plt.colorbar(sc, ax=axes[0, 1], label='Danger Level')
    else:
        axes[0, 1].text(0.5, 0.5, 'Not enough samples for t-SNE', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
    
    # 按左右差异着色
    lr_diff = features['left_right_diff']
    lr_norm = (lr_diff - lr_diff.min()) / (lr_diff.max() - lr_diff.min() + 1e-6)
    
    if embeddings.shape[1] >= 2:
        sc = axes[1, 0].scatter(embed_pca[:, 0], embed_pca[:, 1],
                               c=lr_diff, cmap='coolwarm', alpha=0.5, s=10)
        axes[1, 0].set_xlabel(f'PC1')
        axes[1, 0].set_ylabel(f'PC2')
        axes[1, 0].set_title('PCA of Embeddings (colored by left-right asymmetry)')
        plt.colorbar(sc, ax=axes[1, 0], label='Left - Right Depth')
    
    # 按前方深度着色
    front = features['front_depth']
    
    if embeddings.shape[1] >= 2:
        sc = axes[1, 1].scatter(embed_pca[:, 0], embed_pca[:, 1],
                               c=front, cmap='viridis', alpha=0.5, s=10)
        axes[1, 1].set_xlabel(f'PC1')
        axes[1, 1].set_ylabel(f'PC2')
        axes[1, 1].set_title('PCA of Embeddings (colored by front depth)')
        plt.colorbar(sc, ax=axes[1, 1], label='Front Depth')
    
    plt.suptitle(f'Level {level} Encoder: Embedding Space Structure', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved embedding space analysis to {save_path}")


def plot_embedding_distributions(embeddings: np.ndarray, features: dict,
                                 save_path: str, level: int):
    """绘制不同条件下的 embedding 分布"""
    embed_dim = embeddings.shape[1]
    
    # 根据危险程度分组
    danger = features['danger_level']
    thresholds = np.percentile(danger, [33, 66])
    
    safe_mask = danger < thresholds[0]
    medium_mask = (danger >= thresholds[0]) & (danger < thresholds[1])
    danger_mask = danger >= thresholds[1]
    
    fig, axes = plt.subplots(2, embed_dim, figsize=(4*embed_dim, 8))
    
    colors = {'Safe': 'green', 'Medium': 'orange', 'Dangerous': 'red'}
    
    for i in range(embed_dim):
        # 上行: 直方图
        axes[0, i].hist(embeddings[safe_mask, i], bins=30, alpha=0.5, 
                       label='Safe', color='green', density=True)
        axes[0, i].hist(embeddings[medium_mask, i], bins=30, alpha=0.5,
                       label='Medium', color='orange', density=True)
        axes[0, i].hist(embeddings[danger_mask, i], bins=30, alpha=0.5,
                       label='Dangerous', color='red', density=True)
        axes[0, i].set_xlabel(f'Embedding Dim {i}')
        axes[0, i].set_ylabel('Density')
        axes[0, i].set_title(f'Dim {i} Distribution by Danger Level')
        axes[0, i].legend(fontsize=8)
        
        # 下行: 箱线图
        data = [embeddings[safe_mask, i], embeddings[medium_mask, i], embeddings[danger_mask, i]]
        bp = axes[1, i].boxplot(data, labels=['Safe', 'Medium', 'Danger'], patch_artist=True)
        for patch, color in zip(bp['boxes'], ['green', 'orange', 'red']):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        axes[1, i].set_ylabel(f'Embedding Dim {i}')
        axes[1, i].set_title(f'Dim {i} by Danger Level')
    
    plt.suptitle(f'Level {level} Encoder: Embedding Distributions', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved embedding distributions to {save_path}")


def plot_depth_reconstruction(encoder: DepthEncoder, depth_samples: np.ndarray,
                              save_path: str, level: int, num_examples=8):
    """可视化原始深度 vs 编码后的表示"""
    fig, axes = plt.subplots(num_examples, 2, figsize=(12, 2*num_examples))
    
    indices = np.random.choice(len(depth_samples), num_examples, replace=False)
    
    for i, idx in enumerate(indices):
        depth = depth_samples[idx]
        
        # 计算 embedding
        with torch.no_grad():
            depth_tensor = torch.FloatTensor(depth).unsqueeze(0).to(device)
            embedding = encoder(depth_tensor).cpu().numpy().flatten()
        
        # 绘制深度雷达图
        n_rays = len(depth)
        angles = np.linspace(-np.pi, np.pi, n_rays, endpoint=False)
        
        # 深度极坐标图
        ax_polar = fig.add_subplot(num_examples, 2, 2*i+1, projection='polar')
        ax_polar.plot(angles, depth, 'b-', linewidth=2)
        ax_polar.fill(angles, depth, alpha=0.3)
        ax_polar.set_ylim(0, depth.max() * 1.1)
        ax_polar.set_title(f'Sample {idx}: Depth Profile', fontsize=10)
        
        # Embedding 条形图
        axes[i, 1].bar(range(len(embedding)), embedding, color='steelblue')
        axes[i, 1].set_xlabel('Embedding Dimension')
        axes[i, 1].set_ylabel('Value')
        axes[i, 1].set_title(f'Embedding (min_d={depth.min():.2f})', fontsize=10)
        axes[i, 1].set_xticks(range(len(embedding)))
        
        # 移除原来的空白子图
        axes[i, 0].axis('off')
    
    plt.suptitle(f'Level {level} Encoder: Depth → Embedding Examples', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved depth examples to {save_path}")


def generate_report(embeddings: np.ndarray, features: dict, level: int):
    """生成文字报告"""
    embed_dim = embeddings.shape[1]
    
    print("\n" + "="*60)
    print(f"  Level {level} Encoder Analysis Report")
    print("="*60)
    
    # 计算与各特征的相关性
    print("\n[Correlation with Depth Features]")
    for i in range(embed_dim):
        print(f"\n  Dimension {i}:")
        correlations = []
        for fname, fval in features.items():
            corr = np.corrcoef(embeddings[:, i], fval)[0, 1]
            if not np.isnan(corr):
                correlations.append((fname, corr))
        
        # 按相关性绝对值排序
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        for fname, corr in correlations[:3]:
            sign = "+" if corr > 0 else ""
            print(f"    {fname:20s}: {sign}{corr:.3f}")
    
    # Embedding 统计
    print("\n[Embedding Statistics]")
    for i in range(embed_dim):
        print(f"  Dim {i}: mean={embeddings[:, i].mean():.3f}, "
              f"std={embeddings[:, i].std():.3f}, "
              f"range=[{embeddings[:, i].min():.3f}, {embeddings[:, i].max():.3f}]")
    
    # 可分性分析
    print("\n[Separability Analysis]")
    danger = features['danger_level']
    thresholds = np.percentile(danger, [33, 66])
    safe_mask = danger < thresholds[0]
    danger_mask = danger >= thresholds[1]
    
    for i in range(embed_dim):
        safe_mean = embeddings[safe_mask, i].mean()
        danger_mean = embeddings[danger_mask, i].mean()
        pooled_std = np.sqrt((embeddings[safe_mask, i].std()**2 + embeddings[danger_mask, i].std()**2) / 2)
        
        # Cohen's d
        cohens_d = (danger_mean - safe_mean) / (pooled_std + 1e-6)
        
        effect = "small" if abs(cohens_d) < 0.5 else "medium" if abs(cohens_d) < 0.8 else "large"
        print(f"  Dim {i}: Safe vs Dangerous Cohen's d = {cohens_d:.3f} ({effect} effect)")
    
    print("\n" + "="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Visualize Depth Encoder')
    parser.add_argument('--env', type=str, default='Navigation2DObstacle-v1',
                        help='Environment name')
    parser.add_argument('--model_dir', type=str, 
                        default='preTrained/Navigation2DObstacle-v1/4level',
                        help='Directory containing trained models')
    parser.add_argument('--level', type=int, default=1,
                        help='Which level encoder to visualize (1, 2, ...)')
    parser.add_argument('--save_dir', type=str, default='figures/encoder_analysis',
                        help='Directory to save figures')
    parser.add_argument('--num_samples', type=int, default=2000,
                        help='Number of samples to collect')
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载配置
    config = get_config(args.env)
    
    if not config.use_depth_encoder:
        print("Error: This environment does not use depth encoder")
        return
    
    print(f"Analyzing Level {args.level} encoder for {args.env}")
    print(f"Depth dim: {config.depth_dim}, Embedding dim: {config.embedding_dim}")
    
    # 加载编码器
    encoder = load_encoder(args.model_dir, args.level, config)
    
    # 创建环境并收集样本
    env = gym.make(config.env_name)
    print(f"Collecting {args.num_samples} samples from environment...")
    depth_samples, states = collect_samples(env, config, args.num_samples)
    env.close()
    
    # 计算深度特征
    print("Computing depth features...")
    features = compute_depth_features(depth_samples, config)
    
    # 计算 embeddings
    print("Computing embeddings...")
    embeddings = compute_embeddings(encoder, depth_samples)
    
    # 生成所有可视化
    prefix = f"level{args.level}"
    
    print("\nGenerating visualizations...")
    
    # 1. 相关性分析
    plot_correlation_analysis(
        embeddings, features,
        os.path.join(args.save_dir, f'{prefix}_correlation.png'),
        args.level
    )
    
    # 2. 雅可比分析
    plot_jacobian_analysis(
        encoder, depth_samples,
        os.path.join(args.save_dir, f'{prefix}_jacobian.png'),
        args.level
    )
    
    # 3. Embedding 空间结构
    plot_embedding_space(
        embeddings, features,
        os.path.join(args.save_dir, f'{prefix}_embedding_space.png'),
        args.level
    )
    
    # 4. Embedding 分布
    plot_embedding_distributions(
        embeddings, features,
        os.path.join(args.save_dir, f'{prefix}_distributions.png'),
        args.level
    )
    
    # 5. 深度-Embedding 示例
    plot_depth_reconstruction(
        encoder, depth_samples,
        os.path.join(args.save_dir, f'{prefix}_examples.png'),
        args.level
    )
    
    # 6. 文字报告
    generate_report(embeddings, features, args.level)
    
    print(f"\nAll figures saved to {args.save_dir}/")
    print("Done!")


if __name__ == '__main__':
    main()
