"""
子目标安全约束诊断脚本

功能：
1. 观测原始子目标 vs 安全约束后子目标的差异
2. 统计子目标被修改的频率和幅度
3. 分析子目标与深度信息的关系
4. 可视化子目标投影过程
5. 检验安全约束是否有效防止碰撞

用法：
    python diagnose_safety.py                          # 基本诊断
    python diagnose_safety.py --render                 # 带渲染
    python diagnose_safety.py --episodes 20           # 更多episode
    python diagnose_safety.py --disable_constraint    # 对比：禁用约束
"""
import argparse
import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

import asset
from configs import get_config
from HAC import HAC, project_subgoal_to_safe_region


class SafetyDiagnosticHAC(HAC):
    """带诊断功能的 HAC，记录子目标约束过程"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diagnostic_data = defaultdict(list)
        self.current_episode_data = []
    
    def reset_diagnostics(self):
        """重置当前episode的诊断数据"""
        self.current_episode_data = []
    
    def get_episode_stats(self):
        """获取当前episode的统计"""
        if not self.current_episode_data:
            return {}
        
        data = self.current_episode_data
        num_subgoals = len(data)
        num_modified = sum(1 for d in data if d['was_modified'])
        
        if num_modified > 0:
            avg_modification = np.mean([d['modification_dist'] for d in data if d['was_modified']])
        else:
            avg_modification = 0
        
        return {
            'num_subgoals': num_subgoals,
            'num_modified': num_modified,
            'modification_rate': num_modified / num_subgoals if num_subgoals > 0 else 0,
            'avg_modification_dist': avg_modification,
        }
    
    def _apply_subgoal_safety_constraint(self, subgoal, state):
        """重写以记录诊断信息"""
        if not self.use_subgoal_safety_constraint:
            return subgoal
        
        if self.depth_dim == 0:
            return subgoal
        
        original_subgoal = subgoal.copy()
        depth_readings = state[self.base_state_dim:]
        
        # 投影到安全区域
        safe_subgoal = project_subgoal_to_safe_region(
            subgoal=subgoal,
            state=state,
            depth_readings=depth_readings,
            depth_fov=self.depth_fov,
            safe_margin=self.subgoal_safe_margin,
            min_subgoal_dist=self.subgoal_min_dist,
        )
        
        # 记录诊断数据
        modification_dist = np.linalg.norm(original_subgoal - safe_subgoal)
        was_modified = modification_dist > 0.05
        
        robot_pos = state[:2]
        robot_theta = state[2]
        
        # 计算子目标方向对应的深度
        delta = original_subgoal - robot_pos
        dist_to_original = np.linalg.norm(delta)
        if dist_to_original > 1e-6:
            subgoal_angle_world = np.arctan2(delta[1], delta[0])
            relative_angle = subgoal_angle_world - robot_theta
            while relative_angle > np.pi:
                relative_angle -= 2 * np.pi
            while relative_angle < -np.pi:
                relative_angle += 2 * np.pi
            
            n_rays = len(depth_readings)
            angle_step = self.depth_fov / n_rays
            start_angle = -self.depth_fov / 2
            ray_index = int((relative_angle - start_angle) / angle_step)
            ray_index = np.clip(ray_index, 0, n_rays - 1)
            depth_in_direction = depth_readings[ray_index]
        else:
            depth_in_direction = depth_readings.min()
            relative_angle = 0
        
        diag_entry = {
            'robot_pos': robot_pos.copy(),
            'robot_theta': robot_theta,
            'original_subgoal': original_subgoal.copy(),
            'safe_subgoal': safe_subgoal.copy(),
            'was_modified': was_modified,
            'modification_dist': modification_dist,
            'dist_to_original': dist_to_original,
            'depth_in_direction': depth_in_direction,
            'relative_angle': relative_angle,
            'depth_readings': depth_readings.copy(),
            'min_depth': depth_readings.min(),
            'mean_depth': depth_readings.mean(),
        }
        
        self.current_episode_data.append(diag_entry)
        self.diagnostic_data['all_entries'].append(diag_entry)
        
        return safe_subgoal


def run_diagnostic(args):
    """运行诊断"""
    config = get_config(args.env)
    config.algorithm = 'sac'
    
    # 可选禁用约束（用于对比）
    if args.disable_constraint:
        config.use_subgoal_safety_constraint = False
        print("⚠️  Safety constraint DISABLED for comparison")
    else:
        config.use_subgoal_safety_constraint = True
        print("✅ Safety constraint ENABLED")
    
    print("=" * 60)
    print("  Safety Constraint Diagnostic")
    print("=" * 60)
    print(f"  Safe margin: {config.subgoal_safe_margin}m")
    print(f"  Min subgoal dist: {config.subgoal_min_dist}m")
    print(f"  Episodes: {args.episodes}")
    print("=" * 60)
    
    # 环境
    max_steps = config.H ** config.k_level
    if args.render:
        env = gym.make(config.env_name, render_mode="human", max_steps=max_steps)
    else:
        env = gym.make(config.env_name, max_steps=max_steps)
    
    # 使用诊断版 HAC
    agent = SafetyDiagnosticHAC(config, render=args.render, algorithm='sac')
    
    # 加载模型（如果存在）
    model_dir = config.get_save_directory()
    base_name = config.get_filename()
    best_name = base_name + '_best'
    best_path = os.path.join(model_dir, f"{best_name}_level_1_actor.pth")
    default_path = os.path.join(model_dir, f"{base_name}_level_1_actor.pth")
    
    if os.path.exists(best_path):
        model_name = best_name
        print(f"Loading best model...")
        agent.load(model_dir, model_name)
    elif os.path.exists(default_path):
        model_name = base_name
        print(f"Loading default model...")
        agent.load(model_dir, model_name)
    else:
        print(f"⚠️  No trained model found, using random policy for diagnostic")
    
    # 运行诊断
    episode_stats = []
    collision_counts = []
    
    for ep in range(1, args.episodes + 1):
        agent.reset_episode()
        agent.reset_diagnostics()
        
        state, _ = env.reset()
        
        if hasattr(env.unwrapped, 'obstacles'):
            agent.set_obstacles(env.unwrapped.obstacles)
        
        last_state, done = agent.run_HAC(
            env, config.k_level - 1, state, config.goal_state, is_subgoal_test=True
        )
        
        success = agent.check_goal(last_state, config.goal_state, config.goal_threshold)
        
        # 检查碰撞（通过检查是否提前终止且未到达目标）
        collision = done and not success and agent.timestep < max_steps
        collision_counts.append(int(collision))
        
        stats = agent.get_episode_stats()
        stats['success'] = success
        stats['collision'] = collision
        stats['reward'] = agent.reward
        stats['steps'] = agent.timestep
        episode_stats.append(stats)
        
        status = "✓" if success else ("💥" if collision else "✗")
        if stats.get('num_subgoals', 0) > 0:
            mod_info = f"Modified: {stats['num_modified']}/{stats['num_subgoals']}"
        else:
            mod_info = "Constraint OFF"
        print(f"Ep {ep}: R={agent.reward:.1f} Steps={agent.timestep} {mod_info} {status}")
    
    env.close()
    
    # ==================== 分析结果 ====================
    print("\n" + "=" * 60)
    print("  DIAGNOSTIC RESULTS")
    print("=" * 60)
    
    # 总体统计
    total_subgoals = sum(s.get('num_subgoals', 0) for s in episode_stats)
    total_modified = sum(s.get('num_modified', 0) for s in episode_stats)
    successes = sum(1 for s in episode_stats if s['success'])
    collisions = sum(collision_counts)
    
    print(f"\n📊 Overall Statistics:")
    print(f"  Total subgoals generated: {total_subgoals}")
    print(f"  Total subgoals modified:  {total_modified} ({100*total_modified/max(1,total_subgoals):.1f}%)")
    print(f"  Success rate: {successes}/{args.episodes} ({100*successes/args.episodes:.1f}%)")
    print(f"  Collision rate: {collisions}/{args.episodes} ({100*collisions/args.episodes:.1f}%)")
    
    if total_modified > 0:
        all_entries = agent.diagnostic_data['all_entries']
        modified_entries = [e for e in all_entries if e['was_modified']]
        
        avg_mod_dist = np.mean([e['modification_dist'] for e in modified_entries])
        avg_depth_when_modified = np.mean([e['depth_in_direction'] for e in modified_entries])
        avg_original_dist = np.mean([e['dist_to_original'] for e in modified_entries])
        
        print(f"\n📏 Modification Analysis:")
        print(f"  Avg modification distance: {avg_mod_dist:.3f}m")
        print(f"  Avg depth when modified:   {avg_depth_when_modified:.3f}m")
        print(f"  Avg original subgoal dist: {avg_original_dist:.3f}m")
        
        # 分析：原始子目标是否真的超出了深度范围
        violations = [e for e in modified_entries 
                     if e['dist_to_original'] > e['depth_in_direction'] - config.subgoal_safe_margin]
        print(f"  True safety violations:    {len(violations)}/{len(modified_entries)}")
    
    # ==================== 可视化 ====================
    if args.plot and len(agent.diagnostic_data['all_entries']) > 0:
        plot_diagnostics(agent.diagnostic_data, config, args.disable_constraint)
    
    return episode_stats, agent.diagnostic_data


def plot_diagnostics(data, config, disabled=False):
    """可视化诊断结果"""
    entries = data['all_entries']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    title_suffix = " (DISABLED)" if disabled else ""
    fig.suptitle(f"Subgoal Safety Constraint Diagnostics{title_suffix}", fontsize=14)
    
    # 1. 子目标修改距离分布
    ax = axes[0, 0]
    mod_dists = [e['modification_dist'] for e in entries if e['was_modified']]
    if mod_dists:
        ax.hist(mod_dists, bins=30, color='coral', edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(mod_dists), color='red', linestyle='--', label=f'Mean: {np.mean(mod_dists):.2f}m')
        ax.legend()
    ax.set_xlabel('Modification Distance (m)')
    ax.set_ylabel('Count')
    ax.set_title('Subgoal Modification Distance Distribution')
    
    # 2. 原始子目标距离 vs 该方向深度
    ax = axes[0, 1]
    original_dists = [e['dist_to_original'] for e in entries]
    depths = [e['depth_in_direction'] for e in entries]
    colors = ['red' if e['was_modified'] else 'green' for e in entries]
    ax.scatter(depths, original_dists, c=colors, alpha=0.5, s=20)
    
    # 添加安全边界线
    x_range = np.linspace(0, max(depths), 100)
    ax.plot(x_range, x_range - config.subgoal_safe_margin, 'b--', 
            label=f'Safety boundary (margin={config.subgoal_safe_margin}m)')
    ax.plot([0, max(depths)], [0, max(depths)], 'k:', alpha=0.3, label='y=x')
    ax.set_xlabel('Depth in Subgoal Direction (m)')
    ax.set_ylabel('Original Subgoal Distance (m)')
    ax.set_title('Subgoal Distance vs Depth')
    ax.legend()
    
    # 3. 修改发生的角度分布（极坐标）
    ax = axes[0, 2]
    ax = plt.subplot(2, 3, 3, projection='polar')
    modified_angles = [e['relative_angle'] for e in entries if e['was_modified']]
    if modified_angles:
        ax.hist(modified_angles, bins=16, color='coral', alpha=0.7)
    ax.set_title('Modified Subgoal Directions')
    
    # 4. 时序：修改率随 episode 变化
    ax = axes[1, 0]
    # 按每10个子目标分组
    window = 20
    if len(entries) >= window:
        rolling_mod_rate = []
        for i in range(0, len(entries) - window + 1, window // 2):
            batch = entries[i:i+window]
            rate = sum(1 for e in batch if e['was_modified']) / len(batch)
            rolling_mod_rate.append(rate)
        ax.plot(rolling_mod_rate, 'b-', linewidth=2)
        ax.axhline(np.mean(rolling_mod_rate), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(rolling_mod_rate):.2%}')
        ax.legend()
    ax.set_xlabel('Window Index')
    ax.set_ylabel('Modification Rate')
    ax.set_title('Modification Rate Over Time')
    ax.set_ylim([0, 1])
    
    # 5. 深度分布（修改 vs 未修改）
    ax = axes[1, 1]
    min_depths_modified = [e['min_depth'] for e in entries if e['was_modified']]
    min_depths_safe = [e['min_depth'] for e in entries if not e['was_modified']]
    if min_depths_modified:
        ax.hist(min_depths_modified, bins=20, alpha=0.6, label='Modified', color='red')
    if min_depths_safe:
        ax.hist(min_depths_safe, bins=20, alpha=0.6, label='Not Modified', color='green')
    ax.set_xlabel('Min Depth Reading (m)')
    ax.set_ylabel('Count')
    ax.set_title('Min Depth: Modified vs Safe Subgoals')
    ax.legend()
    
    # 6. 子目标轨迹示例（最后几个）
    ax = axes[1, 2]
    recent = entries[-50:] if len(entries) > 50 else entries
    for e in recent:
        robot = e['robot_pos']
        orig = e['original_subgoal']
        safe = e['safe_subgoal']
        
        # 机器人位置
        ax.plot(robot[0], robot[1], 'ko', markersize=3, alpha=0.3)
        
        if e['was_modified']:
            # 原始子目标（红色）
            ax.plot(orig[0], orig[1], 'rx', markersize=5, alpha=0.5)
            # 安全子目标（绿色）
            ax.plot(safe[0], safe[1], 'g^', markersize=5, alpha=0.7)
            # 连线
            ax.plot([orig[0], safe[0]], [orig[1], safe[1]], 'b-', alpha=0.3, linewidth=0.5)
        else:
            ax.plot(orig[0], orig[1], 'g^', markersize=4, alpha=0.5)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Subgoal Projections (Recent)')
    ax.set_aspect('equal')
    ax.legend(['Robot', 'Unsafe', 'Safe'], loc='upper left')
    
    plt.tight_layout()
    
    # 保存
    save_path = 'figures/safety_diagnostic.png'
    os.makedirs('figures', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📈 Diagnostic plot saved to: {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Safety Constraint Diagnostic')
    parser.add_argument('--env', type=str, default='Navigation2DObstacle-v1')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--plot', action='store_true', default=True)
    parser.add_argument('--disable_constraint', action='store_true',
                        help='Disable safety constraint for comparison')
    args = parser.parse_args()
    
    run_diagnostic(args)


if __name__ == '__main__':
    main()
