"""
HAC 调试训练脚本 - 用于诊断训练问题
"""
import argparse
import torch
import gymnasium as gym
import numpy as np
import os

import asset  # 注册自定义环境
from configs import get_config
from HAC import HAC


def debug_config(config):
    """调试配置"""
    print("\n" + "=" * 60)
    print("配置诊断")
    print("=" * 60)
    
    print(f"\n[环境]")
    print(f"  env_name: {config.env_name}")
    print(f"  state_dim: {config.state_dim}")
    print(f"  action_dim: {config.action_dim}")
    
    print(f"\n[目标空间]")
    print(f"  goal_indices: {config.goal_indices}")
    print(f"  effective_goal_dim: {config.effective_goal_dim}")
    print(f"  goal_state: {config.goal_state}")
    print(f"  goal_threshold: {config.goal_threshold}")
    
    print(f"\n[子目标空间]")
    print(f"  subgoal_bounds: {config.get_subgoal_bounds()}")
    print(f"  subgoal_offset: {config.get_subgoal_offset()}")
    print(f"  subgoal_clip_low: {config.get_subgoal_clip_low()}")
    print(f"  subgoal_clip_high: {config.get_subgoal_clip_high()}")
    
    print(f"\n[动作空间]")
    print(f"  action_bounds: {config.action_bounds}")
    print(f"  action_offset: {config.action_offset}")
    print(f"  action_clip_low: {config.action_clip_low}")
    print(f"  action_clip_high: {config.action_clip_high}")
    
    print(f"\n[HAC参数]")
    print(f"  k_level: {config.k_level}")
    print(f"  H: {config.H}")
    print(f"  lamda: {config.lamda}")
    print(f"  gamma: {config.gamma}")
    print(f"  max_steps (H^k_level): {config.H ** config.k_level}")
    
    print(f"\n[探索噪声]")
    print(f"  exploration_action_noise: {config.exploration_action_noise}")
    print(f"  exploration_state_noise: {config.exploration_state_noise}")


def debug_environment(env, config):
    """调试环境"""
    print("\n" + "=" * 60)
    print("环境诊断")
    print("=" * 60)
    
    state, _ = env.reset()
    print(f"\n[初始状态]")
    print(f"  shape: {state.shape}")
    print(f"  state: {state}")
    print(f"  提取的目标部分: {state[config.goal_indices]}")
    
    print(f"\n[目标]")
    print(f"  goal_state: {config.goal_state}")
    
    # 计算初始距离
    goal_part = state[config.goal_indices]
    dist = np.abs(goal_part - config.goal_state)
    print(f"  初始距离: {dist}")
    print(f"  阈值: {config.goal_threshold}")
    print(f"  是否达成: {np.all(dist < config.goal_threshold)}")
    
    # 测试随机动作
    print(f"\n[随机动作测试]")
    for i in range(5):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        print(f"  step {i+1}: action={action}, reward={reward}, terminated={terminated}")
        print(f"           state[:5]={next_state[:5]}")


def debug_subgoal_generation(agent, state, goal):
    """调试子目标生成"""
    print("\n" + "=" * 60)
    print("子目标生成诊断")
    print("=" * 60)
    
    for level in range(agent.k_level - 1, -1, -1):
        print(f"\n[Level {level}]")
        action = agent.HAC[level].select_action(state, goal, deterministic=True)
        print(f"  输入状态: {state[:5]}...")
        print(f"  输入目标: {goal}")
        print(f"  输出动作/子目标: {action}")
        
        if level > 0:
            # 高层输出子目标
            print(f"  子目标范围: [{agent.subgoal_clip_low}, {agent.subgoal_clip_high}]")
            # 检查是否在合理范围
            in_range = np.all(action >= agent.subgoal_clip_low) and np.all(action <= agent.subgoal_clip_high)
            print(f"  子目标在范围内: {in_range}")
        else:
            # 底层输出动作
            print(f"  动作范围: [{agent.action_clip_low}, {agent.action_clip_high}]")


def run_debug_episode(env, agent, config, max_steps=100):
    """运行一个调试episode"""
    print("\n" + "=" * 60)
    print("调试 Episode")
    print("=" * 60)
    
    state, _ = env.reset()
    agent.reset_episode()
    
    goal = config.goal_state
    print(f"\n初始状态: {state[:5]}")
    print(f"目标: {goal}")
    
    total_reward = 0
    goals_achieved = 0
    
    for step in range(max_steps):
        # 底层动作
        level_0_goal = agent.goals[0] if agent.goals[0] is not None else goal
        action = agent.HAC[0].select_action(state, level_0_goal, deterministic=False)
        
        # 添加探索噪声
        if np.random.random() > 0.2:
            action = action + np.random.normal(0, config.exploration_action_noise)
            action = np.clip(action, config.action_clip_low, config.action_clip_high)
        
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 20 == 0 or terminated or truncated:
            state_goal = state[config.goal_indices]
            dist = np.abs(state_goal - goal)
            print(f"\nStep {step}:")
            print(f"  位置: ({state[0]:.2f}, {state[1]:.2f}), θ={state[2]:.2f}")
            print(f"  速度: v={state[3]:.2f}, ω={state[4]:.2f}")
            print(f"  动作: a_v={action[0]:.2f}, a_ω={action[1]:.2f}")
            print(f"  距离目标: {dist}")
            print(f"  奖励: {reward}, 累计: {total_reward}")
            
            if info.get('reached_goal', False):
                goals_achieved += 1
                print(f"  *** 达成目标! ***")
        
        if terminated or truncated:
            print(f"\nEpisode结束: terminated={terminated}, truncated={truncated}")
            break
        
        state = next_state
    
    print(f"\n总结: {step+1} 步, 总奖励={total_reward}, 达成目标次数={goals_achieved}")
    return total_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Navigation2DSimple-v1')
    parser.add_argument('--algorithm', type=str, default='sac')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    
    # 获取配置
    config = get_config(args.env)
    config.algorithm = args.algorithm
    
    # 诊断配置
    debug_config(config)
    
    # 创建环境
    max_steps = config.H ** config.k_level
    print(f"\n创建环境，max_steps={max_steps}")
    
    if args.render:
        env = gym.make(config.env_name, render_mode="human", max_steps=max_steps)
    else:
        env = gym.make(config.env_name, max_steps=max_steps)
    
    # 诊断环境
    debug_environment(env, config)
    
    # 创建智能体
    agent = HAC(config, render=args.render, algorithm=config.algorithm)
    
    # 诊断子目标生成
    state, _ = env.reset()
    debug_subgoal_generation(agent, state, config.goal_state)
    
    # 运行调试episode
    print("\n\n" + "=" * 60)
    print("运行调试 Episode (底层策略直接控制)")
    print("=" * 60)
    run_debug_episode(env, agent, config, max_steps=200)
    
    env.close()


if __name__ == '__main__':
    main()
