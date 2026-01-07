"""
诊断脚本：验证 HAC 各层 encoder 训练配置是否符合预期
"""
import torch
import numpy as np
import gymnasium as gym

# 导入 asset 触发环境注册
import asset

from HAC import HAC
from configs import Navigation2DObstacleConfig

# 实例化配置
Config = Navigation2DObstacleConfig()

def diagnose_encoder_setup():
    """诊断 encoder 设置"""
    print("=" * 60)
    print("诊断 1: Encoder 初始化配置")
    print("=" * 60)
    
    # 创建环境和 agent
    env = gym.make(Config.env_name, render_mode=None)
    agent = HAC(Config, render=False)
    
    print(f"\nk_level = {Config.k_level}")
    print(f"use_depth_encoder = {Config.use_depth_encoder}")
    print(f"encoder_finetune_lr = {Config.encoder_finetune_lr}")
    
    # 检查每一层
    for level in range(Config.k_level):
        print(f"\n--- Level {level} ---")
        if level == 0:
            print("  Type: MPC (无 encoder)")
            continue
        
        policy = agent.HAC[level]
        print(f"  Type: SAC")
        print(f"  encoder_train_mode: {policy.encoder_train_mode}")
        print(f"  depth_encoder: {policy.depth_encoder is not None}")
        print(f"  encoder_finetune_optimizer: {policy.encoder_finetune_optimizer is not None}")
        
        if policy.depth_encoder is not None:
            # 检查 encoder 参数的 requires_grad
            encoder_params = list(policy.depth_encoder.parameters())
            requires_grad = [p.requires_grad for p in encoder_params]
            print(f"  encoder params requires_grad: {all(requires_grad)}")
    
    return agent, env


def diagnose_gradient_flow(agent, env):
    """诊断梯度流动"""
    print("\n" + "=" * 60)
    print("诊断 2: 梯度流动测试 (Phase 1 - 初始状态)")
    print("=" * 60)
    
    # 填充一些假数据到 replay buffer
    state = env.reset()[0]
    goal = np.array([8.0, 8.0])
    goal_dim = agent.goal_dim  # 使用 agent 的 goal_dim
    
    for level in range(1, Config.k_level):
        for _ in range(100):
            fake_action = np.random.randn(goal_dim)  # 子目标维度
            fake_next_state = state + np.random.randn(*state.shape) * 0.1
            agent.replay_buffer[level].add((
                state, fake_action, -1.0, fake_next_state, goal, 0.99, 0.0
            ))
    
    # 清零梯度
    for level in range(1, Config.k_level):
        policy = agent.HAC[level]
        if policy.depth_encoder is not None:
            policy.depth_encoder.zero_grad()
    
    # 执行一次 update
    agent.update(n_iter=1, batch_size=32)
    
    # 检查 encoder 梯度
    print("\n梯度检查:")
    for level in range(1, Config.k_level):
        policy = agent.HAC[level]
        if policy.depth_encoder is not None:
            grad_norm = 0.0
            for p in policy.depth_encoder.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.norm().item() ** 2
            grad_norm = grad_norm ** 0.5
            
            has_grad = grad_norm > 1e-10
            print(f"  Level {level}: encoder grad_norm = {grad_norm:.6f}, "
                  f"has_grad = {has_grad}, "
                  f"optimizer = {policy.encoder_finetune_optimizer is not None}")
            
            # 期望行为
            if level == 1:
                expected = "无梯度 (finetune 模式，尚未启用 finetune)"
                actual = "无梯度" if not has_grad else "有梯度 ❌"
            else:
                expected = "有梯度 (rl 模式)"
                actual = "有梯度 ✓" if has_grad else "无梯度 ❌"
            
            print(f"    期望: {expected}")
            print(f"    实际: {actual}")


def diagnose_after_finetune_enable(agent, env):
    """诊断启用 finetune 后的梯度流动"""
    print("\n" + "=" * 60)
    print("诊断 3: 梯度流动测试 (Phase 2 - 启用 Level 1 finetune)")
    print("=" * 60)
    
    # 启用 Level 1 encoder finetune
    print("\n调用 enable_level1_encoder_finetune()...")
    agent.enable_level1_encoder_finetune()
    
    # 检查状态变化
    policy = agent.HAC[1]
    print(f"\nLevel 1 状态:")
    print(f"  encoder_finetune_optimizer: {policy.encoder_finetune_optimizer is not None}")
    
    # 清零梯度
    for level in range(1, Config.k_level):
        policy = agent.HAC[level]
        if policy.depth_encoder is not None:
            policy.depth_encoder.zero_grad()
    
    # 执行一次 update
    agent.update(n_iter=1, batch_size=32)
    
    # 检查 encoder 梯度
    print("\n梯度检查:")
    for level in range(1, Config.k_level):
        policy = agent.HAC[level]
        if policy.depth_encoder is not None:
            grad_norm = 0.0
            for p in policy.depth_encoder.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.norm().item() ** 2
            grad_norm = grad_norm ** 0.5
            
            has_grad = grad_norm > 1e-10
            print(f"  Level {level}: encoder grad_norm = {grad_norm:.6f}, "
                  f"has_grad = {has_grad}")
            
            # 期望行为
            if level == 1:
                expected = "有梯度 (finetune 已启用)"
                actual = "有梯度 ✓" if has_grad else "无梯度 ❌"
            else:
                expected = "有梯度 (rl 模式)"
                actual = "有梯度 ✓" if has_grad else "无梯度 ❌"
            
            print(f"    期望: {expected}")
            print(f"    实际: {actual}")


def diagnose_optimizer_lr(agent):
    """诊断 optimizer 学习率"""
    print("\n" + "=" * 60)
    print("诊断 4: Optimizer 学习率检查")
    print("=" * 60)
    
    for level in range(1, Config.k_level):
        policy = agent.HAC[level]
        print(f"\n--- Level {level} ---")
        
        # Actor optimizer lr
        actor_lr = policy.actor_optimizer.param_groups[0]['lr']
        print(f"  actor_optimizer lr: {actor_lr}")
        
        # Encoder optimizer lr
        if policy.encoder_finetune_optimizer is not None:
            enc_lr = policy.encoder_finetune_optimizer.param_groups[0]['lr']
            print(f"  encoder_finetune_optimizer lr: {enc_lr}")
            
            if level == 1:
                expected_lr = Config.encoder_finetune_lr
                if abs(enc_lr - expected_lr) < 1e-10:
                    print(f"    ✓ 符合预期 (encoder_finetune_lr = {expected_lr})")
                else:
                    print(f"    ❌ 不符合预期 (应为 {expected_lr})")
            else:
                expected_lr = Config.lr
                if abs(enc_lr - expected_lr) < 1e-10:
                    print(f"    ✓ 符合预期 (正常 lr = {expected_lr})")
                else:
                    print(f"    ❌ 不符合预期 (应为 {expected_lr})")
        else:
            print(f"  encoder_finetune_optimizer: None")
            if level == 1:
                print(f"    (finetune 尚未启用)")


def main():
    print("HAC Encoder 训练配置诊断")
    print("预期行为:")
    print("  - Level 0: MPC, 无 encoder")
    print("  - Level 1: finetune 模式, 初始无 optimizer, Phase 2 用小学习率微调")
    print("  - Level 2+: rl 模式, 初始有正常学习率 optimizer")
    print()
    
    # 诊断 1: 初始化配置
    agent, env = diagnose_encoder_setup()
    
    # 诊断 2: Phase 1 梯度流动
    diagnose_gradient_flow(agent, env)
    
    # 诊断 4: Phase 1 optimizer lr
    diagnose_optimizer_lr(agent)
    
    # 诊断 3: Phase 2 梯度流动
    diagnose_after_finetune_enable(agent, env)
    
    # 诊断 4: Phase 2 optimizer lr
    diagnose_optimizer_lr(agent)
    
    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)
    
    env.close()


if __name__ == "__main__":
    main()
