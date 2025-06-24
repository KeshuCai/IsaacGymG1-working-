
from isaacgym import gymapi, gymtorch
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from policy import MLPPolicy

# 设备与超参数
device = torch.device("cuda")
num_envs = 16
obs_dim = 13
act_dim = 12  # 修改为你模型的 DOF 数量
lr = 3e-4
total_steps = 1000

# 初始化 Isaac Gym
gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim_params.use_gpu_pipeline = True
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0, 0, -9.81)
sim = gym.create_sim(0, 0, gymapi.SimType.SIM_PHYSX, sim_params)
gym.add_ground(sim, gymapi.PlaneParams())

# 加载模型
asset_opts = gymapi.AssetOptions()
asset_opts.fix_base_link = True
asset = gym.load_asset(sim, "/home/kc/IsaacGymEnvs/assets/mjcf/g1_description"  ,"g1_29dof_with_hand.xml")#, "your_robot.xml", asset_opts)

# 创建多个环境
envs = []
actors = []
spacing = 2.0
for i in range(num_envs):
    env = gym.create_env(sim, gymapi.Vec3(-spacing, -spacing, 0), gymapi.Vec3(spacing, spacing, spacing), num_envs)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3((i % 4) * spacing, (i // 4) * spacing, 0.3)
    actor = gym.create_actor(env, asset, pose, f"robot_{i}", i, 1)
    envs.append(env)
    actors.append(actor)

# 准备 Tensor API
gym.prepare_sim(sim)
root_tensor = gym.acquire_actor_root_state_tensor(sim)
root_states = gymtorch.wrap_tensor(root_tensor).view(num_envs, -1).to(device)

# 控制 target tensor 与 actor 索引
dof_targets = torch.zeros((num_envs, act_dim), device=device)
actor_indices = torch.arange(num_envs, dtype=torch.int32, device=device)

# 初始化策略网络
policy = MLPPolicy(obs_dim, act_dim).to(device)
optimizer = optim.Adam(policy.parameters(), lr=lr)

# 启动训练循环
for step in range(total_steps):
    # 获取当前观测
    obs = root_states[:, :obs_dim]

    # 策略输出动作，缩放为控制范围
    action = policy(obs).tanh()
    dof_targets[:] = action * 0.5  # 缩放控制幅度

    # 应用关节目标位置
    gym.set_dof_position_target_tensor_indexed(
        sim,
        gymtorch.unwrap_tensor(dof_targets),
        gymtorch.unwrap_tensor(actor_indices),
        num_envs
    )

    # 仿真并更新张量
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.refresh_actor_root_state_tensor(sim)

    # 奖励函数（保持站稳）
    next_obs = root_states[:, :obs_dim]
    reward = 1.0 - (next_obs[:, 2] - 0.3).abs()  # z 高度接近 0.3 得分高
    loss = -reward.mean()

    # 策略更新（简单的 PG）
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"[Step {step}] Reward mean: {reward.mean().item():.4f}")