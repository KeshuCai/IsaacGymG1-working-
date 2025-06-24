import os

import numpy as np
from isaacgym import gymapi, gymtorch, gymutil
import torch
from torch import nn, optim

# ========== 配置 ==========
num_envs = 16
num_obs = 86  # 示例：qpos(43) + qvel(43)
num_act = 43  # DOF 数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 初始化 gym ==========
gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim_params.physx.use_gpu = True
sim_params.use_gpu_pipeline = True
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

asset_root = "/home/kc/IsaacGymEnvs/assets/mjcf/g1_description"  
asset_file = "g1_29dof_with_hand.xml"
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False
asset_options.disable_gravity = False
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_VEL
asset_options.use_mesh_materials = True
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

envs = []
actor_handles = []
env_spacing = 2.0

for i in range(num_envs):
    env = gym.create_env(sim, gymapi.Vec3(-env_spacing, 0, 0), gymapi.Vec3(env_spacing, env_spacing, env_spacing), num_envs)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
    actor_handle = gym.create_actor(env, asset, pose, f"g1_{i}", i, 0)
    envs.append(env)
    actor_handles.append(actor_handle)

dof_state_tensor = gym.acquire_dof_state_tensor(sim)
dof_state = gymtorch.wrap_tensor(dof_state_tensor).view(num_envs, -1, 2).to(device)

# ========== PPO 网络 ==========
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
            nn.Tanh(),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs):
        return self.actor(obs), self.critic(obs)

policy = ActorCritic(num_obs, num_act).to(device)
optimizer = optim.Adam(policy.parameters(), lr=3e-4)

# ========== 训练循环 ==========
for step in range(10000):
    # 构造 observation（这里只是占位示例，替换为你的真实 obs）
    obs = torch.cat([dof_state[:, :, 0], dof_state[:, :, 1]], dim=-1)

    # 采样动作
    with torch.no_grad():
        action, _ = policy(obs)
    action = action.cpu().numpy()

    # 设置目标速度
    vel_target_tensor = gym.acquire_dof_target_tensor(sim)
    vel_target = gymtorch.wrap_tensor(vel_target_tensor).view(num_envs, num_act).to(device)

    # action 是 (num_envs, num_dofs) 的 torch tensor
    vel_target[:] = action

    # 步进仿真
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.refresh_dof_state_tensor(sim)

    # 假设构造 reward
    reward = -torch.norm(dof_state[:, :, 0] - 0.0, dim=-1).mean()

    # 反向传播损失（这里使用简单的 value baseline loss 作为 PPO 占位）
    _, value = policy(obs)
    loss = (value.squeeze() - reward) ** 2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"[Step {step}] Loss: {loss.item():.4f}, Avg reward: {reward.item():.4f}")
