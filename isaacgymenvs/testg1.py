
import numpy as np
from isaacgym import gymapi, gymtorch
from isaacgymenvs.tasks.g1_robot import G1Robot
import torch
# 配置字典
cfg = {
    "physics_engine": "physx",
    "env": {
        "numEnvs": 4,
        "episodeLength": 500,
        "envSpacing": 2.0,
        "numActions": 41,
        "asset": {
            "assetRoot": "/home/kc/IsaacGymEnvs/assets",
            "assetFileName": "mjcf/g1_description/g1_29dof_lock_waist_with_hand_rev_1_0.xml"
        }
    },
    "sim": {
        "gravity": [0.0, 0.0, -9.81],
        "up_axis": "z",
        "dt": 1 / 60,
        "substeps": 2,
        "use_gpu_pipeline": True,
        "physx": {
            "use_gpu": True
        }
    }
}

# 初始化 G1Robot 环境
env = G1Robot(
    cfg=cfg,
    sim_device="cuda:0",
    graphics_device_id=0,
    headless=False,
    force_render=True
)

# 确保必要的张量初始化完成
if hasattr(env, "post_create_sim"):
    env.post_create_sim()

print("✅ Environment and viewer initialized.")

# 无限循环测试 Viewer 刷新与动作施加
try:
    while True:
        dummy_actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
        env.pre_physics_step(dummy_actions)
        env.post_physics_step()
except KeyboardInterrupt:
    print("👋 Viewer exited manually.")
finally:
    if env.viewer is not None:
        env.gym.destroy_viewer(env.viewer)
    if env.sim is not None:
        env.gym.destroy_sim(env.sim)