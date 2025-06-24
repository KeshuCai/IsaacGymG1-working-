from isaacgym import gymapi, gymtorch
import torch
import os

# ------------------------
# 初始化 Isaac Gym 接口
# ------------------------
gym = gymapi.acquire_gym()
print("[Init] Isaac Gym acquired")

# ------------------------
# 设置仿真参数
# ------------------------
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.use_gpu_pipeline = True  # ✅ 启用 GPU 物理仿真和 Tensor 流水线
print("[Config] use_gpu_pipeline set to True")

# ------------------------
# 创建仿真（使用 GPU）
# ------------------------
compute_device_id = 0
graphics_device_id = 0

sim = gym.create_sim(
    compute_device_id,
    graphics_device_id,
    gymapi.SimType.SIM_PHYSX,
    sim_params
)
assert sim is not None, "Failed to create sim"
print("[Sim] Created on GPU")

# ------------------------
# 添加地面
# ------------------------
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)
print("[World] Ground plane added")

# ------------------------
# 加载机器人模型（MJCF）
# ------------------------
asset_root = "/home/kc/IsaacGymEnvs/assets/mjcf/g1_description"  # 替换为你的 MJCF 文件路径
asset_file = "g1_29dof_with_hand.xml"      # 替换为你的 MJCF 文件名

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True

asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
assert asset is not None, "Failed to load asset"
print(f"[Asset] Loaded {asset_file}")

# ------------------------
# 创建多个环境 + 机器人
# ------------------------
num_envs = 16
spacing = 2.0
envs = []
actors = []

for i in range(num_envs):
    env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)

    env = gym.create_env(sim, env_lower, env_upper, num_envs)
    envs.append(env)

    x = (i % 4) * spacing
    y = (i // 4) * spacing
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(x, y, 0.3)  # 抬高以避免穿地

    actor_handle = gym.create_actor(env, asset, pose, f"Robot_{i}", i, 1)
    actors.append(actor_handle)

print(f"[Env] Created {num_envs} envs with robots")

# ------------------------
# 准备 Tensor API（关键）
# ------------------------
gym.prepare_sim(sim)
print("[Tensor] Tensor API prepared")

# 获取 root state tensor（包括位置/速度等）
root_tensor = gym.acquire_actor_root_state_tensor(sim)
root_states = gymtorch.wrap_tensor(root_tensor).view(num_envs, -1).to("cuda")
print("[Tensor] Root state tensor device:", root_states.device)

# ------------------------
# 创建 Viewer 可视化窗口
# ------------------------
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
assert viewer is not None, "Failed to create viewer"

# 设置初始相机位置
cam_pos = gymapi.Vec3(5.0, 5.0, 5.0)
cam_target = gymapi.Vec3(4.0, 4.0, 0.0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# ------------------------
# 仿真主循环
# ------------------------
step_count = 0
while not gym.query_viewer_has_closed(viewer):
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # 更新 root state 张量
    gym.refresh_actor_root_state_tensor(sim)
    root_states = gymtorch.wrap_tensor(root_tensor).view(num_envs, -1).to("cuda")

    # 每 20 步打印一次第一个机器人的位置
    if step_count % 20 == 0:
        pos = root_states[0, 0:3]
        print(f"[Step {step_count}] Robot_0 position: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")

    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)
    step_count += 1

# ------------------------
# 清理资源
# ------------------------
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
print("[Done] Viewer closed, sim destroyed")