from isaacgym import gymapi

# 初始化 gym
gym = gymapi.acquire_gym()

# 仿真参数
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.use_gpu_pipeline = True
sim = gym.create_sim(0, 0, gymapi.SimType.SIM_PHYSX, sim_params)
assert sim is not None, "Failed to create sim"

# 加载 MJCF 模型（修改为你自己的路径）
asset_root = "/home/kc/IsaacGymEnvs/assets/mjcf/g1_description"
asset_file = "g1_29dof_with_hand.xml"
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True

asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
assert asset is not None, "Failed to load asset"

# 打印 DOF 信息
dof_count = gym.get_asset_dof_count(asset)
print(f"Total DOFs: {dof_count}\n")

for i in range(dof_count):
    dof_name = gym.get_asset_dof_name(asset, i)
    dof_type = gym.get_asset_dof_type(asset, i)
    dof_type_str = ["POS", "VEL", "EFFORT"][int(dof_type)] if int(dof_type) < 3 else "UNKNOWN"
    dof_props = gym.get_asset_dof_properties(asset)[i]
    print(f"[{i}] DOF: {dof_name} | Type: {dof_type_str} | Limit: ({dof_props['lower']}, {dof_props['upper']})")

# 打印关节（joint）信息
joint_count = gym.get_asset_joint_count(asset)
print(f"\nTotal joints: {joint_count}")
for i in range(joint_count):
    joint_name = gym.get_asset_joint_name(asset, i)
    joint_type = gym.get_asset_joint_type(asset, i)
    print(f"[{i}] Joint: {joint_name} | Type: {joint_type}")

# 打印链接（link）信息
body_count = gym.get_asset_rigid_body_count(asset)
print(f"\nTotal rigid bodies: {body_count}")
for i in range(body_count):
    body_name = gym.get_asset_rigid_body_name(asset, i)
    print(f"[{i}] Rigid body: {body_name}")