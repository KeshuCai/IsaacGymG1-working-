# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgymenvs.utils.torch_jit_utils import scale, unscale, quat_mul, quat_conjugate, quat_from_angle_axis, \
    to_torch, get_axis_params, torch_rand_float, tensor_clamp, compute_heading_and_up, compute_rot, normalize_angle

from isaacgymenvs.tasks.base.vec_task import VecTask


class G1Robot(VecTask):

    def __init__(self,
                 cfg,
                 rl_device,
                 sim_device,
                 graphics_device_id,
                 headless,
                 virtual_screen_capture,
                 force_render):
        # ——— 参数配置 ——————————————————————————————
        self.debug_viz = cfg["env"].get("enableDebugVis", False)
        self.cfg                          = cfg
        self.randomization_params         = cfg["task"]["randomization_params"]
        self.randomize                    = cfg["task"]["randomize"]
        self.dof_vel_scale                = cfg["env"]["dofVelocityScale"]
        self.angular_velocity_scale       = cfg["env"].get("angularVelocityScale", 0.1)
        self.contact_force_scale          = cfg["env"]["contactForceScale"]
        self.power_scale                  = cfg["env"]["powerScale"]
        self.heading_weight               = cfg["env"]["headingWeight"]
        self.up_weight                    = cfg["env"]["upWeight"]
        self.actions_cost_scale           = cfg["env"]["actionsCost"]
        self.energy_cost_scale            = cfg["env"]["energyCost"]
        self.joints_at_limit_cost_scale   = cfg["env"]["jointsAtLimitCost"]
        self.death_cost                   = cfg["env"]["deathCost"]
        self.termination_height           = cfg["env"]["terminationHeight"]
        plane = cfg["env"]["plane"]
        self.plane_static_friction        = plane["staticFriction"]
        self.plane_dynamic_friction       = plane["dynamicFriction"]
        self.plane_restitution            = plane["restitution"]
        self.max_episode_length           = cfg["env"]["episodeLength"]
        # 强制指定 obs/action 维度
        cfg["env"]["numObservations"]     = 188
        cfg["env"]["numActions"]          = 41
        
        
        # ——— 父类初始化 ——————————————————————————————
        super().__init__(config=self.cfg,
                         rl_device=rl_device,
                         sim_device=sim_device,
                         graphics_device_id=graphics_device_id,
                         headless=headless,
                         virtual_screen_capture=virtual_screen_capture,
                         force_render=force_render)

        # 可视化摄像机（可选）
        if self.viewer is not None:
            cam_pos    = gymapi.Vec3(10.0, -5.0, 2.4)
            cam_target = gymapi.Vec3(0.0, 0.0, 1.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # ——— 获取并包装 GPU 张量 ——————————————————————————————
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)

        # 刚体力传感器：左右脚各 1，共 2×6=12
        sensor_tensor   = self.gym.acquire_force_sensor_tensor(self.sim)
        sensors_per_env = 2
        if sensor_tensor is None:
            # 如果没创建或无效，就退回零张量
            self.vec_sensor_tensor = torch.zeros(
                self.num_envs, sensors_per_env * 6,
                device=self.device, dtype=torch.float32
            )
        else:
            self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor)\
                                         .view(self.num_envs, sensors_per_env * 6)

        # 关节扭矩张量
        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor)\
                                         .view(self.num_envs, self.num_dof)

        # 刷新并保存 root/dof 状态
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.root_states         = gymtorch.wrap_tensor(actor_root_state)
        self.initial_root_states = self.root_states.clone()
        # 清除初始旋转速度
        self.initial_root_states[:, 7:13] = 0

        # 拆分 dof_pos/dof_vel
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos   = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel   = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        # 初始关节位置（置于上下限或 0）
        self.initial_dof_pos = torch.zeros_like(self.dof_pos, device=self.device)
        zero_tensor = torch.tensor([0.0], device=self.device)
        self.initial_dof_pos = torch.where(
            self.dof_limits_lower > zero_tensor,
            self.dof_limits_lower,
            torch.where(self.dof_limits_upper < zero_tensor,
                        self.dof_limits_upper,
                        self.initial_dof_pos)
        )
        self.initial_dof_vel = torch.zeros_like(self.dof_vel, device=self.device)

        # 后续 observations 构造所需常量
        self.up_vec          = to_torch(get_axis_params(1., self.up_axis_idx),
                                        device=self.device).repeat(self.num_envs, 1)
        self.heading_vec     = to_torch([1, 0, 0], device=self.device)\
                                        .repeat(self.num_envs, 1)
        self.inv_start_rot   = quat_conjugate(self.start_rotation)\
                                        .repeat(self.num_envs, 1)
        self.basis_vec0      = self.heading_vec.clone()
        self.basis_vec1      = self.up_vec.clone()
        self.targets         = to_torch([1000, 0, 0], device=self.device)\
                                        .repeat(self.num_envs, 1)
        self.target_dirs     = to_torch([1, 0, 0], device=self.device)\
                                        .repeat(self.num_envs, 1)
        self.dt              = self.cfg["sim"]["dt"]
        self.potentials      = to_torch([-1000. / self.dt], device=self.device)\
                                        .repeat(self.num_envs)
        self.prev_potentials = self.potentials.clone()
        
    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        # ——— 计算环境阵列范围 ——————————————————————————————
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3( spacing,  spacing, spacing)

        # ——— 资产路径 ——————————————————————————————————————
        asset_root = os.path.join(os.path.dirname(__file__), '../../assets')
        asset_file = "mjcf/g1_description/g1_29dof_lock_waist_with_hand_rev_1_0.xml"
        if "asset" in self.cfg["env"]:
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)
        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        # ——— 加载资产 ——————————————————————————————————————
        options = gymapi.AssetOptions()
        options.angular_damping        = 0.01
        options.max_angular_velocity   = 100.0
        options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, options)

        # ——— 读取 actuator 参数 ——————————————————————————————
        props = self.gym.get_asset_actuator_properties(humanoid_asset)
        motor_efforts = [p.motor_effort for p in props]
        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts    = to_torch(motor_efforts, device=self.device)

        # ——— 创建刚体力传感器 —————————————————————————————
        sensor_pose = gymapi.Transform()
        right_idx   = self.gym.find_asset_rigid_body_index(humanoid_asset, "right_ankle_roll_link")
        left_idx    = self.gym.find_asset_rigid_body_index(humanoid_asset, "left_ankle_roll_link")
        self.gym.create_asset_force_sensor(humanoid_asset, right_idx, sensor_pose)
        self.gym.create_asset_force_sensor(humanoid_asset, left_idx,  sensor_pose)

        # ——— 保存 DoF/Body 数量 ——————————————————————————————
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_dof    = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(humanoid_asset)

        # ——— 初始放置 ——————————————————————————————————————
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*get_axis_params(0.7, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self.start_rotation = torch.tensor(
            [start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w],
            device=self.device
        )

        # ——— 创建各环境实例 ——————————————————————————————————
        self.humanoid_handles = []
        self.envs             = []
        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            handle  = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", i, 0, 0)
            self.gym.enable_actor_dof_force_sensors(env_ptr, handle)
            # 可视化上色（可选）
            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(env_ptr, handle, j,
                                              gymapi.MESH_VISUAL,
                                              gymapi.Vec3(0.97, 0.38, 0.06))
            self.envs.append(env_ptr)
            self.humanoid_handles.append(handle)

        # ——— 读取并保存关节限位 —————————————————————————————
        dof_prop = self.gym.get_actor_dof_properties(env_ptr, handle)
        lowers, uppers = [], []
        for j in range(self.num_dof):
            low, high = dof_prop['lower'][j], dof_prop['upper'][j]
            if low > high:
                lowers.append(high); uppers.append(low)
            else:
                lowers.append(low);   uppers.append(high)
        self.dof_limits_lower = to_torch(lowers, device=self.device)
        self.dof_limits_upper = to_torch(uppers, device=self.device)

        # ——— 新增：对锁死关节做安全扩展 ——————————————————————
        # 标记锁死关节（upper == lower）
        self.locked_dofs = (self.dof_limits_upper == self.dof_limits_lower)
        eps = 1e-6  # 极小值
        # 构造 safe_lower 与 safe_upper，用于后续归一化时避免除零
        self.safe_lower = self.dof_limits_lower.clone()
        self.safe_upper = self.dof_limits_upper.clone()
        self.safe_upper[self.locked_dofs] += eps

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf = compute_humanoid_reward(
            self.obs_buf,
            self.reset_buf,
            self.progress_buf,
            self.actions,
            self.up_weight,
            self.heading_weight,
            self.potentials,
            self.prev_potentials,
            self.actions_cost_scale,
            self.energy_cost_scale,
            self.joints_at_limit_cost_scale,
            self.max_motor_effort,
            self.motor_efforts,
            self.termination_height,
            self.death_cost,
            self.max_episode_length
        )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)

        self.gym.refresh_dof_force_tensor(self.sim)
        self.obs_buf[:], self.potentials[:], self.prev_potentials[:], self.up_vec[:], self.heading_vec[:] = compute_humanoid_observations(
            self.obs_buf, self.root_states, self.targets, self.potentials,
            self.inv_start_rot, self.dof_pos, self.dof_vel, self.dof_force_tensor,
            self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
            self.vec_sensor_tensor, self.actions, self.dt, self.contact_force_scale, self.angular_velocity_scale,
            self.basis_vec0, self.basis_vec1)

    def reset_idx(self, env_ids):
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # 1) 为所选 env 采样初始位置和速度扰动
        positions = torch_rand_float(
            -0.2, 0.2,
            (len(env_ids), self.num_dof),
            device=self.device
        )
        velocities = torch_rand_float(
            -0.1, 0.1,
            (len(env_ids), self.num_dof),
            device=self.device
        )

        # 2) 更新 dof_pos / dof_vel
        self.dof_pos[env_ids] = tensor_clamp(
            self.initial_dof_pos[env_ids] + positions,
            self.dof_limits_lower, self.dof_limits_upper
        )
        self.dof_vel[env_ids] = velocities

        # 3) 检查数值合法性
        assert torch.all(torch.isfinite(self.dof_pos[env_ids])), "❌ dof_pos has NaNs!"
        assert torch.all(torch.isfinite(self.dof_vel[env_ids])), "❌ dof_vel has NaNs!"

        # 4) 重置根状态
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        new_roots = self.initial_root_states[env_ids].clone()
        assert torch.all(torch.isfinite(new_roots)), "❌ initial_root_states has NaNs!"
        self.root_states[env_ids] = new_roots
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )

        # 5) 正确地把 dof_pos 和 dof_vel 写回到 self.dof_state（shape = [num_envs, num_dof, 2]）
        #    ——— 注意索引第 2 维 channel，而不是第 1 维 DOF
        dof_state_reshaped = self.dof_state.view(self.num_envs, self.num_dof, 2)
        # channel 0 存 position
        dof_state_reshaped[:, :, 0] = self.dof_pos
        # channel 1 存 velocity
        dof_state_reshaped[:, :, 1] = self.dof_vel
        # （self.dof_state 本身因此也被更新）

        # 6) 将更新后的 dof_state 写回到物理引擎
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )

        # 7) 重置潜在、进度等 buffers
        to_target = self.targets[env_ids] - new_roots[:, 0:3]
        to_target[:, self.up_axis_idx] = 0.0
        self.prev_potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.dt
        self.potentials[env_ids] = self.prev_potentials[env_ids].clone()

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

        # 8) 清空上一帧动作，防止未定义
        self.actions[env_ids] = torch.zeros(
            (len(env_ids), self.num_actions),
            device=self.device
        )

        # （可选）调试输出
        #print(f"[✅ reset_idx] Finished resetting envs: {env_ids.tolist()}")

    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()

        if not torch.all(torch.isfinite(self.actions)):
            print("🚨 Actions contain NaN! Sample:", self.actions[0])
            print("==> Shape:", self.actions.shape)
            raise ValueError("Actions contain NaN!")

        forces = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
        force_tensor = gymtorch.unwrap_tensor(forces)
        self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)
    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)
        root_pos = self.root_states[0, :3]  # Tensor([x, y, z])
        
        self.gym.refresh_actor_root_state_tensor(self.sim)
        # 重新 wrap 指针
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        root_states = gymtorch.wrap_tensor(actor_root_state)

        # 打印 Env0 的根位置
        x, y, z = root_states[0, 0].item(), root_states[0, 1].item(), root_states[0, 2].item()
        print(f"[DEBUG] Env0 root position: x={x:.3f}, y={y:.3f}, z={z:.3f}")
        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)

            points = []
            colors = []
            for i in range(self.num_envs):
                origin = self.gym.get_env_origin(self.envs[i])
                pose = self.root_states[:, 0:3][i].cpu().numpy()
                glob_pos = gymapi.Vec3(origin.x + pose[0], origin.y + pose[1], origin.z + pose[2])
                points.append([glob_pos.x, glob_pos.y, glob_pos.z, glob_pos.x + 4 * self.heading_vec[i, 0].cpu().numpy(),
                               glob_pos.y + 4 * self.heading_vec[i, 1].cpu().numpy(),
                               glob_pos.z + 4 * self.heading_vec[i, 2].cpu().numpy()])
                colors.append([0.97, 0.1, 0.06])
                points.append([glob_pos.x, glob_pos.y, glob_pos.z, glob_pos.x + 4 * self.up_vec[i, 0].cpu().numpy(), glob_pos.y + 4 * self.up_vec[i, 1].cpu().numpy(),
                               glob_pos.z + 4 * self.up_vec[i, 2].cpu().numpy()])
                colors.append([0.05, 0.99, 0.04])

            self.gym.add_lines(self.viewer, None, self.num_envs * 2, points, colors)

#####################################################################
###=========================jit functions=========================###
#####################################################################

from torch import Tensor
from typing import Tuple
@torch.jit.script
def compute_humanoid_reward(
    obs_buf:          Tensor,
    reset_buf:        Tensor,
    progress_buf:     Tensor,
    actions:          Tensor,
    up_weight:        float,
    heading_weight:   float,
    potentials:       Tensor,
    prev_potentials:  Tensor,
    actions_cost_scale:      float,
    energy_cost_scale:       float,
    joints_at_limit_cost_scale: float,
    max_motor_effort:        float,
    motor_efforts:           Tensor,
    termination_height:      float,
    death_cost:              float,
    max_episode_length:      int
) -> Tuple[Tensor, Tensor]:
    # 自由度数目 = motor_efforts 长度
    num_dof = motor_efforts.size(0)
    base   = 1 + 3 + 3 + 5  # torso_z + vel_loc(3) + angvel(3) + [yaw,roll,angle,up,heading]
    start  = base  # 12

    # 切出 pos, vel, force（按 obs_buf 中的拼接顺序）
    pos   = obs_buf[:, start              : start + num_dof]
    vel   = obs_buf[:, start + num_dof    : start + 2 * num_dof]
    force = obs_buf[:, start + 2*num_dof  : start + 3 * num_dof]

    # —— 原版的动作成本 & 极限成本 ——
    heading_w  = torch.ones_like(obs_buf[:, 11]) * heading_weight
    heading_r  = torch.where(obs_buf[:, 11] > 0.8,
                              heading_w,
                              heading_weight * obs_buf[:, 11] / 0.8)
    up_r       = torch.where(obs_buf[:, 10] > 0.93,
                              torch.ones_like(heading_r) * up_weight,
                              torch.zeros_like(heading_r))
    actions_cost = torch.sum(actions ** 2, dim=-1)

    # 关节极限惩罚
    scaled_cost = joints_at_limit_cost_scale * (torch.abs(pos) - 0.98) / 0.02
    dof_limit_cost = torch.sum(
        (torch.abs(pos) > 0.98) * scaled_cost * (motor_efforts / max_motor_effort).unsqueeze(0),
        dim=-1
    )

    # 能耗惩罚（action × force）
    electricity_cost = torch.sum(
        torch.abs(actions * force) * (motor_efforts / max_motor_effort).unsqueeze(0),
        dim=-1
    )

    alive_r    = torch.ones_like(potentials) * 2.0
    prog_r     = potentials - prev_potentials

    total = prog_r + alive_r + up_r + heading_r \
            - actions_cost_scale * actions_cost \
            - energy_cost_scale  * electricity_cost \
            - dof_limit_cost

    # 掉落或超时重置
    total = torch.where(obs_buf[:, 0] < termination_height,
                         torch.ones_like(total) * death_cost,
                         total)
    reset = torch.where(obs_buf[:, 0] < termination_height,
                        torch.ones_like(reset_buf),
                        reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1,
                        torch.ones_like(reset_buf),
                        reset)
    return total, reset

@torch.jit.script
def compute_humanoid_observations(
    obs_buf: Tensor,
    root_states: Tensor,
    targets: Tensor,
    potentials: Tensor,
    inv_start_rot: Tensor,
    dof_pos: Tensor,
    dof_vel: Tensor,
    dof_force: Tensor,
    dof_limits_lower: Tensor,
    dof_limits_upper: Tensor,
    dof_vel_scale: float,
    sensor_force_torques: Tensor,
    actions: Tensor,
    dt: float,
    contact_force_scale: float,
    angular_velocity_scale: float,
    basis_vec0: Tensor,
    basis_vec1: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    已集成“方案一”安全扩展：
    动态计算 safe_lower、safe_upper、locked_dofs，
    对锁死关节归一化为 0，避免 inf/NaN。
    """
    # ——— 躯干状态与速度 —————————————————————————————
    torso_position = root_states[:, 0:3]
    torso_rotation = root_states[:, 3:7]
    velocity       = root_states[:, 7:10]
    ang_velocity   = root_states[:, 10:13]

    # ——— 势能更新 —————————————————————————————————————
    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    prev_potentials = potentials.clone()
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    # ——— 朝向与投影 ————————————————————————————————————
    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2
    )
    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position
    )
    roll            = normalize_angle(roll).unsqueeze(-1)
    yaw             = normalize_angle(yaw).unsqueeze(-1)
    angle_to_target = normalize_angle(angle_to_target).unsqueeze(-1)

    # ——— 方案一：动态构造 safe_lower/safe_upper/locked_dofs ——
    locked_dofs = dof_limits_upper == dof_limits_lower
    eps: float   = 1e-6
    safe_lower  = dof_limits_lower
    # 对锁死关节仅扩展上限
    safe_upper  = dof_limits_upper + locked_dofs.to(dtype=dof_limits_upper.dtype) * eps

    # 钳制再归一化
    dof_pos_clamped = torch.max(
        torch.min(dof_pos, safe_upper.unsqueeze(0)),
        safe_lower.unsqueeze(0)
    )
    dof_pos_scaled = unscale(dof_pos_clamped, safe_lower, safe_upper)
    # 锁死关节显式置 0
    dof_pos_scaled = torch.where(
        locked_dofs.unsqueeze(0),
        torch.zeros_like(dof_pos_scaled),
        dof_pos_scaled
    )

    # ——— 传感器补零 & NaN/Inf 检测 —————————————————————
    if sensor_force_torques is None:
        sensor_force_torques = torch.zeros((dof_pos.shape[0], 0), device=dof_pos.device)
    # （可按需保留原有打印与 nan_to_num）

    # ——— 拼接 observation —————————————————————————————————
    obs = torch.cat((
        torso_position[:, 2:3],
        vel_loc,
        angvel_loc * angular_velocity_scale,
        yaw, roll, angle_to_target,
        up_proj.unsqueeze(-1), heading_proj.unsqueeze(-1),
        dof_pos_scaled,
        dof_vel * dof_vel_scale,
        dof_force * contact_force_scale,
        sensor_force_torques * contact_force_scale,
        actions
    ), dim=-1)

    return obs, potentials, prev_potentials, up_vec, heading_vec



