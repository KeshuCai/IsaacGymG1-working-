import mujoco
from pathlib import Path

# 替换为你自己的 XML 路径
XML_PATH = Path("/home/kc/IsaacGymEnvs/assets/mjcf/g1_description/g1_29dof_lock_waist_with_hand_rev_1_0.xml")  # ← 请替换成你的文件路径

# 加载模型
model = mujoco.MjModel.from_xml_path(str(XML_PATH))
print(f"✅ 成功加载模型：{XML_PATH.name}")
print(f"🔧 DOF 总数: {model.nv}, actuator 数: {model.nu}")
print("=" * 60)

# 打印每个关节（DOF）信息
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    joint_type = model.jnt_type[i]
    joint_range = model.jnt_range[i]

    joint_type_str = {
        mujoco.mjtJoint.mjJNT_FREE: "FREE",
        mujoco.mjtJoint.mjJNT_BALL: "BALL",
        mujoco.mjtJoint.mjJNT_SLIDE: "SLIDE",
        mujoco.mjtJoint.mjJNT_HINGE: "HINGE"
    }.get(joint_type, "UNKNOWN")

    print(f"[Joint {i:02d}] Name: {name}, Type: {joint_type_str}, Range: {joint_range[0]:.3f} to {joint_range[1]:.3f}")

print("=" * 60)

# 打印 actuator 信息（有 actuator 才能被控制）
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    trnid = model.actuator_trnid[i]
    joint_id = trnid[0]
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    gainprm = model.actuator_gainprm[i]
    biasprm = model.actuator_biasprm[i]

    print(f"[Actuator {i:02d}] Name: {name}, Controls joint: {joint_name}")
    print(f"    Gain: {gainprm[:3]}, Bias: {biasprm[:3]}")

print("=" * 60)

