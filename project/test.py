from isaacgym import gymapi
gym = gymapi.acquire_gym()

sim_params = gymapi.SimParams()
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

viewer = gym.create_viewer(sim, gymapi.CameraProperties())

if viewer is None:
    print("❌ Failed to create viewer")
else:
    print("✅ Viewer created successfully")

while not gym.query_viewer_has_closed(viewer):
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)