# all sim destroy

from isaacgym import gymapi
from isaacgym import gymutil
import random
import math

gym = gymapi.acquire_gym()

# get default set of parameters
sim_params = gymapi.SimParams()

# set common parameters
sim_params.dt = 1 / 60
sim_params.substeps = 2

# issacgym basic axis -> y axis up
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

# set PhysX-specific parameters
sim_params.physx.use_gpu = True
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 6
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0

# set Flex-specific parameters
# sim_params.flex.solver_type = 5
# sim_params.flex.num_outer_iterations = 4
# sim_params.flex.num_inner_iterations = 20
# sim_params.flex.relaxation = 0.8
# sim_params.flex.warm_start = 0.5

# compute_device_id : GPU selection for simulation
# graphics_device_id : GPU selection for rendering 
compute_device_id = 0
graphics_device_id = 0

# configure the ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
plane_params.distance = 0 # distance of the plane from the origin
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
plane_params.restitution = 0

cam_props = gymapi.CameraProperties()

# set up the env grid
num_envs = 4096
# env_spacing = 2.0
# env_spacing_row = 4.0
# env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing) # -2,0,-2
# env_upper = gymapi.Vec3(env_spacing, env_spacing_row, env_spacing) # 2,2,2

env_lower = gymapi.Vec3(0.0, 0.0, 0.0) # -2,0,-2
env_upper = gymapi.Vec3(0.0, 0.0, 0.0) # 2,2,2

# cache some common handles for later use
envs = []
actor_handles = []

# # create and populate the environments
# for i in range(num_envs):
#     env = gym.create_env(sim, env_lower, env_upper, 1) # last parameter : how many row
#     envs.append(env)

#     # # height = random.uniform(1.0, 2.5)
#     # x_rand = random.uniform(-20.0, 20.0)
#     # y_rand = random.uniform(-20.0, 20.0)

#     # pose = gymapi.Transform()
#     # pose.p = gymapi.Vec3(x_rand, y_rand, 0.0)
#     # # pose.p = gymapi.Vec3(0.0, 0.0, 0.0)

#     # x_rand = random.uniform(-20.0, 20.0)
#     # y_rand = random.uniform(-20.0, 20.0)

#     pose = gymapi.Transform()
#     pose.p = gymapi.Vec3(0.0, 0.0, 0.0)

#     actor_handle = gym.create_actor(env, asset, pose, "robot", i, 1)
#     # Configure DOF properties
#     props = gym.get_actor_dof_properties(env, actor_handle)
#     props["driveMode"] = (gymapi.DOF_MODE_POS)
#     props["stiffness"] = (5000.0, 5000.0)
#     props["damping"] = (100.0, 100.0)
#     gym.set_actor_dof_properties(env, actor_handle, props)
#     # Set DOF drive targets
#     pri_dof_handle = gym.find_actor_dof_handle(env, actor_handle, 'joint1')
#     rev_dof_handle = gym.find_actor_dof_handle(env, actor_handle, 'joint2')
#     gym.set_dof_target_position(env, pri_dof_handle, 0.6)
#     gym.set_dof_target_position(env, rev_dof_handle, 0)
    
#     actor_handles.append(actor_handle)

cnt = 0
dataset = []

class Data:
    def __init__(self, env_handle=None, actor_handle=None, world_x=0, world_y=0):
        self.env_handle = env_handle
        self.actor_handle = actor_handle
        self.world_x = world_x
        self.world_y = world_y
        self.left_yaw_max_0 = 0
        self.right_yaw_max_0 = 0
        self.left_yaw_max_45 = 0
        self.right_yaw_max_45 = 0
        self.left_yaw_max_90 = 0
        self.right_yaw_max_90 = 0
        self.left_yaw_max_135 = 0
        self.right_yaw_max_135 = 0
        self.area_state = 0

while True:

    cnt += 1 # cnt 60 = 1s
    if cnt > 1:
        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

        gym.sync_frame_time(sim)

    if cnt == 1:
        sim = gym.create_sim(compute_device_id, graphics_device_id, gymapi.SIM_PHYSX, sim_params)
        # create the ground plane
        gym.add_ground(sim, plane_params)

        asset_root = "../"
        asset_file = "urdf/rectangular_model.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

        # spacing = 2.0
        # lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        # upper = gymapi.Vec3(spacing, spacing, spacing)

        # env = gym.create_env(sim, lower, upper, 8)

        # pose = gymapi.Transform()
        # pose.p = gymapi.Vec3(0.0, 1.0, 0.0)
        # pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), -0.5 * math.pi)

        # actor_handle = gym.create_actor(env, asset, pose, "MyActor", 0, 1)

        # viewer setting
        viewer = gym.create_viewer(sim, cam_props)

   

    elif cnt == 60: # initialize
        # create and populate the environments
        for i in range(num_envs):
            env = gym.create_env(sim, env_lower, env_upper, 1) # last parameter : how many row
            envs.append(env)

            x_rand = random.uniform(-20.0, 20.0)
            y_rand = random.uniform(-20.0, 20.0)

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(x_rand, y_rand, 0.0)

            actor_handle = gym.create_actor(env, asset, pose, "robot", i, 1)

            data = Data(env, actor_handle, x_rand, y_rand)

            # Configure DOF properties
            props = gym.get_actor_dof_properties(env, actor_handle)
            props["driveMode"] = (gymapi.DOF_MODE_POS)
            props["stiffness"] = (5000.0, 5000.0)
            props["damping"] = (100.0, 100.0)
            gym.set_actor_dof_properties(env, actor_handle, props)
            # Set DOF drive targets
            pri_dof_handle = gym.find_actor_dof_handle(env, actor_handle, 'joint1')
            rev_dof_handle = gym.find_actor_dof_handle(env, actor_handle, 'joint2')
            gym.set_dof_target_position(env, pri_dof_handle, 3)
            gym.set_dof_target_position(env, rev_dof_handle, 0)
            
            actor_handles.append(actor_handle)
            dataset.append(data)

    elif cnt == 120: # get down
        for i in range(len(envs)):
            gym.set_dof_target_position(envs[i], rev_dof_handle, 0)
            gym.set_dof_target_position(envs[i], pri_dof_handle, 0.6)

    elif cnt == 180: # evaluate area state and rotate left
        for i in range(len(envs)):
            body_states = gym.get_actor_rigid_body_states(envs[i], actor_handles[i], gymapi.STATE_ALL)
            body_positions = body_states['pose']['p']
            # print(body_positions[2][2]) # footprint link position z
            if body_positions[2][2] < 1:
                dataset[i].area_state = 1
                gym.set_dof_target_position(envs[i], rev_dof_handle, 3.10)
            else:
                dataset[i].area_state = 0

    elif cnt == 240: # save left_yaw_max
        for data in dataset:
            if data.area_state == 1:
                dof_states = gym.get_actor_dof_states(data.env_handle, data.actor_handle, gymapi.STATE_ALL)
                # print(dof_positions[0]) # prismatic joint position
                # print(dof_positions[1]) # revolute joint position
                dof_positions = dof_states['pos']
                data.left_yaw_max_0 = dof_positions[1]

    elif cnt == 300: # rotate right
        for data in dataset:
            if data.area_state == 1:
                gym.set_dof_target_position(data.env_handle, rev_dof_handle, -3.10)

    elif cnt == 360:
        for data in dataset:
            if data.area_state == 1:
                for_states = gym.get_actor_dof_states(data.env_handle, data.actor_handle, gymapi.STATE_ALL)
                dof_positions = dof_states['pos']
                data.right_yaw_max_0 = dof_positions[1]
        cnt = 0
        envs.clear()
        actor_handles.clear()
        dataset.clear()
        gym.destroy_sim(sim)
        gym.destroy_viewer(viewer)
