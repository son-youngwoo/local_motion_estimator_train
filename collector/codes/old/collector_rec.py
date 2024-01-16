from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
import random
import math
from terrain_utils import *
from PIL import Image
import csv

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
sim = gym.create_sim(compute_device_id, graphics_device_id, gymapi.SIM_PHYSX, sim_params)

# create terrain ##############################################################################33

class SubTerrain:
    def __init__(self, terrain_name="terrain", width=256, length=256, vertical_scale=1.0, horizontal_scale=1.0):
        self.terrain_name = terrain_name
        self.vertical_scale = vertical_scale
        self.horizontal_scale = horizontal_scale
        self.width = width
        self.length = length
        self.height_field_raw = np.zeros((self.width, self.length), dtype=np.int16)

# create all available terrain types
terrain_width = 20.
terrain_length = 20.
horizontal_scale = 0.02  # [m]
vertical_scale = 0.1  # [m]
num_rows = int(terrain_width/horizontal_scale)
num_cols = int(terrain_length/horizontal_scale)
heightfield = np.zeros((num_rows, num_cols), dtype=np.int16)

def new_sub_terrain(): return SubTerrain(width=num_rows, length=num_cols, vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)

num_obs = 135
pillar_height = 1.0 # [m]
pillar_size = 0.5 # [m]

# square pillar
heightfield = random_square_pillar(new_sub_terrain(), num_obs, pillar_height, pillar_size).height_field_raw

# circular pillar
# heightfield = random_circular_pillar(new_sub_terrain(), num_obs, pillar_height, pillar_size).height_field_raw

# add the terrain as a triangle mesh
vertices, triangles = convert_heightfield_to_trimesh(heightfield, horizontal_scale=horizontal_scale, vertical_scale=vertical_scale, slope_threshold=1.5)
tm_params = gymapi.TriangleMeshParams()
tm_params.nb_vertices = vertices.shape[0]
tm_params.nb_triangles = triangles.shape[0]
tm_params.transform.p.x = 0.
tm_params.transform.p.y = 0.
gym.add_triangle_mesh(sim, vertices.flatten(), triangles.flatten(), tm_params)

##############################################################################

# ground plane ####

# #configure the ground plane
# plane_params = gymapi.PlaneParams()
# plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
# plane_params.distance = 0 # distance of the plane from the origin
# plane_params.static_friction = 1
# plane_params.dynamic_friction = 1
# plane_params.restitution = 0

# # create the ground plane
# gym.add_ground(sim, plane_params)

# ground plane ####

asset_root = "../"
asset_file = "urdf/rec.urdf"

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# viewer setting
cam_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, cam_props)

env_lower = gymapi.Vec3(0.0, 0.0, 0.0) # -2,0,-2
env_upper = gymapi.Vec3(0.0, 0.0, 0.0) # 2,2,2

# cache some common handles for later use
envs = []
actor_handles = []


## common ##

# set up the env grid
num_envs = 4096

cnt = 0
dataset = []

######################

## respawn ##

process_flag = 1
step = 1 # [deg]
init_pose = list(range(0,180,step)) # [deg]
iteration = 0
init_z = 1.5 # [m]
target_z = 1.0 # [m]
respawn_offset = 60 # [p]

######################

## map processing ##

robot_z = 0.6 # [m]
local_map_size = 2 # [m]
height_max = 0 # [m]
height_min = -0.6 # [m]
image_size = 100 # [p]

######################

start1 = 0
end1 = 0

class Data:

    def __init__(self, _id=0, _env_handle = None, _actor_handle = None, _x_world=0, _y_world=0):
        self.id = _id
        self.env_handle = _env_handle
        self.actor_handle = _actor_handle
        self.x_world = _x_world 
        self.y_world = _y_world 
        self.rotatable_area = []

    def GetElevationMap(_data, _heightfield):
        x = int(_data.x_world / horizontal_scale) # [p]
        y = int(_data.y_world / horizontal_scale) # [p]
        
        x1 = int(x - image_size / 2) # [p]
        x2 = int(x + image_size / 2) # [p]
        y1 = int(y - image_size / 2) # [p]
        y2 = int(y + image_size / 2) # [p]

        local_map_array = _heightfield[x1:x2, y1:y2] - robot_z
        local_map_array = (255 * ( local_map_array - height_min ) / (height_max - height_min)).astype(np.uint8)

        local_map_image = Image.fromarray(local_map_array)

        save_path = "../../RoA_Planner_dataset/test/image/" + str(_data.id) + ".png"
        local_map_image.save(save_path)

    def getGlobalMap(_heightfield):
        global_map_name = "square"

        global_map_array = _heightfield - robot_z
        global_map_array = (255 * ( global_map_array - height_min ) / (height_max - height_min)).astype(np.uint8)
        
        global_map_image = Image.fromarray(global_map_array)
        
        save_path = "../../RoA_Planner/global_map/" + global_map_name + ".png"
        global_map_image.save(save_path)

    def GetLabel(_data):
        # RotatableAreaList = []
        # for _rotatable_area in _data.rotatable_area:
        #     RotatableAreaList.append(_rotatable_area)
        # data = [_data.id, RotatableAreaList]
        # file_path = '../../RoA_Planner_dataset/square/square.csv'
        # with open(file_path, 'a', newline='') as csvfile:
        #     csv_writer = csv.writer(csvfile)
        #     csv_writer.writerow(data)
        
        data = [_data.id]
        
        for _rotatable_area in _data.rotatable_area:
            data.append(_rotatable_area)
        file_path = '../../RoA_Planner_dataset/test/square.csv'
        with open(file_path, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(data)

Data.getGlobalMap(heightfield)

# create and populate the environments
for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, 1) # last parameter : how many row
    envs.append(env)
    
    rand_pos = (random.randint(0 + respawn_offset, num_rows - respawn_offset)*horizontal_scale, random.randint(0 + respawn_offset, num_cols - respawn_offset)*horizontal_scale) # [m]

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(rand_pos[0], rand_pos[1], init_z)

    actor_handle = gym.create_actor(env, asset, pose, "robot", i, 1)
    # Configure DOF properties
    props = gym.get_actor_dof_properties(env, actor_handle)
    props["driveMode"] = (gymapi.DOF_MODE_POS)
    props["stiffness"] = (5000.0, 5000.0)
    props["damping"] = (100.0, 100.0)
    gym.set_actor_dof_properties(env, actor_handle, props)
    # Set DOF drive targets
    pri_dof_handle = gym.find_actor_dof_handle(env, actor_handle, 'joint1')
    rev_dof_handle = gym.find_actor_dof_handle(env, actor_handle, 'joint2')
    gym.set_dof_target_position(env, pri_dof_handle, 0)
    gym.set_dof_target_position(env, rev_dof_handle, 0)
    
    actor_handles.append(actor_handle)

    data = Data(i+1, env, actor_handle, rand_pos[0], rand_pos[1])
    dataset.append(data)

while True:
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    gym.sync_frame_time(sim)

    if process_flag == 1:
        cnt += 1 # cnt 60 = 1s
        
        if cnt == 1:
            init_pose_deg = init_pose[iteration]
            init_pose_rad = np.deg2rad(init_pose_deg)

            rotation_axis = np.array([1, 0, 0])

            half_angle = init_pose_rad / 2
            sin_half_angle = np.sin(half_angle)
            cos_half_angle = np.cos(half_angle)
            quaternion = np.array([cos_half_angle, rotation_axis[0] * sin_half_angle, rotation_axis[1] * sin_half_angle, rotation_axis[2] * sin_half_angle])

            for i in range(len(envs)):
                state = gym.get_actor_rigid_body_states(envs[i], actor_handles[i], gymapi.STATE_POS)
                state['pose']['p'].fill((dataset[i].x_world, dataset[i].y_world, init_z))
                state['pose']['r'].fill((quaternion[0], quaternion[1], quaternion[2], quaternion[3]))
                print(quaternion[0],quaternion[1],quaternion[2],quaternion[3])

                
                # random respawn            
                gym.set_actor_rigid_body_states(envs[i], actor_handles[i], state, gymapi.STATE_POS)
        
        elif cnt == 30:
            for i in range(len(envs)):
                # go down
                gym.set_dof_target_position(envs[i], pri_dof_handle, target_z)

        elif cnt == 60:
            for i in range(len(envs)):
                body_states = gym.get_actor_rigid_body_states(envs[i], actor_handles[i], gymapi.STATE_POS)
                body_positions = body_states['pose']['p']
                if body_positions[2][2] < pillar_height:
                    dataset[i].rotatable_area.append(1) # robot can be in this direction
                else:
                    dataset[i].rotatable_area.append(0) # robot can't be in this direction

            iteration += 1
            cnt = 0
      
            if iteration == len(init_pose):
                process_flag = 0
                iteration = 0

    else: # save dataset and  prepare next iteration
        for i in range(len(envs)):
            Data.GetLabel(dataset[i])
            Data.GetElevationMap(dataset[i], heightfield)

            rand_pos = (random.randint(0 + respawn_offset, num_rows - respawn_offset)*horizontal_scale, random.randint(0 + respawn_offset, num_cols - respawn_offset)*horizontal_scale) # [m]
        
            dataset[i].id += num_envs
            dataset[i].x_world = rand_pos[0]
            dataset[i].y_world = rand_pos[1]
            dataset[i].rotatable_area.clear()

        process_flag = 1


    # def GetRotatableArea(_data):
    #     OpenList = []
    #     StateList = []
    #     for index, area_state in enumerate(_data.area_state):
    #         StateList.append(area_state)
    #         if area_state == 1:
    #             OpenList.append(init_pose[index])

    #     if len(OpenList) == len(init_pose): # fully-rotatable area
    #         rotatable_area = (0,180)
    #         _data.rotatable_area.append(rotatable_area)
        
    #     elif len(OpenList) == 0: # non-rotatable area
    #         rotatable_area = (0,0)
    #         _data.rotatable_area.append(rotatable_area)
        
    #     else: # partially-rotatable area
    #         _rotatable_area = []
    #         start_flag = 1
    #         end_flag = 1
    #         for index, area_state in enumerate(_data.area_state):
    #             try:
    #                 if area_state == 1:
    #                     if start_flag == 1:
    #                         start = init_pose[index]
    #                         # a.append(init_pose[index])
    #                         start_flag = 0
    #                         end_flag = 1
    #                 else: 
    #                     if end_flag == 1:
    #                         end = init_pose[index-1]
    #                         rotatable_area = [start, end]
    #                         _rotatable_area.append(rotatable_area)
    #                         # a.append(init_pose[index-1])
    #                         end_flag = 0
    #                         start_flag = 1   
    #             except IndexError:
    #                 if end_flag == 0:
    #                     end = init_pose[index-1]
    #                     rotatable_area = [start, end]
    #                     _rotatable_area.append(rotatable_area)

    #         if StateList[0] == 1 and StateList[-1] == 1:
    #             _rotatable_area[0][0] = _rotatable_area[-1][0]
    #             rotatable_area = _rotatable_area.pop()
    #             _data.rotatable_area = rotatable_area
    #         else:
    #             _data.rotatable_area = _rotatable_area

            # cnt1 = 0
            # _rotatable_area = []

            # for index,open in enumerate(OpenList):
            #     try: 
            #         if OpenList[index+1] == open + step:
            #             cnt1 += 1
            #         elif OpenList[index+1] != open + step:   
            #             end = OpenList[index]
            #             start = end - cnt1
            #             _rotatable_area.append([start, end])
            #             # mid = (start + end) / 2
            #             # rotatable_area = (mid, end-mid)
            #             cnt1 = 0
            #     except IndexError:
            #         pass

            # if OpenList[0] == -180 and OpenList[-1] == 179:
