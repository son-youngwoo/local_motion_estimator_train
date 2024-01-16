from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
import random
import math
from terrain_utils import *
from PIL import Image
import csv
import os

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
sim_params.physx.max_gpu_contact_pairs = 16 * 1024 * 1024


# set Flex-specific parameters
# sim_params.flex.solver_type = 5
# sim_params.flex.num_outer_iterations = 4
# sim_params.flex.num_inner_iterations = 20
# sim_params.flex.relaxation = 0.8
# sim_params.flex.warm_start = 0.5

# compute_device_id : GPU selection for simulation
# graphics_device_id : GPU selection for rendering 
compute_device_id = 1
graphics_device_id = -1
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
terrain_width = 60.
terrain_length = 60.
horizontal_scale = 0.02  # [m]
vertical_scale = 0.1  # [m]
num_rows = int(terrain_width/horizontal_scale)
num_cols = int(terrain_length/horizontal_scale)
heightfield = np.zeros((num_rows, num_cols), dtype=np.int16)

def new_sub_terrain(): return SubTerrain(width=num_rows, length=num_cols, vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)

pillar_height = 1.0 # [m]

####### train environment #######
heightfield = random_rotated_rectangular_pillar(new_sub_terrain(), 1000, pillar_height).height_field_raw
# heightfield = random_triangular_pillar(new_sub_terrain(), 700, pillar_height).height_field_raw
# heightfield = random_rotated_ellipse_pillar(new_sub_terrain(), 1000, pillar_height).height_field_raw

####### test environment #######
# heightfield = random_polygon_pillar(new_sub_terrain(), 700, pillar_height).height_field_raw
# heightfield = random_wall_pillar(new_sub_terrain(), 500, pillar_height).height_field_raw

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
# asset_file = "urdf/aidin_8/urdf/aidin8_fix.urdf"
asset_file = "urdf/rec_gamma1.urdf"

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
cnt_whole_process = 0
dataset = []

######################


## respawn ##

process_flag = 0
step = 1 # [deg]
init_pose = list(range(0,360,step)) # [deg]
iteration = 0
init_z = 1.8 # [m]
target_z = 0.6 - init_z # [m]
respawn_offset = 60 # [p]

######################

## map processing ##

robot_z = 0.6 # [m]
local_map_size = 2 # [m]
height_max = 0 # [m]
height_min = -0.6 # [m]
image_size = 100 # [p]

######################

# edge_max = 0.1*math.sqrt(2)
# edge_min = 0.1

edge_max = 0.5
edge_min = 0.15

start1 = 0
end1 = 0

class Data:

    def __init__(self, _id=0, _env_handle = None, _actor_handle = None):
        self.id = _id
        self.env_handle = _env_handle
        self.actor_handle = _actor_handle
        self.x_init_world = 0 
        self.y_init_world = 0
        self.yaw_init_world = 0
        self.x_tar_world = 0 
        self.y_tar_world = 0 
        self.isSuccess = None
    
    def countImage(folder_path):
        count = 0
        
        # 폴더 내의 파일 탐색
        files = os.listdir(folder_path)
        for file in files:
            if file.lower().endswith('.png'):
                count += 1
                    
        return count
    
    def getElevationMap(_data, _heightfield, _num_image, _base_path):
        x = int(_data.x_init_world / horizontal_scale) # [p]
        y = int(_data.y_init_world / horizontal_scale) # [p]
        
        x1 = int(x - image_size / 2) # [p]
        x2 = int(x + image_size / 2) # [p]
        y1 = int(y - image_size / 2) # [p]
        y2 = int(y + image_size / 2) # [p]

        local_map_array = _heightfield[x1:x2, y1:y2] - robot_z
        local_map_array = np.clip(local_map_array, height_min, height_max)
        local_map_array = (255 * ( local_map_array - height_min ) / (height_max - height_min)).astype(np.uint8)

        local_map_image = Image.fromarray(local_map_array)

        save_path = _base_path + "images/" + str(_data.id + _num_image) + ".png"
        
        local_map_image.save(save_path)

    def getGlobalMap(_heightfield, _num_image, _base_path):
        global_map_name = str(int(_num_image/(num_envs*10)) + 1)

        global_map_array = _heightfield - robot_z
        global_map_array = np.clip(global_map_array, height_min, height_max)

        global_map_array = (255 * ( global_map_array - height_min ) / (height_max - height_min)).astype(np.uint8)

        global_map_image = Image.fromarray(global_map_array)
        
        save_path = _base_path + global_map_name + ".png"
        global_map_image.save(save_path)

    def getLabel(_data, _num_image, _base_path):
        file_path = _base_path + "dataset.csv"

        data = [_data.id + _num_image, _data.x_init_world, _data.y_init_world, _data.yaw_init_world, _data.x_tar_world, _data.y_tar_world, _data.isSuccess]

        with open(file_path, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(data)


# create and populate the environments
# all env's frame is (0,0,0) 
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
    props["stiffness"] = (50000.0)#, 50000.0, 50000.0, 50000.0, 50000.0, 50000.0, 50000.0, 50000.0, 50000.0, 50000.0, 50000.0, 50000.0, 50000.0, 50000.0)
    props["damping"] = (100.0)#, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0)
    gym.set_actor_dof_properties(env, actor_handle, props)

    # Set DOF drive targets for dataset collection process
    pri_dof_handle = gym.find_actor_dof_handle(env, actor_handle, 'joint1')
    # rev_dof_handle = gym.find_actor_dof_handle(env, actor_handle, 'joint2')

    gym.set_dof_target_position(env, pri_dof_handle, 0)
    # gym.set_dof_target_position(env, rev_dof_handle, 0)

    # Set DOF drive targets for stand pose
    # rev_dof_handle_RFJ1 = gym.find_actor_dof_handle(env, actor_handle, 'RFJ1')
    # rev_dof_handle_RFJ2 = gym.find_actor_dof_handle(env, actor_handle, 'RFJ2')
    # rev_dof_handle_RFJ3 = gym.find_actor_dof_handle(env, actor_handle, 'RFJ3')
    # rev_dof_handle_LFJ1 = gym.find_actor_dof_handle(env, actor_handle, 'LFJ1')
    # rev_dof_handle_LFJ2 = gym.find_actor_dof_handle(env, actor_handle, 'LFJ2')
    # rev_dof_handle_LFJ3 = gym.find_actor_dof_handle(env, actor_handle, 'LFJ3')
    # rev_dof_handle_LBJ1 = gym.find_actor_dof_handle(env, actor_handle, 'LBJ1')
    # rev_dof_handle_LBJ2 = gym.find_actor_dof_handle(env, actor_handle, 'LBJ2')
    # rev_dof_handle_LBJ3 = gym.find_actor_dof_handle(env, actor_handle, 'LBJ3')
    # rev_dof_handle_RBJ1 = gym.find_actor_dof_handle(env, actor_handle, 'RBJ1')
    # rev_dof_handle_RBJ2 = gym.find_actor_dof_handle(env, actor_handle, 'RBJ2')
    # rev_dof_handle_RBJ3 = gym.find_actor_dof_handle(env, actor_handle, 'RBJ3')
    
    # # stand pose
    # gym.set_dof_target_position(env, rev_dof_handle_RFJ1, 0)
    # gym.set_dof_target_position(env, rev_dof_handle_RFJ2, 0.81)
    # gym.set_dof_target_position(env, rev_dof_handle_RFJ3, -0.1)
    # gym.set_dof_target_position(env, rev_dof_handle_LFJ1, 0)
    # gym.set_dof_target_position(env, rev_dof_handle_LFJ2, 0.81)
    # gym.set_dof_target_position(env, rev_dof_handle_LFJ3, -0.1)
    # gym.set_dof_target_position(env, rev_dof_handle_LBJ1, 0)
    # gym.set_dof_target_position(env, rev_dof_handle_LBJ2, 0.71)
    # gym.set_dof_target_position(env, rev_dof_handle_LBJ3, -0.1)
    # gym.set_dof_target_position(env, rev_dof_handle_RBJ1, 0)
    # gym.set_dof_target_position(env, rev_dof_handle_RBJ2, 0.71)
    # gym.set_dof_target_position(env, rev_dof_handle_RBJ3, -0.1)
    
    actor_handles.append(actor_handle)
    
    scale = 1.25
    gym.set_actor_scale(env, actor_handle, scale)

    data = Data(i+1, env, actor_handle)#, rand_pos[0], rand_pos[1])
    dataset.append(data)

def rotate_z_quaternion(angle_rad):
    half_angle = angle_rad / 2

    q = np.array([np.cos(half_angle), np.sin(half_angle), 0, 0])

    return q


####### train environment #######

base_path = "../../dataset/basic/rectangular_model/1.25/edge_distance(10-14)/random_rectangular/"
# base_path = "../../dataset/basic/rectangular_model/1.25/edge_distance(10-14)/random_triangle/"
# base_path = "../../dataset/basic/rectangular_model/1.25/edge_distance(10-14)/random_ellipse/"

image_path = base_path + "images/"
num_image = Data.countImage(image_path)

Data.getGlobalMap(heightfield, num_image, base_path)

while True:
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    gym.sync_frame_time(sim)

    if process_flag == 0:
        for i in range(len(envs)):
            # initial yaw
            yaw_init_deg = random.randint(0,360) #init_pose[iteration]

            # initial position 
            pos_init = (random.randint(0 + respawn_offset, num_rows - respawn_offset)*horizontal_scale, random.randint(0 + respawn_offset, num_cols - respawn_offset)*horizontal_scale) # [m]

            # target position 
            while True:
                _pos_tar = (random.uniform(0, edge_max), random.uniform(0, edge_max))
                dist = math.sqrt(_pos_tar[0]**2 + _pos_tar[1]**2)
                if  dist >= edge_min and dist <= edge_max:
                    break
            pos_tar = (pos_init[0] + _pos_tar[0], pos_init[1] + _pos_tar[1]) 

            dataset[i].x_init_world = round(pos_init[0], 2)
            dataset[i].y_init_world = round(pos_init[1], 2)
            dataset[i].yaw_init_world = yaw_init_deg        
            dataset[i].x_tar_world = round(pos_tar[0], 2)
            dataset[i].y_tar_world = round(pos_tar[1], 2)
        
        process_flag = 1

    elif process_flag == 1:
        cnt += 1 # cnt 60 = 1s
        
        ## step1
        if cnt == 1:
            for i in range(len(envs)):
                yaw_init_rad = np.deg2rad(dataset[i].yaw_init_world)

                quat_init = rotate_z_quaternion(yaw_init_rad)
                print("pos_init_world : {}, {}".format(dataset[i].x_init_world, dataset[i].y_init_world))
                print("yaw_init_world : ",dataset[i].yaw_init_world)
                
                state = gym.get_actor_rigid_body_states(envs[i], actor_handles[i], gymapi.STATE_ALL)
                state['pose']['p'].fill((dataset[i].x_init_world, dataset[i].y_init_world, init_z))
                state['pose']['r'][0].fill((quat_init[3], quat_init[2], quat_init[1], quat_init[0]))
                
                # respawn            
                gym.set_actor_rigid_body_states(envs[i], actor_handles[i], state, gymapi.STATE_ALL)
                
                gym.set_dof_target_position(envs[i], pri_dof_handle, 0)
                        

        elif cnt == 30:
            for i in range(len(envs)):
                # go down
                gym.set_dof_target_position(envs[i], pri_dof_handle, target_z)

        elif cnt == 50:
            success_envs1 = []
            for i in range(len(envs)):
                body_states = gym.get_actor_rigid_body_states(envs[i], actor_handles[i], gymapi.STATE_POS)
                body_positions = body_states['pose']['p']
                if body_positions[2][2] < pillar_height:
                    success_envs1.append(i)
                else:
                    dataset[i].isSuccess = 0

        ## step3
        elif cnt == 60:
            for i in success_envs1:
                yaw_init_rad = np.deg2rad(dataset[i].yaw_init_world)

                quat_init = rotate_z_quaternion(yaw_init_rad)
                print("pos_tar_world : {}, {}".format(dataset[i].x_tar_world, dataset[i].y_tar_world))
                print("yaw_init_world : ",dataset[i].yaw_init_world)

                state = gym.get_actor_rigid_body_states(envs[i], actor_handles[i], gymapi.STATE_ALL)
                state['pose']['p'].fill((dataset[i].x_tar_world, dataset[i].y_tar_world, init_z))
                state['pose']['r'][0].fill((quat_init[3], quat_init[2], quat_init[1], quat_init[0]))
                
                # respawn            
                gym.set_actor_rigid_body_states(envs[i], actor_handles[i], state, gymapi.STATE_ALL)
                
                gym.set_dof_target_position(envs[i], pri_dof_handle, 0)
                        
        elif cnt == 90:
            for i in success_envs1:
                # go down
                gym.set_dof_target_position(envs[i], pri_dof_handle, target_z)

        elif cnt == 110:
            for i in success_envs1:
                body_states = gym.get_actor_rigid_body_states(envs[i], actor_handles[i], gymapi.STATE_POS)
                body_positions = body_states['pose']['p']
                if body_positions[2][2] < pillar_height: # step3 success
                    dataset[i].isSuccess = 1
                else: # step3 failure
                    dataset[i].isSuccess = 0
                    
            process_flag = 2
                
    elif process_flag == 2: # save dataset and  prepare next iteration
        
        for i in range(len(envs)):
            print("isSuccess : ", dataset[i].isSuccess)
            Data.getLabel(dataset[i], num_image, base_path)
            Data.getElevationMap(dataset[i], heightfield, num_image, base_path)

            dataset[i].id += num_envs

        cnt_whole_process += 1
        process_flag = 0
        cnt = 0


        print("######## WHOLE PROCESS : {}".format(cnt_whole_process))


    if cnt_whole_process == 10:
        gym.destroy_sim(sim)



