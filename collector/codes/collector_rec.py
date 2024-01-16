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
compute_device_id = 0
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

num_obs = 1000
pillar_height = 1.0 # [m]

####### train environment #######
# heightfield = random_rotated_square_pillar(new_sub_terrain(), num_obs, pillar_height).height_field_raw
# heightfield = random_circular_pillar(new_sub_terrain(), num_obs, pillar_height).height_field_raw
# heightfield = random_square_pillar(new_sub_terrain(), num_obs, pillar_height).height_field_raw
# heightfield = random_rectangular_pillar(new_sub_terrain(), 1400, pillar_height).height_field_raw
# heightfield = random_rotated_rectangular_pillar(new_sub_terrain(), 1200, pillar_height).height_field_raw
# heightfield = random_rotated_ellipse_pillar(new_sub_terrain(), 1000, pillar_height).height_field_raw
heightfield = random_polygon_pillar(new_sub_terrain(), 700, pillar_height).height_field_raw

####### test environment #######
# num_obs = 800
# heightfield = random_polygon_pillar(new_sub_terrain(), 800, pillar_height).height_field_raw
# num_obs = 500
# heightfield = random_line_pillar(new_sub_terrain(), 500, pillar_height).height_field_raw

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
asset_file = "urdf/rec2.urdf"
# asset_file = "urdf/aidin_8/urdf/aidin8_fix.urdf"
# asset_file = "urdf/aidin_8/urdf/aidin8_real_fix.urdf"

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

process_flag = 1
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
    
    def countImage(folder_path):
        count = 0
        
        # 폴더 내의 파일 탐색
        files = os.listdir(folder_path)
        for file in files:
            if file.lower().endswith('.png'):
                count += 1
                    
        return count
    
    def GetElevationMap(_data, _heightfield, _num_image, _base_path):
        x = int(_data.x_world / horizontal_scale) # [p]
        y = int(_data.y_world / horizontal_scale) # [p]
        
        x1 = int(x - image_size / 2) # [p]
        x2 = int(x + image_size / 2) # [p]
        y1 = int(y - image_size / 2) # [p]
        y2 = int(y + image_size / 2) # [p]

        local_map_array = _heightfield[x1:x2, y1:y2] - robot_z
        local_map_array = np.clip(local_map_array, height_min, height_max)
        local_map_array = (255 * ( local_map_array - height_min ) / (height_max - height_min)).astype(np.uint8)

        local_map_image = Image.fromarray(local_map_array)

        # save_path = "../../dataset/random_square/images/" + str(_data.id + _num_image) + ".png"
        save_path = _base_path + "images/" + str(_data.id + _num_image) + ".png"
        
        local_map_image.save(save_path)

    def getGlobalMap(_heightfield, _num_image, _base_path):
        # global_map_name = "random_square_" + str(int(_num_image/num_envs*50) + 1)
        global_map_name = str(int(_num_image/(num_envs*10)) + 1)

        global_map_array = _heightfield - robot_z
        global_map_array = np.clip(global_map_array, height_min, height_max)

        global_map_array = (255 * ( global_map_array - height_min ) / (height_max - height_min)).astype(np.uint8)

        global_map_image = Image.fromarray(global_map_array)
        
        # save_path = "../../dataset/global_map/" + global_map_name + ".png"
        save_path = _base_path + global_map_name + ".png"
        global_map_image.save(save_path)

    def GetLabel(_data, _num_image, _base_path):
        # file_path = '../../dataset/random_square/random_square.csv'
        file_path = _base_path + "dataset.csv"

        data = [_data.id + _num_image]
        
        for _rotatable_area in _data.rotatable_area:
            data.append(_rotatable_area)

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
    
    scale = 1.1
    gym.set_actor_scale(env, actor_handle, scale)

    data = Data(i+1, env, actor_handle, rand_pos[0], rand_pos[1])
    dataset.append(data)

def rotate_z_quaternion(angle_rad):
    half_angle = angle_rad / 2

    q = np.array([np.cos(half_angle), np.sin(half_angle), 0, 0])

    return q


####### train environment #######
# base_path = "../../dataset/random_rotated_square/"
# base_path = "../../dataset/random_circle/"
# base_path = "../../dataset/random_square/"
# base_path = "../../dataset/random_rectangular/"
# base_path = "../../dataset/random_rotated_rectangular/"
# base_path = "../../dataset/random_rotated_ellipse/"
# base_path = "../../dataset/rec/random_rotated_ellipse/"
# base_path = "../../dataset/rec/random_rotated_rectangular/"
# base_path = "../../dataset/rec/random_rotated_rectangular_1.1/"
# base_path = "../../dataset/rec/random_rotated_ellipse_1.1/"
base_path = "../../dataset/rec/random_polygon_1.1/"

####### test environment #######
# base_path = "../../dataset/random_polygon/"
# base_path = "../../dataset/random_line/"
# base_path = "../../dataset/rec/random_polygon/"
# base_path = "../../dataset/rec/random_wall/"

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

    if process_flag == 1:
        cnt += 1 # cnt 60 = 1s
        
        if cnt == 1:
            init_pose_deg = init_pose[iteration]
            init_pose_rad = np.deg2rad(init_pose_deg)

            quaternion = rotate_z_quaternion(init_pose_rad)

            for i in range(len(envs)):
                state = gym.get_actor_rigid_body_states(envs[i], actor_handles[i], gymapi.STATE_ALL)
                state['pose']['p'].fill((dataset[i].x_world, dataset[i].y_world, init_z))
                state['pose']['r'][0].fill((quaternion[3], quaternion[2], quaternion[1], quaternion[0]))
                
                # respawn            
                gym.set_actor_rigid_body_states(envs[i], actor_handles[i], state, gymapi.STATE_ALL)
                
                gym.set_dof_target_position(envs[i], pri_dof_handle, 0)
                        
            # print("######## ANGLE : {}".format(init_pose_deg))

        elif cnt == 10:
            for i in range(len(envs)):
                # go down
                gym.set_dof_target_position(envs[i], pri_dof_handle, target_z)

        elif cnt == 30:
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
            Data.GetLabel(dataset[i], num_image, base_path)
            Data.GetElevationMap(dataset[i], heightfield, num_image, base_path)

            rand_pos = (random.randint(0 + respawn_offset, num_rows - respawn_offset)*horizontal_scale, random.randint(0 + respawn_offset, num_cols - respawn_offset)*horizontal_scale) # [m]
        
            dataset[i].id += num_envs
            dataset[i].x_world = rand_pos[0]
            dataset[i].y_world = rand_pos[1]
            dataset[i].rotatable_area.clear()

        cnt_whole_process += 1
        process_flag = 1

        print("######## WHOLE PROCESS : {}".format(cnt_whole_process))


    if cnt_whole_process == 10:
        gym.destroy_sim(sim)



