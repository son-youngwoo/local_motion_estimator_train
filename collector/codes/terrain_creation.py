"""
Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Terrain examples
-------------------------
Demonstrates the use terrain meshes.
Press 'R' to reset the  simulation
"""

import numpy as np
from numpy.random import choice
from numpy.random.mtrand import triangular
from scipy import interpolate
import os

from isaacgym import gymutil, gymapi
# from isaacgym.terrain_utils import *
from terrain_utils import *
from math import sqrt
from PIL import Image


# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments()

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UpAxis.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

if args.physics_engine == gymapi.SIM_FLEX:
    print("WARNING: Terrain creation is not supported for Flex! Switching to PhysX")
    args.physics_engine = gymapi.SIM_PHYSX
sim_params.substeps = 2
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 0
sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

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
# pillar_size = 0.5 # [m]

######## train environment ########
# heightfield = random_rotated_rectangular_pillar(new_sub_terrain(), 1000, pillar_height).height_field_raw
# heightfield = random_rotated_ellipse_pillar(new_sub_terrain(), 1000, pillar_height).height_field_raw
# heightfield = random_triangular_pillar(new_sub_terrain(), 400, pillar_height).height_field_raw

######## test environment ########
# heightfield = random_polygon_pillar(new_sub_terrain(), 400, pillar_height).height_field_raw
heightfield = random_wall_pillar(new_sub_terrain(), 500, pillar_height).height_field_raw

####### global map save #######
robot_z = 0.6
height_max = 0 # [m]
height_min = -0.6 # [m]
base_path = "../../global_map/"

global_map_name = "1"

global_map_array = heightfield - robot_z
global_map_array = np.clip(global_map_array, height_min, height_max)

global_map_array = (255 * ( global_map_array - height_min ) / (height_max - height_min)).astype(np.uint8)

global_map_image = Image.fromarray(global_map_array)

# save_path = "../../dataset/global_map/" + global_map_name + ".png"
save_path = base_path + global_map_name + ".png"
global_map_image.save(save_path)


# add the terrain as a triangle mesh
vertices, triangles = convert_heightfield_to_trimesh(heightfield, horizontal_scale=horizontal_scale, vertical_scale=vertical_scale, slope_threshold=1.5)
tm_params = gymapi.TriangleMeshParams()
tm_params.nb_vertices = vertices.shape[0]
tm_params.nb_triangles = triangles.shape[0]
tm_params.transform.p.x = -1.
tm_params.transform.p.y = -1.
gym.add_triangle_mesh(sim, vertices.flatten(), triangles.flatten(), tm_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

cam_pos = gymapi.Vec3(-5, -5, 15)
cam_target = gymapi.Vec3(0, 0, 10)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# subscribe to spacebar event for reset
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")

while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
