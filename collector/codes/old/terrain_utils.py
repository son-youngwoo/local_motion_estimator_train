# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import gymutil, gymapi
import math
import random
from itertools import product
from skimage.draw import polygon
from skimage.draw import line


def random_rotated_square_pillar(terrain, _num_obs, _pillar_height):

    num_obs = _num_obs
    pillar_height_m = _pillar_height # [m]
    pillar_height= pillar_height_m/terrain.vertical_scale # [p]
    pillar_size_m_min = 0.1 # [m]
    pillar_size_m_max = 1 # [m]
    pillar_size_min = math.floor(pillar_size_m_min/terrain.horizontal_scale) # [p]
    pillar_size_max = math.floor(pillar_size_m_max/terrain.horizontal_scale) # [p]
    pillar_angle_min = 2 # [deg]
    pillar_angle_max = 90 # [deg]

    for i in range(num_obs):
        rand_pos = (random.randint(0, terrain.width), random.randint(0, terrain.length)) # [p]
        rand_pos = np.array(rand_pos)
        pillar_size = random.randint(pillar_size_min, pillar_size_max+1) # [p]
        pillar_angle = random.randint(pillar_angle_min, pillar_angle_max) # [deg]
       
        R = np.array([[np.cos(np.radians(pillar_angle)), -np.sin(np.radians(pillar_angle))],[np.sin(np.radians(pillar_angle)), np.cos(np.radians(pillar_angle))]]) # [rad]
        R_inv = np.linalg.inv(R)
          
        x1 = [- math.floor(pillar_size/2)] # [p]
        x2 = [+ math.floor(pillar_size/2)] # [p]
        y1 = [- math.floor(pillar_size/2)] # [p]
        y2 = [+ math.floor(pillar_size/2)] # [p]

        x = range(x1[0], x2[0]+1)
        y = range(y1[0], y2[0]+1)
        
        sides_pos = list(product(x1, y)) + list(product(x2, y)) + list(product(x, y2)) + list(product(x, y1))
        rot_sides_pos = []

        for pos in sides_pos:
            pos = np.array(pos)
            rot_pos = np.dot(R_inv, pos).astype(int) + rand_pos
            rot_pos = np.round(rot_pos)
            rot_sides_pos.append(tuple(rot_pos))

        dict_rot_sides_pos = {}

        for x, y in rot_sides_pos:
            try:
                dict_rot_sides_pos[y].append(x)
            except KeyError:
                dict_rot_sides_pos[y] = [x]

        for y in dict_rot_sides_pos.keys():
            x_max = max(dict_rot_sides_pos[y])
            x_min = min(dict_rot_sides_pos[y])
             
            for x in range(x_min, x_max+1):
                try:
                    terrain.height_field_raw[x,y] = pillar_height
                except IndexError:
                    continue
    
    count_free = np.count_nonzero(terrain.height_field_raw == 0)
    count_obs = np.count_nonzero(terrain.height_field_raw == pillar_height)
    print("free, obs : {}, {}".format(count_free, count_obs))
    
    return terrain

def random_square_pillar(terrain, _num_obs, _pillar_height):

    num_obs = _num_obs
    rand_pos_list = []
    pillar_height_m = _pillar_height # [m]
    pillar_height = pillar_height_m/terrain.vertical_scale # [p]
    pillar_size_m_min = 0.1 # [m]
    pillar_size_m_max = 1 # [m]
    pillar_size_min = math.floor(pillar_size_m_min/terrain.horizontal_scale) # [p]
    pillar_size_max = math.floor(pillar_size_m_max/terrain.horizontal_scale) # [p]

    for i in range(num_obs):
        rand_pos = (random.randint(0, terrain.width), random.randint(0, terrain.length)) # [p]
        rand_pos_list.append(rand_pos)
      
    for rand_pos in rand_pos_list:
        # pillar_size = random.uniform(0.1, 1)

        # x1 = rand_pos[0] - math.floor(pillar_size/terrain.horizontal_scale/2) # [p]
        # x2 = rand_pos[0] + math.floor(pillar_size/terrain.horizontal_scale/2) # [p]
        # y1 = rand_pos[1] - math.floor(pillar_size/terrain.horizontal_scale/2) # [p]
        # y2 = rand_pos[1] + math.floor(pillar_size/terrain.horizontal_scale/2) # [p]
        pillar_size = random.randint(pillar_size_min, pillar_size_max+1) # [p]

        x1 = rand_pos[0] - math.floor(pillar_size/2) # [p]
        x2 = rand_pos[0] + math.floor(pillar_size/2) # [p]
        y1 = rand_pos[1] - math.floor(pillar_size/2) # [p]
        y2 = rand_pos[1] + math.floor(pillar_size/2) # [p]
        
        # if math.floor(pillar_size/terrain.horizontal_scale/2) % 2 == 0:
        #     x2 += 1
        #     y2 += 1

        # if x1 < 0 or x2 <0 or y1 < 0 or y2 <0 or x1 >= terrain.width or x2 >= terrain.width or y1 >= terrain.length or y2 >= terrain.length:
        #     print("Delete obstacle which is out of map")
        #     continue
        try:
            terrain.height_field_raw[x1:x2, y1:y2] = pillar_height
        except IndexError:
            print("Delete obstacle which is out of map")
            break
    
    count_free = np.count_nonzero(terrain.height_field_raw == 0)
    count_obs = np.count_nonzero(terrain.height_field_raw == pillar_height)
    print("free, obs : {}, {}".format(count_free, count_obs))          
    
    return terrain

def random_rectangular_pillar(terrain, _num_obs, _pillar_height):

    num_obs = _num_obs
    rand_pos_list = []
    pillar_height_m = _pillar_height # [m]
    pillar_height = pillar_height_m/terrain.vertical_scale # [p]
    pillar_size_m_min = 0.1 # [m]
    pillar_size_m_max = 1 # [m]
    pillar_size_min = math.floor(pillar_size_m_min/terrain.horizontal_scale) # [p]
    pillar_size_max = math.floor(pillar_size_m_max/terrain.horizontal_scale) # [p]

    for i in range(num_obs):
        rand_pos = (random.randint(0, terrain.width), random.randint(0, terrain.length)) # [p]
        rand_pos_list.append(rand_pos)
      
    for rand_pos in rand_pos_list:
        # pillar_size = random.uniform(0.1, 1)

        # x1 = rand_pos[0] - math.floor(pillar_size/terrain.horizontal_scale/2) # [p]
        # x2 = rand_pos[0] + math.floor(pillar_size/terrain.horizontal_scale/2) # [p]
        # y1 = rand_pos[1] - math.floor(pillar_size/terrain.horizontal_scale/2) # [p]
        # y2 = rand_pos[1] + math.floor(pillar_size/terrain.horizontal_scale/2) # [p]
        pillar_size_width = random.randint(pillar_size_min, pillar_size_max+1) # [p]
        pillar_size_height = random.randint(pillar_size_min, pillar_size_max+1) # [p]

        x1 = rand_pos[0] - math.floor(pillar_size_width/2) # [p]
        x2 = rand_pos[0] + math.floor(pillar_size_width/2) # [p]
        y1 = rand_pos[1] - math.floor(pillar_size_height/2) # [p]
        y2 = rand_pos[1] + math.floor(pillar_size_height/2) # [p]
        
        # if math.floor(pillar_size/terrain.horizontal_scale/2) % 2 == 0:
        #     x2 += 1
        #     y2 += 1

        # if x1 < 0 or x2 <0 or y1 < 0 or y2 <0 or x1 >= terrain.width or x2 >= terrain.width or y1 >= terrain.length or y2 >= terrain.length:
        #     print("Delete obstacle which is out of map")
        #     continue
        try:
            terrain.height_field_raw[x1:x2, y1:y2] = pillar_height
        except IndexError:
            print("Delete obstacle which is out of map")
            break
    
    count_free = np.count_nonzero(terrain.height_field_raw == 0)
    count_obs = np.count_nonzero(terrain.height_field_raw == pillar_height)
    print("free, obs : {}, {}".format(count_free, count_obs))          
    
    return terrain

def random_circular_pillar(terrain, _num_obs, _pillar_height):

    num_obs = _num_obs
    rand_pos_list = []
    pillar_height_m = _pillar_height # [m]
    pillar_height = pillar_height_m/terrain.vertical_scale # [p]
    pillar_size_m_min = 0.1 # [m]
    pillar_size_m_max = 1 # [m]
    pillar_size_min = math.floor(pillar_size_m_min/terrain.horizontal_scale) # [p]
    pillar_size_max = math.floor(pillar_size_m_max/terrain.horizontal_scale) # [p]

    for i in range(num_obs):
        rand_pos = (random.randint(0, terrain.width), random.randint(0, terrain.length)) # [p]
        rand_pos_list.append(rand_pos)
    
    for rand_pos in rand_pos_list:
        pillar_size = random.randint(pillar_size_min, pillar_size_max+1) # [p]

        x1 = rand_pos[0] - math.floor(pillar_size/2) # [p]
        x2 = rand_pos[0] + math.floor(pillar_size/2) # [p]
        y1 = rand_pos[1] - math.floor(pillar_size/2) # [p]
        y2 = rand_pos[1] + math.floor(pillar_size/2) # [p]

        for x in range(x1, x2+1):
            for y in range(y1, y2+1):
                distance = math.sqrt((x - rand_pos[0])**2 + (y - rand_pos[1]) ** 2)
                if distance <= pillar_size/2:
                    try:
                        terrain.height_field_raw[x][y] = pillar_height
                    except IndexError:
                        continue
    
    count_free = np.count_nonzero(terrain.height_field_raw == 0)
    count_obs = np.count_nonzero(terrain.height_field_raw == pillar_height)
    print("free, obs : {}, {}".format(count_free, count_obs))
    
    return terrain

def random_rotated_ellipse_pillar(terrain, _num_obs, _pillar_height):

    num_obs = _num_obs
    pillar_height_m = _pillar_height # [m]
    pillar_height = pillar_height_m/terrain.vertical_scale # [p]
    pillar_size_m_min = 0.1 # [m]
    pillar_size_m_max = 1 # [m]
    pillar_size_min = math.floor(pillar_size_m_min/terrain.horizontal_scale) # [p]
    pillar_size_max = math.floor(pillar_size_m_max/terrain.horizontal_scale) # [p]
    pillar_angle_min = 0 # [deg]
    pillar_angle_max = 180 # [deg]

    for i in range(num_obs):
        rand_pos = (random.randint(0, terrain.width), random.randint(0, terrain.length)) # [p]
        rand_pos = np.array(rand_pos)
        pillar_size_major = random.randint(pillar_size_min, pillar_size_max+1) # [p]
        pillar_size_minor = random.randint(pillar_size_min, pillar_size_max+1) # [p]
        pillar_angle = random.randint(pillar_angle_min, pillar_angle_max) # [deg]
       
        R = np.array([[np.cos(np.radians(pillar_angle)), -np.sin(np.radians(pillar_angle))],[np.sin(np.radians(pillar_angle)), np.cos(np.radians(pillar_angle))]]) # [rad]
        
        if pillar_size_major >= pillar_size_minor:
            eval_size = pillar_size_major
        else:
            eval_size = pillar_size_minor

        for x in range(-round(eval_size/2), round(eval_size/2)+1):
            for y in range(-round(eval_size/2), round(eval_size/2)+1):
                pos = np.array([x, y])
                rot_pos = np.dot(R, pos).astype(int)
                if (rot_pos[0] / (pillar_size_major/2)) ** 2 + (rot_pos[1] / (pillar_size_minor/2)) ** 2 <= 1:
                # if (pos[0] / (pillar_size_major/2)) ** 2 + (pos[1] / (pillar_size_minor/2)) ** 2 <= 1:
                    try:
                        terrain.height_field_raw[pos[0] + rand_pos[0], pos[1] + rand_pos[1]] = pillar_height
                    except IndexError:
                        continue
    
    count_free = np.count_nonzero(terrain.height_field_raw == 0)
    count_obs = np.count_nonzero(terrain.height_field_raw == pillar_height)
    print("free, obs : {}, {}".format(count_free, count_obs))
    
    return terrain

def random_rotated_rectangular_pillar(terrain, _num_obs, _pillar_height):

    num_obs = _num_obs
    pillar_height_m = _pillar_height # [m]
    pillar_height= pillar_height_m/terrain.vertical_scale # [p]
    pillar_size_m_min = 0.1 # [m]
    pillar_size_m_max = 1 # [m]
    pillar_size_min = math.floor(pillar_size_m_min/terrain.horizontal_scale) # [p]
    pillar_size_max = math.floor(pillar_size_m_max/terrain.horizontal_scale) # [p]
    pillar_angle_min = 0 # [deg]
    pillar_angle_max = 90 # [deg]

    for i in range(num_obs):
        rand_pos = [random.randint(0, terrain.width), random.randint(0, terrain.length)] # [p]
        rand_pos = np.array(rand_pos).reshape((2, 1))
        pillar_size_width = random.randint(pillar_size_min, pillar_size_max+1) # [p]
        pillar_size_height = random.randint(pillar_size_min, pillar_size_max+1) # [p]
        pillar_angle = random.randint(pillar_angle_min, pillar_angle_max) # [deg]
        
        R = np.array([[np.cos(np.radians(pillar_angle)), -np.sin(np.radians(pillar_angle))],[np.sin(np.radians(pillar_angle)), np.cos(np.radians(pillar_angle))]]) # [rad]
        # R_inv = np.linalg.inv(R)
          
        x1 = [- math.floor(pillar_size_width/2)] # [p]
        x2 = [+ math.floor(pillar_size_width/2)] # [p]
        y1 = [- math.floor(pillar_size_height/2)] # [p]
        y2 = [+ math.floor(pillar_size_height/2)] # [p]
        rotated_rectangular_points = []

        # rectangular_points = np.array([[x1, y1], [x1, y2], [x2, y1], [x2, y2]])
        rectangular_points = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
        for rectangular_point in rectangular_points:
            rectangular_point = np.array(rectangular_point)
            rotated_rectangular_point = np.dot(R, rectangular_point) + rand_pos

            rotated_rectangular_points.append(rotated_rectangular_point)
        rotated_rectangular_points = np.array(rotated_rectangular_points, dtype=np.int32).reshape((4,2))
        
        rr, cc = polygon(rotated_rectangular_points[:, 1], rotated_rectangular_points[:, 0])
        
        try:
            terrain.height_field_raw[rr, cc] = pillar_height
        except IndexError:
            continue
    
    count_free = np.count_nonzero(terrain.height_field_raw == 0)
    count_obs = np.count_nonzero(terrain.height_field_raw == pillar_height)
    print("free, obs : {}, {}".format(count_free, count_obs))

    return terrain

def random_polygon_pillar(terrain, _num_obs, _pillar_height):

    num_obs = _num_obs
    pillar_height_m = _pillar_height # [m]
    pillar_height= pillar_height_m/terrain.vertical_scale # [p]
    side_num_min = 4
    side_num_max = 6
    side_length_m_min = 0.3
    side_length_m_max = 0.8
    # side_length_m_min = 0.8
    # side_length_m_max = 1.5
    side_length_min = math.floor(side_length_m_min/terrain.horizontal_scale) # [p]
    side_length_max = math.floor(side_length_m_max/terrain.horizontal_scale) # [p]

    for i in range(num_obs):
        rand_pos = (random.randint(0, terrain.width), random.randint(0, terrain.length)) # [p]
        rand_pos = np.array(rand_pos)
        
        side_num = np.random.randint(side_num_min, side_num_max+1)

        # 6각형의 각 변의 길이를 랜덤하게 설정
        side_lengths = np.random.randint(side_length_min, side_length_max, size=side_num)

        # 6각형의 꼭짓점 좌표 계산
        polygon_points = []
        for i in range(side_num):
            angle = 2 * np.pi / 6 * i
            x = int(rand_pos[0] + side_lengths[i] * np.cos(angle))
            y = int(rand_pos[1] + side_lengths[i] * np.sin(angle))
            polygon_points.append((x, y))

        polygon_points = np.array(polygon_points, dtype=np.int32)

        # 6각형 내부를 1로 채우기
        rr, cc = polygon(polygon_points[:, 1], polygon_points[:, 0])
        
        try:
            terrain.height_field_raw[rr, cc] = pillar_height
        except IndexError:
            continue
    
    count_free = np.count_nonzero(terrain.height_field_raw == 0)
    count_obs = np.count_nonzero(terrain.height_field_raw == pillar_height)
    print("free, obs : {}, {}".format(count_free, count_obs))

    return terrain

####### test environment #######


def random_arc_pillar(terrain, _num_obs, _pillar_height):

    num_obs = _num_obs
    rand_pos_list = []
    pillar_height_m = _pillar_height # [m]
    pillar_height = pillar_height_m/terrain.vertical_scale # [p]
    pillar_size_m_min = 3 # [m]
    pillar_size_m_max = 7 # [m]
    pillar_size_min = math.floor(pillar_size_m_min/terrain.horizontal_scale) # [p]
    pillar_size_max = math.floor(pillar_size_m_max/terrain.horizontal_scale) # [p]
    thickness = math.floor(0.2/terrain.horizontal_scale) #[p]
    pillar_angle_min = 90 # [deg]
    pillar_angle_max = 180 # [deg]


    for i in range(num_obs):
        rand_pos = (random.randint(0, terrain.width), random.randint(0, terrain.length)) # [p]
        rand_pos_list.append(rand_pos)
    
    for rand_pos in rand_pos_list:
        pillar_size = random.randint(pillar_size_min, pillar_size_max+1) # [p]

        x1 = rand_pos[0] - math.floor(pillar_size/2) # [p]
        x2 = rand_pos[0] + math.floor(pillar_size/2) # [p]
        y1 = rand_pos[1] - math.floor(pillar_size/2) # [p]
        y2 = rand_pos[1] + math.floor(pillar_size/2) # [p]

        # pillar_angle_start = random.randint(0, 359) # [deg]
        # pillar_angle_range = random.randint(pillar_angle_min, pillar_angle_max) # [deg]
        # pillar_angle_end = pillar_angle_start + pillar_angle_range
        # if pillar_angle_end >= 360:
        #     pillar_angle_end -= 360
        # if pillar_angle_end < pillar_angle_start:
        #     pillar_angle_list_start = list(range(pillar_angle_start, 359+1))
        #     pillar_angle_list_end = list(range(0, pillar_angle_end+1))
        #     pillar_angle_list = pillar_angle_list_end + pillar_angle_list_start
        # else:
        #     pillar_angle_list = list(range(pillar_angle_start, pillar_angle_end+1))
        
        # for x in range(x1, x2+1):
        #     for y in range(y1, y2+1):
        #         distance = math.sqrt((x - rand_pos[0])**2 + (y - rand_pos[1]) ** 2)
        #         if distance <= pillar_size/2 and distance >= (pillar_size/2 - thickness):
        #             if (terrain.width > x and x >= 0) and (terrain.length > y and y >= 0): 
        #                 angle = round(np.degrees(math.atan2(y-rand_pos[1],x-rand_pos[0])))
        #                 if angle < 0:
        #                     angle += 360
        #                 # print("angle : ",angle)
        #                 if angle in pillar_angle_list:
        #                     terrain.height_field_raw[x][y] = pillar_height


        for x in range(x1, x2+1):
            for y in range(y1, y2+1):
                distance = math.sqrt((x - rand_pos[0])**2 + (y - rand_pos[1]) ** 2)
                if distance <= pillar_size/2 and distance >= (pillar_size/2 - thickness):
                    if (terrain.width > x and x >= 0) and (terrain.length > y and y >= 0): 
                        terrain.height_field_raw[x][y] = pillar_height
                    
        for x in range(x1, x2+1):
            for y in range(y1, y2+1):
                distance = math.sqrt((x - rand_pos[0])**2 + (y - rand_pos[1]) ** 2)
                if distance <= pillar_size/2 - thickness:
                    try:
                        terrain.height_field_raw[x][y] = 0
                    except IndexError:
                        continue
                        
    for i in list(range(1, 15)):
        print(i)
        terrain.height_field_raw[200*i:200*i + 70,:] = 0
        terrain.height_field_raw[:,200*i:200*i + 70] = 0

        
        # for x in range(x1, x2+1):
        #     for y in range(y1, y2+1):
        #         distance = math.sqrt((x - rand_pos[0])**2 + (y - rand_pos[1]) ** 2)
        #         if distance <= pillar_size/2 and distance >= (pillar_size/2 - thickness):
        #             if (terrain.width > x and x >= 0) and (terrain.length > y and y >= 0): 
        #                 terrain.height_field_raw[x][y] = pillar_height

        # for x in range(x1, x2+1):
        #     for y in range(y1, y2+1):
        #         angle = round(np.degrees(math.atan2(y-rand_pos[1],x-rand_pos[0])))
        #         if angle < 0:
        #             angle += 360
        #         # print("angle : ",angle)
        #         if angle not in pillar_angle_list:
        #             try:
        #                 terrain.height_field_raw[x][y] = 0
        #             except IndexError:
        #                 continue
        
    
    count_free = np.count_nonzero(terrain.height_field_raw == 0)
    count_obs = np.count_nonzero(terrain.height_field_raw == pillar_height)
    print("free, obs : {}, {}".format(count_free, count_obs))
    
    return terrain

def random_line_pillar(terrain,_num_obs, _pillar_height):

    num_obs = _num_obs
    pillar_height_m = _pillar_height # [m]
    pillar_height= pillar_height_m/terrain.vertical_scale # [p]
    pillar_size_m_min = 1 # [m]
    pillar_size_m_max = 5 # [m]
    pillar_size_min = math.floor(pillar_size_m_min/terrain.horizontal_scale) # [p]
    pillar_size_max = math.floor(pillar_size_m_max/terrain.horizontal_scale) # [p]
    pillar_angle_min = 2 # [deg]
    pillar_angle_max = 180 # [deg]

    for i in range(num_obs):
        rand_pos = (random.randint(0, terrain.width), random.randint(0, terrain.length)) # [p]
        rand_pos = np.array(rand_pos)
        pillar_width = random.randint(pillar_size_min, pillar_size_max+1) # [p]
        pillar_height = math.floor(0.2/terrain.horizontal_scale) # [p]
        
        pillar_angle = random.randint(pillar_angle_min, pillar_angle_max) # [deg]
       
        R = np.array([[np.cos(np.radians(pillar_angle)), -np.sin(np.radians(pillar_angle))],[np.sin(np.radians(pillar_angle)), np.cos(np.radians(pillar_angle))]]) # [rad]
        R_inv = np.linalg.inv(R)
          
        x1 = [- math.floor(pillar_width/2)] # [p]
        x2 = [+ math.floor(pillar_width/2)] # [p]
        y1 = [- math.floor(pillar_height/2)] # [p]
        y2 = [+ math.floor(pillar_height/2)] # [p]

        x = range(x1[0], x2[0]+1)
        y = range(y1[0], y2[0]+1)
        
        sides_pos = list(product(x1, y)) + list(product(x2, y)) + list(product(x, y2)) + list(product(x, y1))
        rot_sides_pos = []

        for pos in sides_pos:
            pos = np.array(pos)
            rot_pos = np.dot(R_inv, pos).astype(int) + rand_pos
            rot_pos = np.round(rot_pos)
            rot_sides_pos.append(tuple(rot_pos))

        dict_rot_sides_pos = {}

        for x, y in rot_sides_pos:
            try:
                dict_rot_sides_pos[y].append(x)
            except KeyError:
                dict_rot_sides_pos[y] = [x]

        for y in dict_rot_sides_pos.keys():
            x_max = max(dict_rot_sides_pos[y])
            x_min = min(dict_rot_sides_pos[y])
             
            for x in range(x_min, x_max+1):
                try:
                    terrain.height_field_raw[x,y] = pillar_height
                except IndexError:
                    continue
    count_free = np.count_nonzero(terrain.height_field_raw == 0)
    count_obs = np.count_nonzero(terrain.height_field_raw == pillar_height)
    print("free, obs : {}, {}".format(count_free, count_obs))

    return terrain

# def random_hallway(terrain, _num_obs, _pillar_height):

#     num_obs = _num_obs
#     pillar_height_m = _pillar_height # [m]
#     pillar_height= pillar_height_m/terrain.vertical_scale # [p]
#     side_num_min = 3
#     side_num_max = 5
#     side_length_m_min = 0.3
#     side_length_m_max = 0.8
#     side_length_min = math.floor(side_length_m_min/terrain.horizontal_scale) # [p]
#     side_length_max = math.floor(side_length_m_max/terrain.horizontal_scale) # [p]
#     hallway_width_m_min = 0.6
#     hallway_width_m_max = 1.0
#     hallway_width_min = math.floor(hallway_width_m_min/terrain.horizontal_scale) # [p]
#     hallway_width_max = math.floor(hallway_width_m_max/terrain.horizontal_scale) # [p]

#     # # side wall 
#     # wall_width = 5 # math.floor(0.32/terrain.horizontal_scale) # [p]
#     # terrain.height_field_raw[:, 0:wall_width] = pillar_height
#     # terrain.height_field_raw[:, terrain.width-wall_width:terrain.width] = pillar_height
#     # terrain.height_field_raw[0:wall_width ,:] = pillar_height
#     # terrain.height_field_raw[terrain.length-wall_width:terrain.length ,:] = pillar_height

#     # start = 15

#     # # hallway
#     # while(1):
#     #     hallway_width = random.randint(hallway_width_min, hallway_width_max)
        
#     #     start = start + hallway_width
#     #     end = start + wall_width
        
#     #     if end < terrain.length:
#     #         terrain.height_field_raw[hallway_width:terrain.width-hallway_width, start:end] = pillar_height
#     #         start = end
#     #     else: 
#     #         print("break")
#     #         break

   

#     # 랜덤한 위치, 길이, 각도를 가지는 3개의 직선 그리기
#     for _ in range(3):
        
#         x1 = random.randint(0, terrain.width - 1)
#         y1 = random.randint(0, terrain.height - 1)
#         length = random.randint(50, 200)  # 길이 범위 설정
#         angle = random.uniform(0, 2 * math.pi)  # 랜덤한 각도

#         x2 = int(x1 + length * math.cos(angle))
#         y2 = int(y1 + length * math.sin(angle))
        
#         # rr, cc 배열 생성
#         rr, cc = line(y1, x1, y2, x2)  # skimage.draw.line 사용
        
#         # 유효한 인덱스만 남기기
#         valid_indices = np.where((rr >= 0) & (rr < terrain.width) & (cc >= 0) & (cc < terrain.length))
#         rr, cc = rr[valid_indices], cc[valid_indices]
        
#         terrain.height_field_raw[rr, cc] = pillar_height


#     # for i in range(num_obs):
#     #     rand_pos = (random.randint(0, terrain.width), random.randint(0, terrain.length)) # [p]
#     #     rand_pos = np.array(rand_pos)
        
#     #     side_num = np.random.randint(side_num_min, side_num_max+1)

#     #     # 6각형의 각 변의 길이를 랜덤하게 설정
#     #     side_lengths = np.random.randint(side_length_min, side_length_max, size=side_num)

#     #     # 6각형의 꼭짓점 좌표 계산
#     #     polygon_points = []
#     #     for i in range(side_num):
#     #         angle = 2 * np.pi / 6 * i
#     #         x = int(rand_pos[0] + side_lengths[i] * np.cos(angle))
#     #         y = int(rand_pos[1] + side_lengths[i] * np.sin(angle))
#     #         polygon_points.append((x, y))

#     #     polygon_points = np.array(polygon_points, dtype=np.int32)

#     #     # 6각형 내부를 1로 채우기
#     #     rr, cc = polygon(polygon_points[:, 1], polygon_points[:, 0])
        
#     #     try:
#     #         terrain.height_field_raw[rr, cc] = pillar_height
#     #     except IndexError:
#     #         continue
        
#     return terrain




## set piilar_size
# def square_pillar(terrain, _num_obs, _pillar_height, _pillar_size):

#     num_obs = _num_obs
#     rand_pos_list = []
#     pillar_height = _pillar_height # [m]
#     pillar_size = _pillar_size # [m]

#     for i in range(num_obs):
#         rand_pos = (random.randint(0, terrain.width), random.randint(0, terrain.length)) # [p]
#         rand_pos_list.append(rand_pos)
      
#     for rand_pos in rand_pos_list:
#         x1 = rand_pos[0] - math.floor(pillar_size/terrain.horizontal_scale/2) # [p]
#         x2 = rand_pos[0] + math.floor(pillar_size/terrain.horizontal_scale/2) # [p]
#         y1 = rand_pos[1] - math.floor(pillar_size/terrain.horizontal_scale/2) # [p]
#         y2 = rand_pos[1] + math.floor(pillar_size/terrain.horizontal_scale/2) # [p]
     
#         if math.floor(pillar_size/terrain.horizontal_scale/2) % 2 == 0:
#             x2 += 1
#             y2 += 1

#         if x1 < 0 or x2 <0 or y1 < 0 or y2 <0 or x1 >= terrain.width or x2 >= terrain.width or y1 >= terrain.length or y2 >= terrain.length:
#             print("Delete obstacle which is out of map")
#             continue

#         terrain.height_field_raw[x1:x2, y1:y2] = pillar_height / terrain.vertical_scale
        
#     return terrain

# def circular_pillar(terrain, _num_obs, _pillar_height, _pillar_size):

#     num_obs = _num_obs
#     rand_pos_list = []
#     pillar_height = _pillar_height # [m]
#     pillar_size = _pillar_size # [m]

#     for i in range(num_obs):
#         rand_pos = (random.randint(0, terrain.width), random.randint(0, terrain.length)) # [p]
#         rand_pos_list.append(rand_pos)
    
#     for rand_pos in rand_pos_list:
#         x1 = rand_pos[0] - math.floor(pillar_size/terrain.horizontal_scale/2) # [p]
#         x2 = rand_pos[0] + math.floor(pillar_size/terrain.horizontal_scale/2) # [p]
#         y1 = rand_pos[1] - math.floor(pillar_size/terrain.horizontal_scale/2) # [p]
#         y2 = rand_pos[1] + math.floor(pillar_size/terrain.horizontal_scale/2) # [p]
        
#         if math.floor(pillar_size/terrain.horizontal_scale/2) % 2 == 0:
#             x2 += 1
#             y2 += 1

#         if x1 < 0 or x2 <0 or y1 < 0 or y2 <0 or x1 >= terrain.width or x2 >= terrain.width or y1 >= terrain.length or y2 >= terrain.length:
#             print("Delete obstacle which is out of map")
#             continue

#         for x in range(x1, x2):
#             for y in range(y1, y2):
#                 distance = math.sqrt((x - rand_pos[0])**2 + (y - rand_pos[1]) ** 2)
#                 if distance <= pillar_size/terrain.horizontal_scale / 2:
#                     terrain.height_field_raw[x][y] = pillar_height / terrain.vertical_scale

#     return terrain

# def random_triangle_pillar(terrain, _num_obs, _pillar_height):

#     num_obs = _num_obs
#     rand_pos_list = []
#     pillar_height = _pillar_height # [m]
#     pillar_size_min = 0.1 # [m]
#     pillar_size_max = 1 # [m]
#     pillar_size_min_pixel = math.floor(pillar_size_min/terrain.horizontal_scale) # [p]
#     pillar_size_max_pixel = math.floor(pillar_size_max/terrain.horizontal_scale) # [p]

#     for i in range(num_obs):
#         rand_pos = (random.randint(0, terrain.width), random.randint(0, terrain.length)) # [p]
#         rand_pos_list.append(rand_pos)
    
#     for rand_pos in rand_pos_list:
#         pillar_size_pixel = random.randint(pillar_size_min_pixel, pillar_size_max_pixel+1) # [p]
        
#         height_pixel = math.floor(pillar_size_pixel * (3 ** 0.5) / 2) # [p]
        
#         x = rand_pos[0] # [p]
#         y = rand_pos[1] # [p]

#         for i in range(height_pixel):
#             width = int((pillar_size_pixel - (i / height_pixel) * pillar_size_pixel) / 2) # [p]
#             try:
#                 terrain.height_field_raw[x+i][y-width : y+width] = pillar_height / terrain.vertical_scale
#             except IndexError:
#                 print("Delete obstacle which is out of map")
#                 break        
    
#     return terrain

def random_uniform_terrain(terrain, min_height, max_height, step=1, downsampled_scale=None,):
    """
    Generate a uniform noise terrain

    Parameters
        terrain (SubTerrain): the terrain
        min_height (float): the minimum height of the terrain [meters]
        max_height (float): the maximum height of the terrain [meters]
        step (float): minimum height change between two points [meters]
        downsampled_scale (float): distance between two randomly sampled points ( musty be larger or equal to terrain.horizontal_scale)

    """
    if downsampled_scale is None:
        downsampled_scale = terrain.horizontal_scale

    # switch parameters to discrete units
    min_height = int(min_height / terrain.vertical_scale)
    max_height = int(max_height / terrain.vertical_scale)
    step = int(step / terrain.vertical_scale)

    heights_range = np.arange(min_height, max_height + step, step)
    height_field_downsampled = np.random.choice(heights_range, (int(terrain.width * terrain.horizontal_scale / downsampled_scale), int(
        terrain.length * terrain.horizontal_scale / downsampled_scale)))

    x = np.linspace(0, terrain.width * terrain.horizontal_scale, height_field_downsampled.shape[0])
    y = np.linspace(0, terrain.length * terrain.horizontal_scale, height_field_downsampled.shape[1])

    f = interpolate.interp2d(y, x, height_field_downsampled, kind='linear')

    x_upsampled = np.linspace(0, terrain.width * terrain.horizontal_scale, terrain.width)
    y_upsampled = np.linspace(0, terrain.length * terrain.horizontal_scale, terrain.length)
    z_upsampled = np.rint(f(y_upsampled, x_upsampled))

    terrain.height_field_raw += z_upsampled.astype(np.int16)
    return terrain


def sloped_terrain(terrain, slope=1):
    """
    Generate a sloped terrain

    Parameters:
        terrain (SubTerrain): the terrain
        slope (int): positive or negative slope
    Returns:
        terrain (SubTerrain): update terrain
    """

    x = np.arange(0, terrain.width)
    y = np.arange(0, terrain.length)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = xx.reshape(terrain.width, 1)
    max_height = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * terrain.width)
    terrain.height_field_raw[:, np.arange(terrain.length)] += (max_height * xx / terrain.width).astype(terrain.height_field_raw.dtype)
    return terrain


def pyramid_sloped_terrain(terrain, slope=1, platform_size=1.):
    """
    Generate a sloped terrain

    Parameters:
        terrain (terrain): the terrain
        slope (int): positive or negative slope
        platform_size (float): size of the flat platform at the center of the terrain [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    x = np.arange(0, terrain.width)
    y = np.arange(0, terrain.length)
    center_x = int(terrain.width / 2)
    center_y = int(terrain.length / 2)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = (center_x - np.abs(center_x-xx)) / center_x
    yy = (center_y - np.abs(center_y-yy)) / center_y
    xx = xx.reshape(terrain.width, 1)
    yy = yy.reshape(1, terrain.length)
    max_height = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * (terrain.width / 2))
    terrain.height_field_raw += (max_height * xx * yy).astype(terrain.height_field_raw.dtype)

    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.width // 2 - platform_size
    x2 = terrain.width // 2 + platform_size
    y1 = terrain.length // 2 - platform_size
    y2 = terrain.length // 2 + platform_size

    min_h = min(terrain.height_field_raw[x1, y1], 0)
    max_h = max(terrain.height_field_raw[x1, y1], 0)
    terrain.height_field_raw = np.clip(terrain.height_field_raw, min_h, max_h)
    return terrain


def discrete_obstacles_terrain(terrain, max_height, min_size, max_size, num_rects, platform_size=1.):
    """
    Generate a terrain with gaps

    Parameters:
        terrain (terrain): the terrain
        max_height (float): maximum height of the obstacles (range=[-max, -max/2, max/2, max]) [meters]
        min_size (float): minimum size of a rectangle obstacle [meters]
        max_size (float): maximum size of a rectangle obstacle [meters]
        num_rects (int): number of randomly generated obstacles
        platform_size (float): size of the flat platform at the center of the terrain [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    max_height = int(max_height / terrain.vertical_scale)
    min_size = int(min_size / terrain.horizontal_scale)
    max_size = int(max_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    (i, j) = terrain.height_field_raw.shape
    height_range = [-max_height, -max_height // 2, max_height // 2, max_height]
    width_range = range(min_size, max_size, 4)
    length_range = range(min_size, max_size, 4)

    for _ in range(num_rects):
        width = np.random.choice(width_range)
        length = np.random.choice(length_range)
        start_i = np.random.choice(range(0, i-width, 4))
        start_j = np.random.choice(range(0, j-length, 4))
        terrain.height_field_raw[start_i:start_i+width, start_j:start_j+length] = np.random.choice(height_range)

    x1 = (terrain.width - platform_size) // 2
    x2 = (terrain.width + platform_size) // 2
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0
    return terrain


def wave_terrain(terrain, num_waves=1, amplitude=1.):
    """
    Generate a wavy terrain

    Parameters:
        terrain (terrain): the terrain
        num_waves (int): number of sine waves across the terrain length
    Returns:
        terrain (SubTerrain): update terrain
    """
    amplitude = int(0.5*amplitude / terrain.vertical_scale)
    if num_waves > 0:
        div = terrain.length / (num_waves * np.pi * 2)
        x = np.arange(0, terrain.width)
        y = np.arange(0, terrain.length)
        xx, yy = np.meshgrid(x, y, sparse=True)
        xx = xx.reshape(terrain.width, 1)
        yy = yy.reshape(1, terrain.length)
        terrain.height_field_raw += (amplitude*np.cos(yy / div) + amplitude*np.sin(xx / div)).astype(
            terrain.height_field_raw.dtype)
    return terrain


def stairs_terrain(terrain, step_width, step_height):
    """
    Generate a stairs

    Parameters:
        terrain (terrain): the terrain
        step_width (float):  the width of the step [meters]
        step_height (float):  the height of the step [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    step_width = int(step_width / terrain.horizontal_scale)
    step_height = int(step_height / terrain.vertical_scale)

    num_steps = terrain.width // step_width
    height = step_height
    for i in range(num_steps):
        terrain.height_field_raw[i * step_width: (i + 1) * step_width, :] += height
        height += step_height
    return terrain


def pyramid_stairs_terrain(terrain, step_width, step_height, platform_size=1.):
    """
    Generate stairs

    Parameters:
        terrain (terrain): the terrain
        step_width (float):  the width of the step [meters]
        step_height (float): the step_height [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    step_width = int(step_width / terrain.horizontal_scale)
    step_height = int(step_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    height = 0
    start_x = 0
    stop_x = terrain.width
    start_y = 0
    stop_y = terrain.length
    while (stop_x - start_x) > platform_size and (stop_y - start_y) > platform_size:
        start_x += step_width
        stop_x -= step_width
        start_y += step_width
        stop_y -= step_width
        height += step_height
        terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = height
    return terrain


def stepping_stones_terrain(terrain, stone_size, stone_distance, max_height, platform_size=1., depth=-10):
    """
    Generate a stepping stones terrain

    Parameters:
        terrain (terrain): the terrain
        stone_size (float): horizontal size of the stepping stones [meters]
        stone_distance (float): distance between stones (i.e size of the holes) [meters]
        max_height (float): maximum height of the stones (positive and negative) [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
        depth (float): depth of the holes (default=-10.) [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    stone_size = int(stone_size / terrain.horizontal_scale)
    stone_distance = int(stone_distance / terrain.horizontal_scale)
    max_height = int(max_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    height_range = np.arange(-max_height-1, max_height, step=1)

    start_x = 0
    start_y = 0
    terrain.height_field_raw[:, :] = int(depth / terrain.vertical_scale)
    if terrain.length >= terrain.width:
        while start_y < terrain.length:
            stop_y = min(terrain.length, start_y + stone_size)
            start_x = np.random.randint(0, stone_size)
            # fill first hole
            stop_x = max(0, start_x - stone_distance)
            terrain.height_field_raw[0: stop_x, start_y: stop_y] = np.random.choice(height_range)
            # fill row
            while start_x < terrain.width:
                stop_x = min(terrain.width, start_x + stone_size)
                terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = np.random.choice(height_range)
                start_x += stone_size + stone_distance
            start_y += stone_size + stone_distance
    elif terrain.width > terrain.length:
        while start_x < terrain.width:
            stop_x = min(terrain.width, start_x + stone_size)
            start_y = np.random.randint(0, stone_size)
            # fill first hole
            stop_y = max(0, start_y - stone_distance)
            terrain.height_field_raw[start_x: stop_x, 0: stop_y] = np.random.choice(height_range)
            # fill column
            while start_y < terrain.length:
                stop_y = min(terrain.length, start_y + stone_size)
                terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = np.random.choice(height_range)
                start_y += stone_size + stone_distance
            start_x += stone_size + stone_distance

    x1 = (terrain.width - platform_size) // 2
    x2 = (terrain.width + platform_size) // 2
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0
    return terrain


def convert_heightfield_to_trimesh(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
    """
    Convert a heightfield array to a triangle mesh represented by vertices and triangles.
    Optionally, corrects vertical surfaces above the provide slope threshold:

        If (y2-y1)/(x2-x1) > slope_threshold -> Move A to A' (set x1 = x2). Do this for all directions.
                   B(x2,y2)
                  /|
                 / |
                /  |
        (x1,y1)A---A'(x2',y1)

    Parameters:
        height_field_raw (np.array): input heightfield
        horizontal_scale (float): horizontal scale of the heightfield [meters]
        vertical_scale (float): vertical scale of the heightfield [meters]
        slope_threshold (float): the slope threshold above which surfaces are made vertical. If None no correction is applied (default: None)
    Returns:
        vertices (np.array(float)): array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
        triangles (np.array(int)): array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
    """
    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    y = np.linspace(0, (num_cols-1)*horizontal_scale, num_cols) # [m]
    x = np.linspace(0, (num_rows-1)*horizontal_scale, num_rows) # [m]
    yy, xx = np.meshgrid(y, x) # [m]

    if slope_threshold is not None:

        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[:num_rows-1, :] += (hf[1:num_rows, :] - hf[:num_rows-1, :] > slope_threshold)
        move_x[1:num_rows, :] -= (hf[:num_rows-1, :] - hf[1:num_rows, :] > slope_threshold)
        move_y[:, :num_cols-1] += (hf[:, 1:num_cols] - hf[:, :num_cols-1] > slope_threshold)
        move_y[:, 1:num_cols] -= (hf[:, :num_cols-1] - hf[:, 1:num_cols] > slope_threshold)
        move_corners[:num_rows-1, :num_cols-1] += (hf[1:num_rows, 1:num_cols] - hf[:num_rows-1, :num_cols-1] > slope_threshold)
        move_corners[1:num_rows, 1:num_cols] -= (hf[:num_rows-1, :num_cols-1] - hf[1:num_rows, 1:num_cols] > slope_threshold)
        xx += (move_x + move_corners*(move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners*(move_y == 0)) * horizontal_scale

    # create triangle mesh vertices and triangles from the heightfield grid
    vertices = np.zeros((num_rows*num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2*(num_rows-1)*(num_cols-1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols-1) + i*num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2*i*(num_cols-1)
        stop = start + 2*(num_cols-1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start+1:stop:2, 0] = ind0
        triangles[start+1:stop:2, 1] = ind2
        triangles[start+1:stop:2, 2] = ind3

    return vertices, triangles


class SubTerrain:
    def __init__(self, terrain_name="terrain", width=256, length=256, vertical_scale=1.0, horizontal_scale=1.0):
        self.terrain_name = terrain_name
        self.vertical_scale = vertical_scale
        self.horizontal_scale = horizontal_scale
        self.width = width
        self.length = length
        self.height_field_raw = np.zeros((self.width, self.length), dtype=np.int16)
