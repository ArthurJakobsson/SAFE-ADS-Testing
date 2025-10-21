from beamngpy import BeamNGpy, Scenario, Vehicle, Road, ScenarioObject, MeshRoad
from beamngpy.sensors import Damage, Camera
import numpy as np
import time
import argparse
from datetime import datetime
import os
import yaml
import math
from math import sin, cos
import cv2
import numpy as np
from PIL import Image
import time
import pyautogui
import pickle
from pathlib import Path
import concurrent.futures
import random

def Lane(S_point, Lane_len, Lane_width, lane_num, towards='Hor'):
    lane_list = []
    for i in range(lane_num):
        random_int = np.random.randint(1000)
        road_name = 'Lane_' + towards + str(i) + '_' + str(random_int)
        # road = Road('track_editor_C_border', rid=road_name, texture_length=5)
        road = Road('track_editor_C_center', rid=road_name, texture_length=1)

        # Calculate End point
        if towards == 'Hor':
            E_point = (S_point[0] + Lane_len ,S_point[1],S_point[2])
            road.add_nodes(
                (*S_point, Lane_width),
                (*E_point, Lane_width)
            )
            lane_list.append(road)
            S_point = (S_point[0], S_point[1] + Lane_width, S_point[2])
        elif towards == 'Ver':
            E_point = (S_point[0], S_point[1] - Lane_len, S_point[2])
            road.add_nodes(
                (*S_point, Lane_width),
                (*E_point, Lane_width)
            )
            lane_list.append(road)
            S_point = (S_point[0] + Lane_width, S_point[1], S_point[2])
    return lane_list

def Straight(road_structure, towards, middle_point):
    '''
    Args:
        road_structure:
            Road_type: Straight
            Left_lane:
                - Num: 2
                  len: 100
                  width: 4
            Right_lane:
                - Num: 1
                  len: 100
                  width: 4
        towards: Hor or Ver
        middle_point: default(-107, 20, 100)

    Returns:
        road_list
    '''
    road_list = []

    # Get road args
    left_len = road_structure['Left_lane'][0]['len']
    left_width = road_structure['Left_lane'][0]['width']
    left_num = road_structure['Left_lane'][0]['Num']

    right_len = road_structure['Right_lane'][0]['len']
    right_width = road_structure['Right_lane'][0]['width']
    right_num = road_structure['Right_lane'][0]['Num']

    # Build roads
    if towards == 'Hor':
        S_point_Up = (middle_point[0] - left_len/2, middle_point[1] + left_width/2, middle_point[2])
        S_point_Down = (middle_point[0] - right_len/2, middle_point[1] - (right_num - 0.5) * right_width, middle_point[2])
        if left_num != 0:
            lane_list_1 = Lane(S_point_Up, left_len, left_width, left_num, 'Hor')
            for lane in lane_list_1:
                road_list.append(lane)
        if right_num != 0:
            lane_list_2 = Lane(S_point_Down, right_len, right_width, right_num, 'Hor')
            for lane in lane_list_2:
                road_list.append(lane)
    elif towards == 'Ver':
        S_point_Up = (middle_point[0] + left_width/2 , middle_point[1] + left_len / 2, middle_point[2])
        S_point_Down = (middle_point[0] - (right_num - 0.5) * right_width, middle_point[1] + right_len / 2, middle_point[2])
        if left_num != 0:
            lane_list_1 = Lane(S_point_Up, left_len, left_width, left_num, 'Ver')
            for lane in lane_list_1:
                road_list.append(lane)
        if right_num != 0:
            lane_list_2 = Lane(S_point_Down, right_len, right_width, right_num, 'Ver')
            for lane in lane_list_2:
                road_list.append(lane)

    return road_list

def Intersection(road_structure, middle_point):
    road_list = []
    # Calculate Start point for Hor
    Hor_road_list = Straight(road_structure['Hor_lane'][0], 'Hor', middle_point)

    # Calculate Start point for Ver
    Ver_road_list = Straight(road_structure['Ver_lane'][0], 'Ver', middle_point)

    for road in Hor_road_list:
        road_list.append(road)

    for road in Ver_road_list:
        road_list.append(road)

    return road_list

def convert_trajectory_to_npc_script(trajectory):
    npc_script = []
    t = 1

    for point in trajectory:
        x, y = point
        z = 100
        npc_script.append({'x': x, 'y': y, 'z': z, 't': t})
        t += 1

    return npc_script

def calculate_unit_quaternion(trajectory):
    y_dist = abs(trajectory[-1][-1] - trajectory[0][-1])
    x_dist = abs(trajectory[-1][0] - trajectory[0][0])

    if y_dist >= x_dist:
        if trajectory[-1][-1] - trajectory[0][-1] >= 0:
            return (0, 0, 1, 0)
        else:
            return (0, 0, 0, 1)
    else:
        if trajectory[-1][0] - trajectory[0][0] >= 0:
            return (0, 0, -0.7, 0.7)
        else:
            return (0, 0, 0.7, 0.7)

def adjust_coordinates(car_dict, ori_point):
    x_offset, y_offset, _ = ori_point
    adjusted_car_dict = {}
    for vehicle, data in car_dict.items():
        coordinates, vehicle_type = data
        adjusted_coordinates = [[x + x_offset, y + y_offset] for x, y in coordinates]
        adjusted_car_dict[vehicle] = [adjusted_coordinates, vehicle_type]

    return adjusted_car_dict

def convert_string_to_list_of_lists(string):
    string = string.strip('[]')
    list_of_tuples = eval(string)
    list_of_lists = [list(item) for item in list_of_tuples]
    return list_of_lists

def get_car_info(actors):
    if 'Validation' in actors:
        no_cars = int((len(actors) - 1) / 2)
    else:
        no_cars = int(len(actors) / 2)

    car_dict = {}
    for i in range(no_cars):
        car_dict[f"V{i + 1}"] = [
            convert_string_to_list_of_lists(actors[f"V{i + 1}_traj"]),
            actors[f"V{i + 1}_type"]
        ]
    return car_dict

def load_yaml_files_to_dict(directory_path):
    yaml_dict = {}

    for filename in os.listdir(directory_path):
        if filename.endswith(".yaml"):
            file_id = os.path.splitext(filename)[0]
            file_path = os.path.join(directory_path, filename)

            with open(file_path, 'r', encoding='utf-8') as file:
                content = yaml.safe_load(file)
                yaml_dict[file_id] = content

    return yaml_dict

def run_intersection(bng, road_network, env_info, actors, ID, ADS, folder_path):
    scenario = Scenario('gridmap_v2', 'Intersection')
    ori_point = (-1756, -52, 100)

    lane_per_way = road_network.get('No_lanes', 2)
    lane_width = road_network.get('Width', 4)
    intersection_length = road_network.get('Length', 40)
    yellow_sep_width = 0.25

    total_lane_width = lane_per_way * lane_width + (lane_per_way - 1) * yellow_sep_width
    half_total_lane_width = total_lane_width / 2

    west_start = (ori_point[0] - intersection_length / 2, ori_point[1], ori_point[2])
    east_end = (ori_point[0] + intersection_length / 2, ori_point[1], ori_point[2])

    south_start = (ori_point[0], ori_point[1] - intersection_length / 2, ori_point[2])
    north_end = (ori_point[0], ori_point[1] + intersection_length / 2, ori_point[2])

    road_west = Road('track_editor_C_center', rid='road_west')
    road_west.nodes.extend([
        (west_start[0], west_start[1], west_start[2], total_lane_width),
        (ori_point[0], ori_point[1], ori_point[2], total_lane_width)
    ])
    scenario.add_road(road_west)

    road_east = Road('track_editor_C_center', rid='road_east')
    road_east.nodes.extend([
        (ori_point[0], ori_point[1], ori_point[2], total_lane_width),
        (east_end[0], east_end[1], east_end[2], total_lane_width)
    ])
    scenario.add_road(road_east)

    road_south = Road('track_editor_C_center', rid='road_south')
    road_south.nodes.extend([
        (south_start[0], south_start[1], south_start[2], total_lane_width),
        (ori_point[0], ori_point[1], ori_point[2], total_lane_width)
    ])
    scenario.add_road(road_south)

    road_north = Road('track_editor_C_center', rid='road_north')
    road_north.nodes.extend([
        (ori_point[0], ori_point[1], ori_point[2], total_lane_width),
        (north_end[0], north_end[1], north_end[2], total_lane_width)
    ])
    scenario.add_road(road_north)

    car_dict = get_car_info(actors)
    car_dict = adjust_coordinates(car_dict, ori_point)

    ego_vehicle_id = 'V1'
    ego_data = car_dict[ego_vehicle_id]
    ego_model = 'etk800' if ego_data[1] == 'Car' else 'citybus'
    ego = Vehicle('ego', model=ego_model, color='Green')
    ego_start_point = ego_data[0][0]
    pos = (ego_start_point[0], ego_start_point[1], ori_point[2] + 0.5)
    rot = calculate_unit_quaternion(ego_data[0])
    damage = Damage()
    ego.attach_sensor('damage', damage)
    scenario.add_vehicle(ego, pos=pos, rot_quat=rot)

    npc_vehicles = []
    npc_vehicle_ids = [vid for vid in car_dict if vid != ego_vehicle_id]
    for idx, npc_id in enumerate(npc_vehicle_ids):
        npc_data = car_dict[npc_id]
        npc_model = 'etk800' if npc_data[1] == 'Car' else 'citybus'
        npc = Vehicle(f'npc_{idx}', model=npc_model)
        damage_npc = Damage()
        npc.attach_sensor(f'damage_npc_{idx}', damage_npc)
        npc_start_point = npc_data[0][0]
        pos = (npc_start_point[0], npc_start_point[1], ori_point[2] + 0.5)
        rot = calculate_unit_quaternion(npc_data[0])
        scenario.add_vehicle(npc, pos=pos, rot_quat=rot)
        npc_vehicles.append((npc, npc_data[0]))

    scenario.make(bng)
    bng.load_scenario(scenario)
    bng.start_scenario()

    day_time_mapping = {
        'Daytime': 10,
        'Not mentioned': 10,
        'Nighttime': 50
    }
    time_of_day = day_time_mapping.get(env_info.get('Time'), 10)
    bng.env.set_tod(time_of_day / 100)

    camera_position = (ori_point[0], ori_point[1], ori_point[2] + 50)
    bng.camera.set_free(camera_position, (0, 0, -1))

    time.sleep(1)

    ego.connect(bng)
    for npc, _ in npc_vehicles:
        npc.connect(bng)

    ego_script = convert_trajectory_to_npc_script(ego_data[0])
    ego.ai_set_script(ego_script)

    for npc, npc_traj in npc_vehicles:
        npc_script = convert_trajectory_to_npc_script(npc_traj)
        npc.ai_set_script(npc_script)

    time.sleep(15)

def Create_intersection(road_network,ori_point,scenario):
    lane_len = road_network['Length']
    lane_per_way = (road_network.get('No_lanes', 1) + 1) // 2 if road_network.get('No_lanes',
                                                                                  1) % 2 != 0 else road_network.get(
        'No_lanes', 1) // 2
    lane_width = road_network['Width']
    road_structure = {'Road_type': 'Intersection',
                      'Ver_lane': [{'Left_lane': [{'Num': lane_per_way, 'len': lane_len, 'width': lane_width}],
                                    'Right_lane': [{'Num': lane_per_way, 'len': lane_len, 'width': lane_width}]}],
                      'Hor_lane': [{'Left_lane': [{'Num': lane_per_way, 'len': lane_len, 'width': lane_width}],
                                    'Right_lane': [{'Num': lane_per_way, 'len': lane_len, 'width': lane_width}]}]
                      }
    road_list = Intersection(road_structure, ori_point)
    for roads in road_list:
        scenario.add_road(roads)

    camera_view_point = (ori_point[0], ori_point[1], 140)
    return scenario, camera_view_point

def Create_straight(road_network,ori_point,scenario):
    # Get road settings
    lane_per_way = (road_network.get('No_lanes', 1) + 1) // 2 if road_network.get('No_lanes',
                                                                                  1) % 2 != 0 else road_network.get(
        'No_lanes', 1) // 2
    if lane_per_way == 1:
        no_dash_line = 0
    else:
        no_dash_line = lane_per_way - 1

    # some settings of road markings
    dash_width = 0.25
    lane_width = road_network.get('Width', 4)
    off_road_width = 2.5
    # lane_length = road_network.get('Width', 50) * 10
    lane_length = road_network.get('Length', 50)
    yellow_sep_width = 0.25
    yellow_sep_middle_x_n_Y = lane_per_way * lane_width + no_dash_line * dash_width + ori_point[1] + (
                yellow_sep_width / 2)
    yellow_sep_middle_x_n = (ori_point[0], yellow_sep_middle_x_n_Y, ori_point[2], yellow_sep_width)
    yellow_sep_middle_x_p = ((ori_point[0] + lane_length), yellow_sep_middle_x_n_Y, ori_point[2], yellow_sep_width)

    camera_view_point = ((ori_point[0] + lane_length / 2), yellow_sep_middle_x_n_Y, 140)

    # Draw yellow sep line
    Yellow_sep = Road('m_gm_line_yellow', rid='Yellow_sep')
    Yellow_sep.nodes.extend([yellow_sep_middle_x_n, yellow_sep_middle_x_p])
    scenario.add_road(Yellow_sep)

    # Draw lane
    if lane_per_way == 1:
        lane_x_n_y_n = (ori_point[0], (ori_point[1] + lane_width / 2), ori_point[2], lane_width)
        lane_x_p_y_n = ((ori_point[0] + lane_length), (ori_point[1] + lane_width / 2), ori_point[2], lane_width)
        Lane_y_n = Road('track_editor_C_center', rid='Lane_y_n')
        Lane_y_n.nodes.extend([lane_x_n_y_n, lane_x_p_y_n])
        scenario.add_road(Lane_y_n)

        lane_x_n_y_p = (ori_point[0], (ori_point[1] + yellow_sep_width + 1.5 * lane_width), ori_point[2], lane_width)
        lane_x_p_y_p = (
        (ori_point[0] + lane_length), (ori_point[1] + yellow_sep_width + 1.5 * lane_width), ori_point[2], lane_width)
        Lane_y_p = Road('track_editor_C_center', rid='Lane_y_p')
        Lane_y_p.nodes.extend([lane_x_n_y_p, lane_x_p_y_p])
        scenario.add_road(Lane_y_p)
    else:
        for lane_id in range(lane_per_way):
            # Y N
            y_extension = lane_id * (lane_width + dash_width)
            lane_x_n_y_n = (ori_point[0], (ori_point[1] + lane_width / 2 + y_extension), ori_point[2], lane_width)
            lane_x_p_y_n = (
            (ori_point[0] + lane_length), (ori_point[1] + lane_width / 2 + y_extension), ori_point[2], lane_width)
            Lane_y_ = Road('track_editor_C_center', rid=f'Lane_y_n_{lane_id}')
            Lane_y_.nodes.extend([lane_x_n_y_n, lane_x_p_y_n])
            scenario.add_road(Lane_y_)
            # Y P
            lane_x_n_y_p = (
            ori_point[0], ((yellow_sep_middle_x_n_Y + yellow_sep_width / 2) + lane_width / 2 + y_extension),
            ori_point[2], lane_width)
            lane_x_p_y_p = (
                (ori_point[0] + lane_length),
                ((yellow_sep_middle_x_n_Y + yellow_sep_width / 2) + lane_width / 2 + y_extension), ori_point[2],
                lane_width)
            Lane_y_ = Road('track_editor_C_center', rid=f'Lane_y_p_{lane_id}_p')
            Lane_y_.nodes.extend([lane_x_n_y_p, lane_x_p_y_p])
            scenario.add_road(Lane_y_)

        for dash_id in range(no_dash_line):
            # Y N
            y_extension = dash_id * (lane_width + dash_width)
            dash_x_n_y_n = (
            ori_point[0], (ori_point[1] + dash_width / 2 + lane_width + y_extension), ori_point[2], dash_width)
            dash_x_p_y_n = (
            (ori_point[0] + lane_length), (ori_point[1] + dash_width / 2 + lane_width + y_extension), ori_point[2],
            dash_width)
            dash_y_n = Road('m_gm_line', rid=f'Dash_y_n_{dash_id}')
            dash_y_n.nodes.extend([dash_x_n_y_n, dash_x_p_y_n])
            scenario.add_road(dash_y_n)
            # Y P
            dash_x_n_y_p = (
                ori_point[0],
                ((yellow_sep_middle_x_n_Y + yellow_sep_width / 2) + dash_width / 2 + lane_width + y_extension),
                ori_point[2], dash_width)
            dash_x_p_y_p = (
                (ori_point[0] + lane_length),
                ((yellow_sep_middle_x_n_Y + yellow_sep_width / 2) + dash_width / 2 + lane_width + y_extension),
                ori_point[2],
                dash_width)
            dash_y_p = Road('m_gm_line', rid=f'Dash_y_p_{dash_id}')
            dash_y_p.nodes.extend([dash_x_n_y_p, dash_x_p_y_p])
            scenario.add_road(dash_y_p)

    return scenario, camera_view_point

def Create_tintersection(road_network,ori_point,scenario):
    lane_len_h = road_network['Length_main']
    lane_per_way_h = (road_network.get('No_lanes_main_road', 1) + 1) // 2 if road_network.get('No_lanes_main_road',
                                                                                  1) % 2 != 0 else road_network.get(
        'No_lanes_main_road', 1) // 2
    lane_len_v = road_network['Length_branch']
    lane_per_way_v = (road_network.get('No_lanes_branch_road', 1) + 1) // 2 if road_network.get('No_lanes_branch_road',
                                                                                              1) % 2 != 0 else road_network.get(
        'No_lanes_branch_road', 1) // 2
    lane_width = road_network['Width']
    road_structure = {'Road_type': 'T-Intersection',
                      'Ver_lane': [{'Left_lane': [{'Num': lane_per_way_v, 'len': lane_len_v, 'width': lane_width}],
                                    'Right_lane': [{'Num': lane_per_way_v, 'len': lane_len_v, 'width': lane_width}]}],
                      'Hor_lane': [{'Left_lane': [{'Num': lane_per_way_h, 'len': lane_len_h, 'width': lane_width}],
                                    'Right_lane': [{'Num': lane_per_way_h, 'len': lane_len_h, 'width': lane_width}]}]}
    road_list = T_intersection(road_structure, ori_point)
    for roads in road_list:
        scenario.add_road(roads)

    camera_view_point = (ori_point[0], ori_point[1], 140)
    return scenario, camera_view_point

def check_crash(damage_data):
    total_damage = damage_data.get('damage', 0)
    has_total_damage = total_damage > 0
    part_damage = damage_data.get('part_damage', {})
    if isinstance(part_damage, dict):
        has_part_damage = any(part['damage'] > 0 for part in part_damage.values())
    else:
        has_part_damage = False
    deform_group_damage = damage_data.get('deform_group_damage', {})
    has_deform_damage = any(group['damage'] > 0 for group in deform_group_damage.values())
    extended_damage = damage_data.get('damage_ext', 0)
    has_extended_damage = extended_damage > 0
    has_damage = has_total_damage or has_part_damage or has_deform_damage or has_extended_damage
    return 1 if has_damage else 0

def T_intersection(road_structure, middle_point):
    road_list = []
    # Calculate Start point for Hor
    Hor_road_list = Straight(road_structure['Hor_lane'][0], 'Hor', middle_point)

    # Calculate Start point for Ver
    width = road_structure['Hor_lane'][0]['Right_lane'][0]['width']
    num = road_structure['Hor_lane'][0]['Right_lane'][0]['Num']
    len_lane = road_structure['Hor_lane'][0]['Right_lane'][0]['len']
    ver_middle_point = (middle_point[0], middle_point[1] - num * width - len_lane * 0.5, middle_point[2])
    Ver_road_list = Straight(road_structure['Ver_lane'][0], 'Ver', ver_middle_point)

    for road in Hor_road_list:
        road_list.append(road)

    for road in Ver_road_list:
        road_list.append(road)

    return road_list

def run_merge(actors, road_network, ori_point, scenario,bng,env_info,case_result_folder):
    main_display_width = 2560
    main_display_height = 1440
    screen_left = 0
    screen_top = 0
    screen_width = main_display_width
    screen_height = main_display_height

    car_dict = get_car_info(actors)
    # Bild maps
    s_starting = (ori_point[0],ori_point[1],ori_point[2],8)
    s_ending = (ori_point[0]+100,ori_point[1],ori_point[2],8)
    m_starting = (ori_point[0],ori_point[1]-15,ori_point[2],8)
    m_ending = (ori_point[0]+50,ori_point[1],ori_point[2],8)
    s_road = Road('track_editor_C_center', rid='s_lane')
    s_road.nodes.extend([s_starting,s_ending])
    scenario.add_road(s_road)
    m_road = Road('track_editor_C_center', rid='m_lane')
    m_road.nodes.extend([m_starting, m_ending])
    scenario.add_road(m_road)

    print('Original Vehicle Config:')
    print(car_dict)
    y_diff = abs(car_dict['V1'][0][1][1] - car_dict['V1'][0][-1][1])
    if y_diff < 3:
        # V 1 trajectories
        car_dict['V1'][0] = [[ori_point[0],ori_point[1]-2],[ori_point[0]+25,ori_point[1]-2],[(ori_point[0]+99),ori_point[1]-2]]
        car_dict['V2'][0] = [[ori_point[0],ori_point[1]-15],[ori_point[0]+5,ori_point[1]-12],[(ori_point[0]+50),ori_point[1]]]
    else:
        car_dict['V2'][0] = [[ori_point[0], ori_point[1] - 2], [ori_point[0] + 25, ori_point[1] - 2],
                             [(ori_point[0] + 99), ori_point[1] - 2]]
        car_dict['V1'][0] = [[ori_point[0], ori_point[1] - 15], [ori_point[0] + 5, ori_point[1] - 12],
                             [(ori_point[0] + 50), ori_point[1]]]

    if car_dict['V1'][1] == 'Car':
        ego_model = 'etk800'
    else:
        ego_model = 'citybus'

    # Ego car
    ego = Vehicle('ego', model=ego_model, color='Green')
    ego_start_point = car_dict['V1'][0][0]
    pos = (ego_start_point[0], ego_start_point[1], 100.5)
    rot = calculate_unit_quaternion(car_dict['V1'][0])
    ego.sensors.attach("damage", Damage())
    scenario.add_vehicle(ego, pos=pos, rot_quat=rot)

    # NPC cars
    # V2
    npc_model = 'etk800' if car_dict['V2'][1] == 'Car' else 'citybus'
    npc = Vehicle('npc', model=npc_model)
    damage_npc = Damage()
    npc.attach_sensor('damage_npc', damage_npc)
    npc_start_point = car_dict['V2'][0][0]
    pos = (npc_start_point[0], npc_start_point[1], 100.5)
    rot = calculate_unit_quaternion(car_dict['V2'][0])
    scenario.add_vehicle(npc, pos=pos, rot_quat=rot)

    # Generate scenario
    scenario.make(bng)
    bng.load_scenario(scenario)
    bng.start_scenario()

    day_time_mapping = {
        'Daytime': 10,
        'Not mentioned': 10,
        'Nighttime': 50
    }

    t = day_time_mapping.get(env_info.get('Time'), 10)
    bng.env.set_tod(t / 100)

    camera_view_point = (ori_point[0]+50, ori_point[1]-5, 140)

    # Set monitor camera
    bng.camera.set_free(camera_view_point, (0, 0, -1))

    npc_script = convert_trajectory_to_npc_script(car_dict['V2'][0])
    npc.ai_set_script(npc_script)

    ego_script = convert_trajectory_to_npc_script(car_dict['V1'][0])
    ego.ai_set_script(ego_script)

    print('Simulation starts!')
    time.sleep(1)
    screenshot = pyautogui.screenshot(region=(screen_left, screen_top, screen_width, screen_height))
    save_path = os.path.join(case_result_folder, "1.png")
    screenshot.save(save_path)
    time.sleep(1)
    screenshot = pyautogui.screenshot(region=(screen_left, screen_top, screen_width, screen_height))
    save_path = os.path.join(case_result_folder, "2.png")
    screenshot.save(save_path)
    time.sleep(1)
    screenshot = pyautogui.screenshot(region=(screen_left, screen_top, screen_width, screen_height))
    save_path = os.path.join(case_result_folder, "3.png")
    screenshot.save(save_path)
    time.sleep(1)
    screenshot = pyautogui.screenshot(region=(screen_left, screen_top, screen_width, screen_height))
    save_path = os.path.join(case_result_folder, "4.png")
    screenshot.save(save_path)
    time.sleep(1)
    screenshot = pyautogui.screenshot(region=(screen_left, screen_top, screen_width, screen_height))
    save_path = os.path.join(case_result_folder, "5.png")
    screenshot.save(save_path)
    time.sleep(1)
    screenshot = pyautogui.screenshot(region=(screen_left, screen_top, screen_width, screen_height))
    save_path = os.path.join(case_result_folder, "6.png")
    screenshot.save(save_path)
    time.sleep(1)
    screenshot = pyautogui.screenshot(region=(screen_left, screen_top, screen_width, screen_height))
    save_path = os.path.join(case_result_folder, "7.png")
    screenshot.save(save_path)
    time.sleep(5)
    screenshot = pyautogui.screenshot(region=(screen_left, screen_top, screen_width, screen_height))
    save_path = os.path.join(case_result_folder, "8.png")
    screenshot.save(save_path)
    print('Simulation ends!')
    # ego.poll_sensors('damage')
    ego.sensors.poll()
    damage_data = ego.sensors["damage"].data
    crash_flag = check_crash(damage_data)
    if crash_flag:
        print('Crash detacted!')
    else:
        print('Crash Not Found!')
    print('=====================')
    return crash_flag

def run_curve(actors, road_network, ori_point, scenario,bng,env_info,case_result_folder):
    main_display_width = 2560
    main_display_height = 1440
    screen_left = 0
    screen_top = 0
    screen_width = main_display_width
    screen_height = main_display_height

    car_dict = get_car_info(actors)
    # Bild maps
    road_starting_point = (ori_point[0], ori_point[1], ori_point[2], 8)
    road_ending_point = ((ori_point[0] + 100), ori_point[1], ori_point[2], 8)
    road_middle_point = ((ori_point[0] + 50), (ori_point[1] + 25), ori_point[2], 8)
    Lane_curve = Road('track_editor_C_center', rid='Lane_y_n')
    Lane_curve.nodes.extend([road_starting_point, road_middle_point, road_ending_point])
    scenario.add_road(Lane_curve)

    print('Original Vehicle Config:')
    print(car_dict)
    x_diff = car_dict['V1'][0][1][0] - car_dict['V1'][0][0][0]
    if x_diff > 0:
        # V 1 trajectories
        car_dict['V1'][0] = [[ori_point[0], ori_point[1]], [ori_point[0] + 5, ori_point[1] + 3],
                             [(ori_point[0] + 51), ori_point[1] + 24]]
        car_dict['V2'][0] = [[ori_point[0] + 101, ori_point[1]], [ori_point[0] + 95, ori_point[1] + 3],
                             [(ori_point[0] + 50), ori_point[1] + 24]]
        if len(car_dict) == 3:
            car_dict['V3'][0] = [[ori_point[0] + 103, ori_point[1]], [ori_point[0] + 95, ori_point[1] + 4],
                                 [(ori_point[0] + 50), ori_point[1] + 24]]
    else:
        # V 1 trajectories
        car_dict['V2'][0] = [[ori_point[0], ori_point[1]], [ori_point[0] + 5, ori_point[1] + 3],
                             [(ori_point[0] + 51), ori_point[1] + 24]]
        car_dict['V1'][0] = [[ori_point[0] + 101, ori_point[1]], [ori_point[0] + 95, ori_point[1] + 3],
                             [(ori_point[0] + 50), ori_point[1] + 24]]
        if len(car_dict) == 3:
            car_dict['V3'][0] = [[ori_point[0] + 103, ori_point[1]], [ori_point[0] + 95, ori_point[1] + 4],
                                 [(ori_point[0] + 50), ori_point[1] + 24]]
    if car_dict['V1'][1] == 'Car':
        ego_model = 'etk800'
    else:
        ego_model = 'citybus'

    # Ego car
    ego = Vehicle('ego', model=ego_model, color='Green')
    ego_start_point = car_dict['V1'][0][0]
    pos = (ego_start_point[0], ego_start_point[1], 100.5)
    rot = calculate_unit_quaternion(car_dict['V1'][0])
    ego.sensors.attach("damage", Damage())
    scenario.add_vehicle(ego, pos=pos, rot_quat=rot)

    # NPC cars
    # V2
    npc_model = 'etk800' if car_dict['V2'][1] == 'Car' else 'citybus'
    npc = Vehicle('npc', model=npc_model)
    damage_npc = Damage()
    npc.attach_sensor('damage_npc', damage_npc)
    npc_start_point = car_dict['V2'][0][0]
    pos = (npc_start_point[0], npc_start_point[1], 100.5)
    rot = calculate_unit_quaternion(car_dict['V2'][0])
    scenario.add_vehicle(npc, pos=pos, rot_quat=rot)

    # V3 if we have
    if len(car_dict) == 3:
        npc_model = 'etk800' if car_dict['V3'][1] == 'Car' else 'citybus'
        npc_2 = Vehicle('npc_2', model=npc_model)
        damage_npc_2 = Damage()
        npc_2.attach_sensor('damage_npc_2', damage_npc_2)
        npc_start_point = car_dict['V3'][0][0]
        pos = (npc_start_point[0], npc_start_point[1], 100.5)
        rot = calculate_unit_quaternion(car_dict['V3'][0])
        scenario.add_vehicle(npc_2, pos=pos, rot_quat=rot)

        # Generate scenario
    scenario.make(bng)
    bng.load_scenario(scenario)
    bng.start_scenario()

    day_time_mapping = {
        'Daytime': 10,
        'Not mentioned': 10,
        'Nighttime': 50
    }

    t = day_time_mapping.get(env_info.get('Time'), 10)
    bng.env.set_tod(t / 100)

    camera_view_point = (ori_point[0] + 50, ori_point[1], 140)

    # Set monitor camera
    bng.camera.set_free(camera_view_point, (0, 0, -1))

    npc_script = convert_trajectory_to_npc_script(car_dict['V2'][0])
    npc.ai_set_script(npc_script)

    ego_script = convert_trajectory_to_npc_script(car_dict['V1'][0])
    ego.ai_set_script(ego_script)
    # V3 if we have
    if len(car_dict) == 3:
        npc_script_2 = convert_trajectory_to_npc_script(car_dict['V3'][0])
        npc_2.ai_set_script(npc_script_2)

    print('Simulation starts!')
    time.sleep(1)
    screenshot = pyautogui.screenshot(region=(screen_left, screen_top, screen_width, screen_height))
    save_path = os.path.join(case_result_folder, "1.png")
    screenshot.save(save_path)
    time.sleep(1)
    screenshot = pyautogui.screenshot(region=(screen_left, screen_top, screen_width, screen_height))
    save_path = os.path.join(case_result_folder, "2.png")
    screenshot.save(save_path)
    time.sleep(1)
    screenshot = pyautogui.screenshot(region=(screen_left, screen_top, screen_width, screen_height))
    save_path = os.path.join(case_result_folder, "3.png")
    screenshot.save(save_path)
    time.sleep(1)
    screenshot = pyautogui.screenshot(region=(screen_left, screen_top, screen_width, screen_height))
    save_path = os.path.join(case_result_folder, "4.png")
    screenshot.save(save_path)
    time.sleep(1)
    screenshot = pyautogui.screenshot(region=(screen_left, screen_top, screen_width, screen_height))
    save_path = os.path.join(case_result_folder, "5.png")
    screenshot.save(save_path)
    time.sleep(1)
    screenshot = pyautogui.screenshot(region=(screen_left, screen_top, screen_width, screen_height))
    save_path = os.path.join(case_result_folder, "6.png")
    screenshot.save(save_path)
    time.sleep(1)
    screenshot = pyautogui.screenshot(region=(screen_left, screen_top, screen_width, screen_height))
    save_path = os.path.join(case_result_folder, "7.png")
    screenshot.save(save_path)
    time.sleep(5)
    screenshot = pyautogui.screenshot(region=(screen_left, screen_top, screen_width, screen_height))
    save_path = os.path.join(case_result_folder, "8.png")
    screenshot.save(save_path)
    print('Simulation ends!')
    # ego.poll_sensors('damage')
    ego.sensors.poll()
    damage_data = ego.sensors["damage"].data
    crash_flag = check_crash(damage_data)
    if crash_flag:
        print('Crash detacted!')
    else:
        print('Crash Not Found!')
    print('=====================')
    return crash_flag

def run_scenario(bng, road_network, env_info, actors, road_type, ID, folder_path, seq):

    case_result_folder = os.path.join(folder_path, f'{ID}_{seq}')
    os.makedirs(case_result_folder, exist_ok=True)
    main_display_width = 2560
    main_display_height = 1440
    screen_left = 0
    screen_top = 0
    screen_width = main_display_width
    screen_height = main_display_height

    print('=====================')
    print(f'Reconstruct case: {ID} No.{seq}')
    print(f'Case road type: {road_type}')
    # Initialize scenario
    scenario = Scenario('gridmap_v2', 'ADS_testing')
    # Setup origin point
    ori_point = (-1756, -52, 100)

    if road_type == 'Straight':
        scenario, camera_view_point = Create_straight(road_network,ori_point,scenario)
    elif road_type == 'Intersection':
        scenario, camera_view_point = Create_intersection(road_network, ori_point, scenario)
    elif road_type == 'T-intersection':
        scenario, camera_view_point = Create_tintersection(road_network, ori_point, scenario)
    elif road_type == 'Curve':
        crash_flag = run_curve(actors, road_network, ori_point, scenario,bng,env_info,case_result_folder)
        return crash_flag
    elif road_type == 'Merge':
        crash_flag = run_merge(actors, road_network, ori_point, scenario,bng,env_info,case_result_folder)
        return crash_flag

    car_dict = get_car_info(actors)
    print('Original Vehicle Config:')
    print(car_dict)
    print('Vehicle Config after adjusting coordinates:')
    car_dict = adjust_coordinates(car_dict, ori_point)
    print(car_dict)

    if car_dict['V1'][1] == 'Car':
        ego_model = 'etk800'
    else:
        ego_model = 'citybus'

    # Ego car
    ego = Vehicle('ego', model=ego_model, color='Green')
    ego_start_point = car_dict['V1'][0][0]
    pos = (ego_start_point[0], ego_start_point[1], 100.5)
    rot = calculate_unit_quaternion(car_dict['V1'][0])
    ego.sensors.attach("damage", Damage())
    scenario.add_vehicle(ego, pos=pos, rot_quat=rot)

    # NPC cars
    # V2
    npc_model = 'etk800' if car_dict['V2'][1] == 'Car' else 'citybus'
    npc = Vehicle('npc', model=npc_model)
    damage_npc = Damage()
    npc.attach_sensor('damage_npc', damage_npc)
    npc_start_point = car_dict['V2'][0][0]
    pos = (npc_start_point[0], npc_start_point[1], 100.5)
    rot = calculate_unit_quaternion(car_dict['V2'][0])
    scenario.add_vehicle(npc, pos=pos, rot_quat=rot)

    # V3 if we have
    if len(car_dict) == 3:
        npc_model = 'etk800' if car_dict['V3'][1] == 'Car' else 'citybus'
        npc_2 = Vehicle('npc_2', model=npc_model)
        damage_npc_2 = Damage()
        npc_2.attach_sensor('damage_npc_2', damage_npc_2)
        npc_start_point = car_dict['V3'][0][0]
        pos = (npc_start_point[0], npc_start_point[1], 100.5)
        rot = calculate_unit_quaternion(car_dict['V3'][0])
        scenario.add_vehicle(npc_2, pos=pos, rot_quat=rot)

    # Generate scenario
    scenario.make(bng)
    bng.load_scenario(scenario)
    bng.start_scenario()

    day_time_mapping = {
        'Daytime': 10,
        'Not mentioned': 10,
        'Nighttime': 50
    }

    t = day_time_mapping.get(env_info.get('Time'), 10)
    bng.env.set_tod(t / 100)

    # Set monitor camera
    bng.camera.set_free(camera_view_point, (0, 0, -1))

    npc_script = convert_trajectory_to_npc_script(car_dict['V2'][0])
    npc.ai_set_script(npc_script)

    ego_script = convert_trajectory_to_npc_script(car_dict['V1'][0])
    ego.ai_set_script(ego_script)
    # V3 if we have
    if len(car_dict) == 3:
        npc_script_2 = convert_trajectory_to_npc_script(car_dict['V3'][0])
        npc_2.ai_set_script(npc_script_2)

    print('Simulation starts!')

    time.sleep(1)
    screenshot = pyautogui.screenshot(region=(screen_left, screen_top, screen_width, screen_height))
    save_path = os.path.join(case_result_folder, "1.png")
    screenshot.save(save_path)
    time.sleep(1)
    screenshot = pyautogui.screenshot(region=(screen_left, screen_top, screen_width, screen_height))
    save_path = os.path.join(case_result_folder, "2.png")
    screenshot.save(save_path)
    time.sleep(1)
    screenshot = pyautogui.screenshot(region=(screen_left, screen_top, screen_width, screen_height))
    save_path = os.path.join(case_result_folder, "3.png")
    screenshot.save(save_path)
    time.sleep(1)
    screenshot = pyautogui.screenshot(region=(screen_left, screen_top, screen_width, screen_height))
    save_path = os.path.join(case_result_folder, "4.png")
    screenshot.save(save_path)
    time.sleep(1)
    screenshot = pyautogui.screenshot(region=(screen_left, screen_top, screen_width, screen_height))
    save_path = os.path.join(case_result_folder, "5.png")
    screenshot.save(save_path)
    time.sleep(1)
    screenshot = pyautogui.screenshot(region=(screen_left, screen_top, screen_width, screen_height))
    save_path = os.path.join(case_result_folder, "6.png")
    screenshot.save(save_path)
    time.sleep(1)
    screenshot = pyautogui.screenshot(region=(screen_left, screen_top, screen_width, screen_height))
    save_path = os.path.join(case_result_folder, "7.png")
    screenshot.save(save_path)
    time.sleep(5)
    screenshot = pyautogui.screenshot(region=(screen_left, screen_top, screen_width, screen_height))
    save_path = os.path.join(case_result_folder, "8.png")
    screenshot.save(save_path)

    print('Simulation ends!')
    # ego.poll_sensors('damage')
    ego.sensors.poll()
    damage_data = ego.sensors["damage"].data
    crash_flag = check_crash(damage_data)
    if crash_flag:
        print('Crash detacted!')
    else:
        print('Crash Not Found!')
    print('=====================')
    return crash_flag

def get_dsl(dsls, id):
    for dsl in dsls:
        if dsl['Scenario'] == id:
            return dsl

def get_vn_traj_s(Initial_position,Actions):
    if Initial_position in ['W2E', 'N2S']:
        if Actions in ['Move forward', 'Move Forward', 'move forward', 'Stop']:
            num = str(random.randint(1, 30))
            num_1 = str(random.randint(6, 7))
            traj_list = f'[({num}, {num_1}), (50, 6), (90, 6)]'
            return traj_list
        elif Actions in ['Enter the Wrong Way', 'Enter The Wrong Way', 'enter the wrong way']:
            num = str(random.randint(2, 15))
            num_1 = str(random.randint(40, 60))
            traj_list = f'[({num}, 6), ({num_1}, 10), (90, 11)]'
            return traj_list
        elif Actions in ['Changing Lane', 'changing lane', 'Changing lane']:
            num = str(random.randint(2, 15))
            num_1 = str(random.randint(40, 60))
            traj_list = f'[({num}, 2), ({num_1}, 6), (90, 7)]'
            return traj_list
    else:
        if Actions in ['Move forward', 'Move Forward', 'move forward', 'Stop']:
            num = str(random.randint(85, 92))
            traj_list = f'[({num}, 10), (70, 10), (20, 10)]'
            return traj_list
        elif Actions in ['Enter the Wrong Way', 'Enter The Wrong Way', 'enter the wrong way']:
            num = str(random.randint(85, 92))
            num_1 = str(random.randint(40, 60))
            traj_list = f'[({num}, 10), ({num_1}, 6), (20, 5)]'
            return traj_list
        elif Actions in ['Changing Lane', 'changing lane', 'Changing lane']:
            num = str(random.randint(85, 92))
            num_1 = str(random.randint(40, 60))
            traj_list = f'[({num}, 14), ({num_1}, 10), (20, 9)]'
            return traj_list

def get_vn_traj_x(Initial_position,Actions):
    if Initial_position in ['W2E', 'w2e']:
        if Actions in ['Move forward', 'Move Forward', 'move forward']:
            traj_list = '[(-30, -2), (0, -2), (30, -2)]'
            return traj_list
        elif Actions in ['turn left', 'Turn Left', 'Turn left']:
            num = str(random.randint(-35, -25))
            num_1 = str(random.randint(5, 10))
            traj_list = f'[({num}, -2), (0, 0), (2, {num_1})]'
            return traj_list
        elif Actions in ['turn right', 'Turn Right', 'Turn right']:
            num = str(random.randint(-35, -25))
            num_1 = str(random.randint(-10, -5))
            traj_list = f'[({num}, -2), (-1, -1), (-2, {num_1})]'
            return traj_list
    elif Initial_position in ['E2W', 'e2w']:
        if Actions in ['Move forward', 'Move Forward', 'move forward']:
            traj_list = '[(30, 2), (0, 2), (-30, 2)]'
            return traj_list
        elif Actions in ['turn left', 'Turn Left', 'Turn left']:
            num = str(random.randint(25, 35))
            num_1 = str(random.randint(-10, -5))
            traj_list = f'[({num}, 2), (0, 0), (-2, {num_1})]'
            return traj_list
        elif Actions in ['turn right', 'Turn Right', 'Turn right']:
            num = str(random.randint(25, 35))
            num_1 = str(random.randint(5, 10))
            traj_list = f'[({num}, 2), (1, 1), (2, {num_1})]'
            return traj_list
    elif Initial_position in ['S2N', 's2n']:
        if Actions in ['Move forward', 'Move Forward', 'move forward']:
            traj_list = '[(2, -30), (2, 0), (2, 30)]'
            return traj_list
        elif Actions in ['turn left', 'Turn Left', 'Turn left']:
            num = str(random.randint(-35, -25))
            num_1 = str(random.randint(-10, -5))
            traj_list = f'[(2, {num}), (0, 0), ({num_1}, 2)]'
            return traj_list
        elif Actions in ['turn right', 'Turn Right', 'Turn right']:
            num = str(random.randint(-35, -25))
            num_1 = str(random.randint(5, 10))
            traj_list = f'[(2, {num}), (1, -1), ({num_1}, -2)]'
            return traj_list
    elif Initial_position in ['N2S', 'n2s']:
        if Actions in ['Move forward', 'Move Forward', 'move forward']:
            traj_list = '[(-2, 30), (-2, 0), (-2, -30)]'
            return traj_list
        elif Actions in ['turn left', 'Turn Left', 'Turn left']:
            num = str(random.randint(25, 30))
            num_1 = str(random.randint(5, 10))
            traj_list = f'[(-2, {num}), (0, 0), ({num_1}, -2)]'
            return traj_list
        elif Actions in ['turn right', 'Turn Right', 'Turn right']:
            num = str(random.randint(25, 30))
            num_1 = str(random.randint(-10, -5))
            traj_list = f'[(-2, {num}), (-1, 1), ({num_1}, 2)]'
            return traj_list

def get_vn_traj_t(Initial_position,Actions,stem):
    if stem == 'South':
        if Initial_position in ['W2E', 'w2e']:
            if Actions in ['Move forward', 'Move Forward', 'move forward']:
                traj_list = '[(-25, -2), (0, -2), (25, -2)]'
                return traj_list
        elif Initial_position in ['E2W', 'e2w']:
            if Actions in ['Move forward', 'Move Forward', 'move forward']:
                traj_list = '[(25, 2), (0, 2), (-25, 2)]'
                return traj_list
        elif Initial_position in ['S2N', 's2n']:
            if Actions in ['turn left', 'Turn Left', 'Turn left']:
                traj_list = '[(2, -25), (0, 0), (-10, 2)]'
                return traj_list
            if Actions in ['turn right', 'Turn Right', 'Turn right']:
                traj_list = '[(2, -25), (1, -1), (10, -2)]'
                return traj_list
    elif stem == 'West':
        if Initial_position in ['N2S', 'n2s']:
            if Actions in ['Move forward', 'Move Forward', 'move forward']:
                traj_list = '[(-25, -2), (0, -2), (25, -2)]'
                return traj_list
        elif Initial_position in ['S2N', 's2n']:
            if Actions in ['Move forward', 'Move Forward', 'move forward']:
                traj_list = '[(25, 2), (0, 2), (-25, 2)]'
                return traj_list
        elif Initial_position in ['W2E', 'w2e']:
            if Actions in ['turn left', 'Turn Left', 'Turn left']:
                traj_list = '[(2, -25), (0, 0), (-10, 2)]'
                return traj_list
            if Actions in ['turn right', 'Turn Right', 'Turn right']:
                traj_list = '[(2, -25), (1, -1), (10, -2)]'
                return traj_list
    elif stem == 'East':
        if Initial_position in ['S2N', 's2n']:
            if Actions in ['Move forward', 'Move Forward', 'move forward']:
                traj_list = '[(-25, -2), (0, -2), (25, -2)]'
                return traj_list
        elif Initial_position in ['N2S', 'n2s']:
            if Actions in ['Move forward', 'Move Forward', 'move forward']:
                traj_list = '[(25, 2), (0, 2), (-25, 2)]'
                return traj_list
        elif Initial_position in ['E2W', 'e2w']:
            if Actions in ['turn left', 'Turn Left', 'Turn left']:
                traj_list = '[(2, -25), (0, 0), (-10, 2)]'
                return traj_list
            if Actions in ['turn right', 'Turn Right', 'Turn right']:
                traj_list = '[(2, -25), (1, -1), (10, -2)]'
                return traj_list

def get_vn_traj_r(Initial_position,Actions):
    if Initial_position in ['Main road', 'Main Road', 'main road']:
        if Actions in ['Move forward', 'Move Forward', 'move forward']:
            traj_list = '[(10, 2), (30, 2), (50, 2)]'
            return traj_list
    elif Initial_position in ['On-ramp', 'On-Ramp', 'on-ramp', 'on ramp']:
        if Actions in ['Merge', 'merge']:
            traj_list = '[(15, -10), (30, 0), (50, 2)]'
            return traj_list

def get_vn_traj_c(Initial_position,Actions):
    if Initial_position in ['W2E', 'N2S']:
        if Actions in ['Move forward', 'Move Forward', 'move forward', 'Stop']:
            traj_list = '[(2, 6), (20, 15), (40, 13)]'
            return traj_list
        elif Actions in ['Enter the Wrong Way', 'Enter The Wrong Way', 'enter the wrong way']:
            traj_list = '[(2, 6), (20, 15), (40, 20)]'
            return traj_list
    else:
        if Actions in ['Move forward', 'Move Forward', 'move forward', 'Stop']:
            traj_list = '[(55, 7), (35, 17), (15, 10)]'
            return traj_list
        elif Actions in ['Enter the Wrong Way', 'Enter The Wrong Way', 'enter the wrong way']:
            traj_list = '[(55, 7), (35, 17), (15, 22)]'
            return traj_list

def get_vn_type(model):
    if model in ['Sedan','SUV','Minivan', 'Pickup']:
        return 'Car'
    else:
        return 'Truck'

def convert_dsl(DSLs, meta_msg):
    dsl_yaml = {}
    for case in meta_msg:
        scenario_id = case[-1] # '128697'
        dsl = get_dsl(DSLs,scenario_id)
        Actors = dsl['Actors'][0]
        Road_network = dsl['Road network']
        Env = dsl['Env']
        road_type = case[0]

        case_yaml = {}
        case_yaml['Scenario'] = scenario_id
        Env['Validation'] = 'Passed'
        case_yaml['Env'] = Env

        if road_type == 'Straight':
            case_yaml['Road type'] = 'Straight'
            case_yaml['Road network'] = {'Length': 100, 'No_lanes': 4, 'No_ways': 2, 'Validation': 'Passed', 'Width': 4}
            actor_dict = {}
            v_counter = 1
            for ID, vehicle in Actors.items():
                # Vn_traj
                actor_dict[f'V{v_counter}_traj'] = get_vn_traj_s(vehicle['Initial_position'],vehicle['Actions'])
                # Vn_type
                actor_dict[f'V{v_counter}_type'] = get_vn_type(vehicle['Model'])
                v_counter += 1
            actor_dict['Validation'] = 'Passed'
            case_yaml['Actors'] = actor_dict

        elif road_type == 'Curve':
            case_yaml['Road type'] = 'Curve'
            case_yaml['Road network'] = {'Length': 60, 'No_lanes': 2, 'No_ways': 2, 'Validation': 'Passed', 'Width': 4}
            actor_dict = {}
            v_counter = 1
            for ID, vehicle in Actors.items():
                # ID = 'Vehicle_1'
                # vehicle = {'Model': 'Sedan', 'Initial_position': 'W2E', 'Actions': 'Move forward', 'Speed': 'N/A'}
                # Vn_traj
                actor_dict[f'V{v_counter}_traj'] = get_vn_traj_c(vehicle['Initial_position'], vehicle['Actions'])
                # Vn_type
                actor_dict[f'V{v_counter}_type'] = get_vn_type(vehicle['Model'])
                v_counter += 1
            actor_dict['Validation'] = 'Passed'
            case_yaml['Actors'] = actor_dict

        elif road_type == 'Intersection':
            case_yaml['Road type'] = 'Intersection'
            case_yaml['Road network'] = {'Length': 80, 'No_lanes': 2, 'No_ways': 2, 'Validation': 'Passed', 'Width': 4}
            actor_dict = {}
            v_counter = 1
            for ID, vehicle in Actors.items():
                # ID = 'Vehicle_1'
                # vehicle = {'Model': 'SUV', 'Initial_position': 'E2W', 'Actions': 'Turn left', 'Speed': '55'}
                # Vn_traj
                actor_dict[f'V{v_counter}_traj'] = get_vn_traj_x(vehicle['Initial_position'],vehicle['Actions'])
                # Vn_type
                actor_dict[f'V{v_counter}_type'] = get_vn_type(vehicle['Model'])
                v_counter += 1
            actor_dict['Validation'] = 'Passed'
            case_yaml['Actors'] = actor_dict

        elif road_type == 'T-intersection':
            case_yaml['Road type'] = 'T-intersection'
            case_yaml['Road network'] = {'Length_branch': 35, 'Length_main': 60, 'No_lanes_branch_road': 2,
                                        'No_lanes_main_road': 2, 'No_ways_branch_road': 2, 'No_ways_main_road': 2,
                                        'Validation': 'Passed', 'Width': 4}
            actor_dict = {}
            v_counter = 1
            for ID, vehicle in Actors.items():
                # ID = 'Vehicle_1'
                # vehicle = {'Model': 'SUV', 'Initial_position': 'E2W', 'Actions': 'Turn left', 'Speed': '55'}
                # Vn_traj
                actor_dict[f'V{v_counter}_traj'] = get_vn_traj_t(vehicle['Initial_position'], vehicle['Actions'], Road_network['Stem road direction'])
                # Vn_type
                actor_dict[f'V{v_counter}_type'] = get_vn_type(vehicle['Model'])
                v_counter += 1
            actor_dict['Validation'] = 'Passed'
            case_yaml['Actors'] = actor_dict

        elif road_type == 'Merging':
            case_yaml['Road type'] = 'Merge'
            case_yaml['Road network'] = {'Length_main': 60, 'No_lanes_branch_road': 2, 'No_lanes_main_road': 4,
                                        'No_ways_main_road': 2, 'Validation': 'Passed', 'Width': 4}
            actor_dict = {}
            v_counter = 1
            for ID, vehicle in Actors.items():
                # ID = 'Vehicle_1'
                # vehicle = {'Model': 'SUV', 'Initial_position': 'E2W', 'Actions': 'Turn left', 'Speed': '55'}
                # Vn_traj
                actor_dict[f'V{v_counter}_traj'] = get_vn_traj_r(vehicle['Initial_position'], vehicle['Actions'])
                # Vn_type
                actor_dict[f'V{v_counter}_type'] = get_vn_type(vehicle['Model'])
                v_counter += 1
            actor_dict['Validation'] = 'Passed'
            case_yaml['Actors'] = actor_dict

        dsl_yaml[scenario_id] = case_yaml

    return dsl_yaml

def main():
    parser = argparse.ArgumentParser(description='Scenario Reconstruction in BeamNG')
    parser.add_argument('--dsl_path',
                        default=r'SAFE\Experiment_results\DSL_results_2025-02-28_12-15-25\DSL_extraction_results.pkl')
    parser.add_argument('--meta_msg',
                        default=r'SAFE\Experiment_results\Meta_Message_results_2025-02-23_11-20-25\meta_data_results.pkl')
    args = parser.parse_args()

    with open(args.dsl_path,'rb') as file:
        # [{DSL1},{DSL2},{DSL3},...{DSLN}]
        DSLs = pickle.load(file)

    with open(args.meta_msg,'rb') as file:
        meta_msg = pickle.load(file)

    # Create result folder
    current_path = Path.cwd()
    parent_path = current_path.parent.parent # SAFE
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_folder = parent_path / 'Experiment_results' / f"BeamNG_AutoADS_results_{current_time}"
    os.makedirs(result_folder, exist_ok=True)

    folder_path = os.path.abspath(result_folder)

    # Convert scenario representation to yaml file
    # yaml_data = load_yaml_files_to_dict(args.dsl_path)
    yaml_data = convert_dsl(DSLs, meta_msg)
    fail_list = []
    num_cases = len(yaml_data)
    ss_count = 0
    crash_count = 0

    # BeamNG Settings
    # Specify where BeamNG home and user are
    BNG_HOME = r"remember to set the beamng home"
    BNG_USER = r"remember to set the beamng user"
    beamng = BeamNGpy('localhost', 64256, user=BNG_USER, home=BNG_HOME)
    bng = beamng.open(launch=True)
    bng.set_deterministic()

    start_time = time.time()

    # loop over all the cases
    for ID, DSL in yaml_data.items():
        actors = DSL['Actors']
        env_info = DSL['Env']
        road_network = DSL['Road network']
        road_type = DSL['Road type']

        for i in [1, 2]:
            if i == 1:
                # construct it at the first time
                try:
                    crash_sta = run_scenario(bng, road_network, env_info, actors, road_type, ID, folder_path, i)
                    ss_count += 1
                    crash_count += crash_sta
                except Exception as e:
                    print(f"Scenario {ID} build failed due to error: {e}")
                    fail_list.append(ID)
                # end -> swap vehicles
            elif i == 2:
                try:
                    # swap vehicles
                    V1_type = actors['V1_type']
                    V1_traj = actors['V1_traj']
                    V2_type = actors['V2_type']
                    V2_traj = actors['V2_traj']
                    actors['V1_type'] = V2_type
                    actors['V1_traj'] = V2_traj
                    actors['V2_type'] = V1_type
                    actors['V2_traj'] = V1_traj
                    crash_sta = run_scenario(bng, road_network, env_info, actors, road_type, ID, folder_path, i)
                    ss_count += 1
                    crash_count += crash_sta
                except Exception as e:
                    print(f"Scenario {ID} build failed due to error: {e}")
                    fail_list.append(ID)

    end_time = time.time()
    total_time = end_time - start_time

    print('============= Simulation Results =============')
    print('Total Number of Cases: ', num_cases)
    print(f'Successfully built {ss_count} examples!')
    print(f'{crash_count} collision cases have been found!')
    print(f"Total program running time:{total_time:.2f} s")
    print('==============================================')

if __name__=='__main__':
    main()