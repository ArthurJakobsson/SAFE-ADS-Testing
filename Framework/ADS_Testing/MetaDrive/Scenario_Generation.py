from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.vehicle.vehicle_type import SVehicle, DefaultVehicle, MVehicle, LVehicle, XLVehicle
from metadrive.utils.doc_utils import generate_gif
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.policy.expert_policy import ExpertPolicy
from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation
from metadrive.engine.logger import set_log_level
from pathlib import Path
import logging
import argparse
import pickle
import copy
from datetime import datetime
import time
import os

def action_chain_s(Vehicle, Num_lanes):
    Vehicle_1 = []  # spawn_lane_index_0 & spawn_lane_index_1 & destination & spawn_lane_index_2
    # check initial position
    if Vehicle['Initial_position'] in ['W2E', 'N2S']:
        # coming from left to right
        # check actions
        if Vehicle['Actions'] in ['Move forward', 'Move Forward', 'move forward', 'Stop']:
            # coming from left to right and go straight
            # spawn_lane_index_0 & spawn_lane_index_1 & destination can be:
            # Vehicle_1.append(['>', '>>', '>>>'])
            Vehicle_1.append(['>>', '>>>', '1S0_0_'])
            # Vehicle_1 = [['>','>>','>>>'], ['>>', '>>>', '1S0_0_']]
            # check the number of lanes
            chain = []
            for lane_id in range(Num_lanes):
                spawn_lane_index_2 = lane_id  # 0, 1, 2, ...
                for case_list in Vehicle_1:
                    chain.append(case_list + [spawn_lane_index_2])
            return chain
        elif Vehicle['Actions'] in ['Enter the Wrong Way', 'Enter The Wrong Way', 'enter the wrong way']:
            # coming from left to right and enter the wrong way
            # spawn_lane_index_0 & spawn_lane_index_1 & destination can be:
            # Vehicle_1.append(['>', '>>', '->>'])
            # Vehicle_1.append(['>', '>>', '->>>'])
            # Vehicle_1.append(['>', '>>', '-1S0_0_'])
            # Vehicle_1.append(['>>', '>>>', '->>>'])
            Vehicle_1.append(['>>', '>>>', '-1S0_0_'])
            # Vehicle_1.append(['>>>', '1S0_0_', '-1S0_0_'])
            # check the number of lanes
            chain = []
            for lane_id in range(Num_lanes):
                spawn_lane_index_2 = lane_id  # 0, 1, 2, ...
                for case_list in Vehicle_1:
                    chain.append(case_list + [spawn_lane_index_2])
            return chain
        elif Vehicle['Actions'] in ['Changing Lane', 'changing lane', 'Changing lane']:
            if Num_lanes > 1:
                # Vehicle_1.append(['>', '>>', '->>', 1])
                # Vehicle_1.append(['>', '>>', '->>>', 1])
                # Vehicle_1.append(['>', '>>', '-1S0_0_', 1])
                Vehicle_1.append(['>>', '>>>', '->>>', 1])
                Vehicle_1.append(['>>', '>>>', '-1S0_0_', 1])
                # Vehicle_1.append(['>>>', '1S0_0_', '-1S0_0_', 1])
            else:
                # Vehicle_1.append(['>', '>>', '>>>', 0])
                Vehicle_1.append(['>>', '>>>', '1S0_0_', 0])
            return Vehicle_1
    else:
        # coming form right to left
        # check actions
        if Vehicle['Actions'] in ['Move forward', 'Move Forward', 'move forward', 'Stop']:
            # coming from right to left and go straight
            # spawn_lane_index_0 & spawn_lane_index_1 & destination can be:
            Vehicle_1.append(['-1S0_0_', '->>>', '->>'])
            # Vehicle_1.append(['->>>', '->>', '->'])
            # check the number of lanes
            chain = []
            for lane_id in range(Num_lanes):
                spawn_lane_index_2 = lane_id  # 0, 1, 2, ...
                for case_list in Vehicle_1:
                    chain.append(case_list + [spawn_lane_index_2])
            return chain
        elif Vehicle['Actions'] in ['Enter the Wrong Way', 'Enter The Wrong Way', 'enter the wrong way']:
            # coming from right to left and enter the wrong way
            # spawn_lane_index_0 & spawn_lane_index_1 & destination can be:
            # Vehicle_1.append(['-1S0_0_', '->>>', '>>>'])
            Vehicle_1.append(['-1S0_0_', '->>>', '>>'])
            # Vehicle_1.append(['-1S0_0_', '->>>', '>'])
            # Vehicle_1.append(['->>>', '->>', '>>'])
            # Vehicle_1.append(['->>>', '->>', '>'])
            # Vehicle_1.append(['->>', '->', '>'])
            # check the number of lanes
            chain = []
            for lane_id in range(Num_lanes):
                spawn_lane_index_2 = lane_id  # 0, 1, 2, ...
                for case_list in Vehicle_1:
                    chain.append(case_list + [spawn_lane_index_2])
            return chain
        elif Vehicle['Actions'] in ['Changing Lane', 'changing lane', 'Changing lane']:
            if Num_lanes > 1:
                # Vehicle_1.append(['-1S0_0_', '->>>', '>>>', 1])
                Vehicle_1.append(['-1S0_0_', '->>>', '>>', 1])
                Vehicle_1.append(['-1S0_0_', '->>>', '>', 1])
                # Vehicle_1.append(['->>>', '->>', '>>', 1])
                # Vehicle_1.append(['->>>', '->>', '>', 1])
                # Vehicle_1.append(['->>', '->', '>', 1])
            else:
                Vehicle_1.append(['-1S0_0_', '->>>', '->>', 0])
                # Vehicle_1.append(['->>>', '->>', '->', 0])
            return Vehicle_1

def action_chain_c(Vehicle, Num_lanes):
    Vehicle_1 = []  # spawn_lane_index_0 & spawn_lane_index_1 & destination & spawn_lane_index_2
    # check initial position
    if Vehicle['Initial_position'] in ['W2E', 'N2S']:
        # coming from left to right
        # check actions
        if Vehicle['Actions'] in ['Move forward', 'Move Forward', 'move forward', 'Stop']:
            # coming from left to right and go straight
            # spawn_lane_index_0 & spawn_lane_index_1 & destination can be:
            # Vehicle_1.append(['>>', '>>>', '1C0_0_'])
            Vehicle_1.append(['>>>', '1C0_0_', '1C0_1_'])
            # check the number of lanes
            chain = []
            for lane_id in range(Num_lanes):
                spawn_lane_index_2 = lane_id  # 0, 1, 2, ...
                for case_list in Vehicle_1:
                    chain.append(case_list + [spawn_lane_index_2])
            return chain
        elif Vehicle['Actions'] in ['Enter the Wrong Way', 'Enter The Wrong Way', 'enter the wrong way']:
            # coming from left to right and enter the wrong way
            # spawn_lane_index_0 & spawn_lane_index_1 & destination can be:
            Vehicle_1.append(['>>', '>>>', '-1C0_0_'])
            Vehicle_1.append(['>>>', '1C0_0_', '-1C0_0_'])
            Vehicle_1.append(['>>>', '1C0_0_', '-1C0_1_'])
            # Vehicle_1.append(['1C0_0_', '1C0_1_', '-1C0_1_'])
            # check the number of lanes
            chain = []
            for lane_id in range(Num_lanes):
                spawn_lane_index_2 = lane_id  # 0, 1, 2, ...
                for case_list in Vehicle_1:
                    chain.append(case_list + [spawn_lane_index_2])
            return chain
        elif Vehicle['Actions'] in ['Changing Lane', 'changing lane', 'Changing lane']:
            if Num_lanes > 1:
                Vehicle_1.append(['>>', '>>>', '-1C0_0_', 1])
                Vehicle_1.append(['>>>', '1C0_0_', '-1C0_0_', 1])
                Vehicle_1.append(['>>>', '1C0_0_', '-1C0_1_', 1])
                # Vehicle_1.append(['1C0_0_', '1C0_1_', '-1C0_1_', 1])
            else:
                Vehicle_1.append(['>>', '>>>', '1C0_0_', 0])
                Vehicle_1.append(['>>>', '1C0_0_', '1C0_1_', 0])
            return Vehicle_1
    else:
        # coming form right to left
        # check actions
        if Vehicle['Actions'] in ['Move forward', 'Move Forward', 'move forward', 'Stop']:
            # coming from right to left and go straight
            # spawn_lane_index_0 & spawn_lane_index_1 & destination can be:
            Vehicle_1.append(['-1C0_1_', '-1C0_0_', '->>>'])
            Vehicle_1.append(['-1C0_0_', '->>>', '->>'])
            # check the number of lanes
            chain = []
            for lane_id in range(Num_lanes):
                spawn_lane_index_2 = lane_id  # 0, 1, 2, ...
                for case_list in Vehicle_1:
                    chain.append(case_list + [spawn_lane_index_2])
            return chain
        elif Vehicle['Actions'] in ['Enter the Wrong Way', 'Enter The Wrong Way', 'enter the wrong way']:
            # coming from right to left and enter the wrong way
            # spawn_lane_index_0 & spawn_lane_index_1 & destination can be:
            Vehicle_1.append(['-1C0_1_', '-1C0_0_', '1C0_0_'])
            Vehicle_1.append(['-1C0_1_', '-1C0_0_', '>>>'])
            Vehicle_1.append(['-1C0_0_', '->>>', '>>>'])
            Vehicle_1.append(['-1C0_0_', '->>>', '>>'])
            # check the number of lanes
            chain = []
            for lane_id in range(Num_lanes):
                spawn_lane_index_2 = lane_id  # 0, 1, 2, ...
                for case_list in Vehicle_1:
                    chain.append(case_list + [spawn_lane_index_2])
            return chain
        elif Vehicle['Actions'] in ['Changing Lane', 'changing lane', 'Changing lane']:
            if Num_lanes > 1:
                Vehicle_1.append(['-1C0_1_', '-1C0_0_', '1C0_0_', 1])
                Vehicle_1.append(['-1C0_1_', '-1C0_0_', '>>>', 1])
                Vehicle_1.append(['-1C0_0_', '->>>', '>>>', 1])
                Vehicle_1.append(['-1C0_0_', '->>>', '>>', 1])
            else:
                Vehicle_1.append(['-1C0_1_', '-1C0_0_', '->>>', 0])
                Vehicle_1.append(['-1C0_0_', '->>>', '->>', 0])
            return Vehicle_1

def action_chain_x(Vehicle, Num_lanes):
    Vehicle_1 = []  # spawn_lane_index_0 & spawn_lane_index_1 & destination & spawn_lane_index_2
    # check initial position
    if Vehicle['Initial_position'] in ['W2E', 'w2e']:
        # coming from left to right
        # check actions
        if Vehicle['Actions'] in ['Move forward', 'Move Forward', 'move forward']:
            # coming from left to right and go straight
            # spawn_lane_index_0 & spawn_lane_index_1 & destination can be:
            Vehicle_1.append(['>>', '>>>', '1X1_0_'])
            # check the number of lanes
            chain = []
            for lane_id in range(Num_lanes):
                spawn_lane_index_2 = lane_id  # 0, 1, 2, ...
                for case_list in Vehicle_1:
                    chain.append(case_list + [spawn_lane_index_2])
            return chain
        elif Vehicle['Actions'] in ['turn left', 'Turn Left', 'Turn left']:
            # coming from left to right and turn left
            # spawn_lane_index_0 & spawn_lane_index_1 & destination can be:
            Vehicle_1.append(['>>', '>>>', '1X2_0_'])
            # check the number of lanes
            chain = []
            for lane_id in range(Num_lanes):
                spawn_lane_index_2 = lane_id  # 0, 1, 2, ...
                for case_list in Vehicle_1:
                    chain.append(case_list + [spawn_lane_index_2])
            return chain
        elif Vehicle['Actions'] in ['turn right', 'Turn Right', 'Turn right']:
            # coming from left to right and turn right
            # spawn_lane_index_0 & spawn_lane_index_1 & destination can be:
            Vehicle_1.append(['>>', '>>>', '1X0_0_'])
            # check the number of lanes
            chain = []
            for lane_id in range(Num_lanes):
                spawn_lane_index_2 = lane_id  # 0, 1, 2, ...
                for case_list in Vehicle_1:
                    chain.append(case_list + [spawn_lane_index_2])
            return chain
    elif Vehicle['Initial_position'] in ['E2W', 'e2w']:
        # coming form right to left
        # check actions
        if Vehicle['Actions'] in ['Move forward', 'Move Forward', 'move forward']:
            # coming from right to left and go straight
            # spawn_lane_index_0 & spawn_lane_index_1 & destination can be:
            Vehicle_1.append(['-1X1_1_', '-1X1_0_', '->>>'])
            # check the number of lanes
            chain = []
            for lane_id in range(Num_lanes):
                spawn_lane_index_2 = lane_id  # 0, 1, 2, ...
                for case_list in Vehicle_1:
                    chain.append(case_list + [spawn_lane_index_2])
            return chain
        elif Vehicle['Actions'] in ['turn left', 'Turn Left', 'Turn left']:
            # coming from right to left turn left
            # spawn_lane_index_0 & spawn_lane_index_1 & destination can be:
            Vehicle_1.append(['-1X1_1_', '-1X1_0_', '1X0_0_'])
            # check the number of lanes
            chain = []
            for lane_id in range(Num_lanes):
                spawn_lane_index_2 = lane_id  # 0, 1, 2, ...
                for case_list in Vehicle_1:
                    chain.append(case_list + [spawn_lane_index_2])
            return chain
        elif Vehicle['Actions'] in ['turn right', 'Turn Right', 'Turn right']:
            # coming from right to left turn right
            # spawn_lane_index_0 & spawn_lane_index_1 & destination can be:
            Vehicle_1.append(['-1X1_1_', '-1X1_0_', '1X2_0_'])
            # check the number of lanes
            chain = []
            for lane_id in range(Num_lanes):
                spawn_lane_index_2 = lane_id  # 0, 1, 2, ...
                for case_list in Vehicle_1:
                    chain.append(case_list + [spawn_lane_index_2])
            return chain
    elif Vehicle['Initial_position'] in ['N2S', 'n2s']:
        # coming form top to end
        # check actions
        if Vehicle['Actions'] in ['Move forward', 'Move Forward', 'move forward']:
            # coming from right to left and go straight
            # spawn_lane_index_0 & spawn_lane_index_1 & destination can be:
            Vehicle_1.append(['-1X2_1_', '-1X2_0_', '1X0_0_'])
            # check the number of lanes
            chain = []
            for lane_id in range(Num_lanes):
                spawn_lane_index_2 = lane_id  # 0, 1, 2, ...
                for case_list in Vehicle_1:
                    chain.append(case_list + [spawn_lane_index_2])
            return chain
        elif Vehicle['Actions'] in ['turn left', 'Turn Left', 'Turn left']:
            # coming from right to left turn left
            # spawn_lane_index_0 & spawn_lane_index_1 & destination can be:
            Vehicle_1.append(['-1X2_1_', '-1X2_0_', '1X1_0_'])
            # check the number of lanes
            chain = []
            for lane_id in range(Num_lanes):
                spawn_lane_index_2 = lane_id  # 0, 1, 2, ...
                for case_list in Vehicle_1:
                    chain.append(case_list + [spawn_lane_index_2])
            return chain
        elif Vehicle['Actions'] in ['turn right', 'Turn Right', 'Turn right']:
            # coming from right to left turn right
            # spawn_lane_index_0 & spawn_lane_index_1 & destination can be:
            Vehicle_1.append(['-1X2_1_', '-1X2_0_', '->>>'])
            # check the number of lanes
            chain = []
            for lane_id in range(Num_lanes):
                spawn_lane_index_2 = lane_id  # 0, 1, 2, ...
                for case_list in Vehicle_1:
                    chain.append(case_list + [spawn_lane_index_2])
            return chain
    elif Vehicle['Initial_position'] in ['S2N', 's2n']:
        # coming form end to top
        # check actions
        if Vehicle['Actions'] in ['Move forward', 'Move Forward', 'move forward']:
            # coming form end to top and go straight
            # spawn_lane_index_0 & spawn_lane_index_1 & destination can be:
            Vehicle_1.append(['-1X0_1_', '-1X0_0_', '1X2_0_'])
            # check the number of lanes
            chain = []
            for lane_id in range(Num_lanes):
                spawn_lane_index_2 = lane_id  # 0, 1, 2, ...
                for case_list in Vehicle_1:
                    chain.append(case_list + [spawn_lane_index_2])
            return chain
        elif Vehicle['Actions'] in ['turn left', 'Turn Left', 'Turn left']:
            # coming form end to top turn left
            # spawn_lane_index_0 & spawn_lane_index_1 & destination can be:
            Vehicle_1.append(['-1X0_1_', '-1X0_0_', '->>>'])
            # check the number of lanes
            chain = []
            for lane_id in range(Num_lanes):
                spawn_lane_index_2 = lane_id  # 0, 1, 2, ...
                for case_list in Vehicle_1:
                    chain.append(case_list + [spawn_lane_index_2])
            return chain
        elif Vehicle['Actions'] in ['turn right', 'Turn Right', 'Turn right']:
            # coming form end to top turn right
            # spawn_lane_index_0 & spawn_lane_index_1 & destination can be:
            Vehicle_1.append(['-1X0_1_', '-1X0_0_', '1X1_0_'])
            # check the number of lanes
            chain = []
            for lane_id in range(Num_lanes):
                spawn_lane_index_2 = lane_id  # 0, 1, 2, ...
                for case_list in Vehicle_1:
                    chain.append(case_list + [spawn_lane_index_2])
            return chain

def action_chain_t(Vehicle, Num_lanes, Stem):
    Vehicle_1 = []  # spawn_lane_index_0 & spawn_lane_index_1 & destination & spawn_lane_index_2
    # check Stem road direction
    if Stem == 'South':
        # check initial position
        if Vehicle['Initial_position'] in ['W2E', 'w2e']:
            # coming from left to right
            # check actions
            if Vehicle['Actions'] in ['Move forward', 'Move Forward', 'move forward']:
                # coming from left to right and go straight
                # spawn_lane_index_0 & spawn_lane_index_1 & destination can be:
                Vehicle_1.append(['>>', '>>>', '1T1_0_'])
                Vehicle_1.append(['>>>', '1T1_0_', '1T1_1_'])
                # check the number of lanes
                chain = []
                for lane_id in range(Num_lanes):
                    spawn_lane_index_2 = lane_id  # 0, 1, 2, ...
                    for case_list in Vehicle_1:
                        chain.append(case_list + [spawn_lane_index_2])
                return chain
        elif Vehicle['Initial_position'] in ['E2W', 'e2w']:
            # coming form right to left
            # check actions
            if Vehicle['Actions'] in ['Move forward', 'Move Forward', 'move forward']:
                # coming from right to left and go straight
                # spawn_lane_index_0 & spawn_lane_index_1 & destination can be:
                Vehicle_1.append(['-1T1_1_', '-1T1_0_', '->>>'])
                Vehicle_1.append(['-1T1_0_', '->>>', '->>'])
                # check the number of lanes
                chain = []
                for lane_id in range(Num_lanes):
                    spawn_lane_index_2 = lane_id  # 0, 1, 2, ...
                    for case_list in Vehicle_1:
                        chain.append(case_list + [spawn_lane_index_2])
                return chain
        elif Vehicle['Initial_position'] in ['S2N', 's2n']:
            if Vehicle['Actions'] in ['turn left', 'Turn Left', 'Turn left']:
                # coming form end to top turn left
                # spawn_lane_index_0 & spawn_lane_index_1 & destination can be:
                Vehicle_1.append(['-1T0_1_', '-1T0_0_', '->>>'])
                Vehicle_1.append(['-1T0_0_', '->>>', '->>'])
                # check the number of lanes
                chain = []
                for lane_id in range(Num_lanes):
                    spawn_lane_index_2 = lane_id  # 0, 1, 2, ...
                    for case_list in Vehicle_1:
                        chain.append(case_list + [spawn_lane_index_2])
                return chain
            elif Vehicle['Actions'] in ['turn right', 'Turn Right', 'Turn right']:
                # coming form end to top turn right
                # spawn_lane_index_0 & spawn_lane_index_1 & destination can be:
                Vehicle_1.append(['-1T0_1_', '-1T0_0_', '1T1_0_'])
                Vehicle_1.append(['-1T0_0_', '1T1_0_', '1T1_1_'])
                # check the number of lanes
                chain = []
                for lane_id in range(Num_lanes):
                    spawn_lane_index_2 = lane_id  # 0, 1, 2, ...
                    for case_list in Vehicle_1:
                        chain.append(case_list + [spawn_lane_index_2])
                return chain
    elif Stem == 'West':
        # check initial position
        if Vehicle['Initial_position'] in ['N2S', 'n2s']:
            # coming from left to right
            # check actions
            if Vehicle['Actions'] in ['Move forward', 'Move Forward', 'move forward']:
                # coming from left to right and go straight
                # spawn_lane_index_0 & spawn_lane_index_1 & destination can be:
                Vehicle_1.append(['>>', '>>>', '1T1_0_'])
                Vehicle_1.append(['>>>', '1T1_0_', '1T1_1_'])
                # check the number of lanes
                chain = []
                for lane_id in range(Num_lanes):
                    spawn_lane_index_2 = lane_id  # 0, 1, 2, ...
                    for case_list in Vehicle_1:
                        chain.append(case_list + [spawn_lane_index_2])
                return chain
        elif Vehicle['Initial_position'] in ['S2N', 's2n']:
            # coming form right to left
            # check actions
            if Vehicle['Actions'] in ['Move forward', 'Move Forward', 'move forward']:
                # coming from right to left and go straight
                # spawn_lane_index_0 & spawn_lane_index_1 & destination can be:
                Vehicle_1.append(['-1T1_1_', '-1T1_0_', '->>>'])
                Vehicle_1.append(['-1T1_0_', '->>>', '->>'])
                # check the number of lanes
                chain = []
                for lane_id in range(Num_lanes):
                    spawn_lane_index_2 = lane_id  # 0, 1, 2, ...
                    for case_list in Vehicle_1:
                        chain.append(case_list + [spawn_lane_index_2])
                return chain
        elif Vehicle['Initial_position'] in ['W2E', 'w2e']:
            if Vehicle['Actions'] in ['turn left', 'Turn Left', 'Turn left']:
                # coming form end to top turn left
                # spawn_lane_index_0 & spawn_lane_index_1 & destination can be:
                Vehicle_1.append(['-1T0_1_', '-1T0_0_', '->>>'])
                Vehicle_1.append(['-1T0_0_', '->>>', '->>'])
                # check the number of lanes
                chain = []
                for lane_id in range(Num_lanes):
                    spawn_lane_index_2 = lane_id  # 0, 1, 2, ...
                    for case_list in Vehicle_1:
                        chain.append(case_list + [spawn_lane_index_2])
                return chain
            elif Vehicle['Actions'] in ['turn right', 'Turn Right', 'Turn right']:
                # coming form end to top turn right
                # spawn_lane_index_0 & spawn_lane_index_1 & destination can be:
                Vehicle_1.append(['-1T0_1_', '-1T0_0_', '1T1_0_'])
                Vehicle_1.append(['-1T0_0_', '1T1_0_', '1T1_1_'])
                # check the number of lanes
                chain = []
                for lane_id in range(Num_lanes):
                    spawn_lane_index_2 = lane_id  # 0, 1, 2, ...
                    for case_list in Vehicle_1:
                        chain.append(case_list + [spawn_lane_index_2])
                return chain
    elif Stem == 'East':
        # check initial position
        if Vehicle['Initial_position'] in ['S2N', 's2n']:
            # coming from left to right
            # check actions
            if Vehicle['Actions'] in ['Move forward', 'Move Forward', 'move forward']:
                # coming from left to right and go straight
                # spawn_lane_index_0 & spawn_lane_index_1 & destination can be:
                Vehicle_1.append(['>>', '>>>', '1T1_0_'])
                Vehicle_1.append(['>>>', '1T1_0_', '1T1_1_'])
                # check the number of lanes
                chain = []
                for lane_id in range(Num_lanes):
                    spawn_lane_index_2 = lane_id  # 0, 1, 2, ...
                    for case_list in Vehicle_1:
                        chain.append(case_list + [spawn_lane_index_2])
                return chain
        elif Vehicle['Initial_position'] in ['N2S', 'n2s']:
            # coming form right to left
            # check actions
            if Vehicle['Actions'] in ['Move forward', 'Move Forward', 'move forward']:
                # coming from right to left and go straight
                # spawn_lane_index_0 & spawn_lane_index_1 & destination can be:
                Vehicle_1.append(['-1T1_1_', '-1T1_0_', '->>>'])
                Vehicle_1.append(['-1T1_0_', '->>>', '->>'])
                # check the number of lanes
                chain = []
                for lane_id in range(Num_lanes):
                    spawn_lane_index_2 = lane_id  # 0, 1, 2, ...
                    for case_list in Vehicle_1:
                        chain.append(case_list + [spawn_lane_index_2])
                return chain
        elif Vehicle['Initial_position'] in ['E2W', 'e2w']:
            if Vehicle['Actions'] in ['turn left', 'Turn Left', 'Turn left']:
                # coming form end to top turn left
                # spawn_lane_index_0 & spawn_lane_index_1 & destination can be:
                Vehicle_1.append(['-1T0_1_', '-1T0_0_', '->>>'])
                Vehicle_1.append(['-1T0_0_', '->>>', '->>'])
                # check the number of lanes
                chain = []
                for lane_id in range(Num_lanes):
                    spawn_lane_index_2 = lane_id  # 0, 1, 2, ...
                    for case_list in Vehicle_1:
                        chain.append(case_list + [spawn_lane_index_2])
                return chain
            elif Vehicle['Actions'] in ['turn right', 'Turn Right', 'Turn right']:
                # coming form end to top turn right
                # spawn_lane_index_0 & spawn_lane_index_1 & destination can be:
                Vehicle_1.append(['-1T0_1_', '-1T0_0_', '1T1_0_'])
                Vehicle_1.append(['-1T0_0_', '1T1_0_', '1T1_1_'])
                # check the number of lanes
                chain = []
                for lane_id in range(Num_lanes):
                    spawn_lane_index_2 = lane_id  # 0, 1, 2, ...
                    for case_list in Vehicle_1:
                        chain.append(case_list + [spawn_lane_index_2])
                return chain

def action_chain_r(Vehicle, Num_lanes):
    Vehicle_1 = []  # spawn_lane_index_0 & spawn_lane_index_1 & destination & spawn_lane_index_2
    # check initial position
    if Vehicle['Initial_position'] in ['Main road', 'Main Road', 'main road']:
        # drive on the main road
        # check actions
        if Vehicle['Actions'] in ['Move forward', 'Move Forward', 'move forward']:
            # coming from left to right and go straight on the main road
            # spawn_lane_index_0 & spawn_lane_index_1 & destination can be:
            Vehicle_1.append(['1r0_0_', '1r0_1_', '1r0_2_'])
            # check the number of lanes
            chain = []
            for lane_id in range(Num_lanes):
                spawn_lane_index_2 = lane_id  # 0, 1, 2, ...
                for case_list in Vehicle_1:
                    chain.append(case_list + [spawn_lane_index_2])
            return chain
    elif Vehicle['Initial_position'] in ['On-ramp', 'On-Ramp', 'on-ramp', 'on ramp']:
        # coming form left to right on ramp
        # check actions
        if Vehicle['Actions'] in ['Merge', 'merge']:
            # coming from right to left and go straight
            # spawn_lane_index_0 & spawn_lane_index_1 & destination can be:
            Vehicle_1.append(['1r0_0_', '1r1_4_', '1r0_1_', 0])
            return Vehicle_1
        elif Vehicle['Actions'] in ['turn left', 'Turn Left', 'Turn left']:
            Vehicle_1.append(['1r0_0_', '1r1_4_', '-1r0_0_', 0])
            return Vehicle_1
        elif Vehicle['Actions'] in ['turn right', 'Turn Right', 'Turn right']:
            Vehicle_1.append(['1r0_0_', '1r1_4_', '1r0_1_', 0])
            return Vehicle_1

def action_pairs(Actors, Road_network, road_type):
    # Input: {'Vehicle_1': {'Model': 'Sedan', 'Initial_position': 'W2E', 'Actions': 'Move forward', 'Speed': 'N/A'},
    # 'Vehicle_2': {'Model': 'Sedan', 'Initial_position': 'E2W', 'Actions': 'Enter the Wrong Way', 'Speed': 'N/A'}}

    # Output: {
    #   'Vehicle_1':[(spawn_lane_index_1, spawn_lane_index_2, spawn_lane_index_3, destination), ... ]
    #   'Vehicle_2':[(spawn_lane_index_1, spawn_lane_index_2, spawn_lane_index_3, destination), ... ]
    # }

    # For positions: convert N to W, S to E
    Num_cars = len(Actors)
    Num_lanes = (int(Road_network['Number of lanes']) + 1) // 2 # Number of lanes on a single way

    if road_type == 'Straight':
        V1_chain = action_chain_s(Actors['Vehicle_1'], Num_lanes)
        V2_chain = action_chain_s(Actors['Vehicle_2'], Num_lanes)
        if Num_cars == 3:
            V3_chain = action_chain_s(Actors['Vehicle_2'], Num_lanes)
            return {'Vehicle_1': V1_chain, 'Vehicle_2': V2_chain, 'Vehicle_3': V3_chain}
        else:
            return {'Vehicle_1': V1_chain, 'Vehicle_2': V2_chain}
    elif road_type == 'Curve':
        V1_chain = action_chain_c(Actors['Vehicle_1'], Num_lanes)
        V2_chain = action_chain_c(Actors['Vehicle_2'], Num_lanes)
        if Num_cars == 3:
            V3_chain = action_chain_c(Actors['Vehicle_2'], Num_lanes)
            return {'Vehicle_1': V1_chain, 'Vehicle_2': V2_chain, 'Vehicle_3': V3_chain}
        else:
            return {'Vehicle_1': V1_chain, 'Vehicle_2': V2_chain}
    elif road_type == 'Intersection':
        V1_chain = action_chain_x(Actors['Vehicle_1'], Num_lanes)
        V2_chain = action_chain_x(Actors['Vehicle_2'], Num_lanes)
        if Num_cars == 3:
            V3_chain = action_chain_x(Actors['Vehicle_2'], Num_lanes)
            return {'Vehicle_1': V1_chain, 'Vehicle_2': V2_chain, 'Vehicle_3': V3_chain}
        else:
            return {'Vehicle_1': V1_chain, 'Vehicle_2': V2_chain}
    elif road_type == 'T-intersection':
        V1_chain = action_chain_t(Actors['Vehicle_1'], Num_lanes, Road_network['Stem road direction'])
        V2_chain = action_chain_t(Actors['Vehicle_2'], Num_lanes, Road_network['Stem road direction'])
        if Num_cars == 3:
            V3_chain = action_chain_t(Actors['Vehicle_2'], Num_lanes, Road_network['Stem road direction'])
            return {'Vehicle_1': V1_chain, 'Vehicle_2': V2_chain, 'Vehicle_3': V3_chain}
        else:
            return {'Vehicle_1': V1_chain, 'Vehicle_2': V2_chain}
    elif road_type == 'Merging':
        V1_chain = action_chain_r(Actors['Vehicle_1'], Num_lanes)
        V2_chain = action_chain_r(Actors['Vehicle_2'], Num_lanes)
        if Num_cars == 3:
            V3_chain = action_chain_r(Actors['Vehicle_2'], Num_lanes)
            return {'Vehicle_1': V1_chain, 'Vehicle_2': V2_chain, 'Vehicle_3': V3_chain}
        else:
            return {'Vehicle_1': V1_chain, 'Vehicle_2': V2_chain}

def action_filter(actions):
    car_num = len(actions)
    if car_num == 3:
        v1_list = actions['Vehicle_1']
        v2_list = actions['Vehicle_2']
        v3_list = actions['Vehicle_3']
        pairs = []
        for v1_case in v1_list:
            # ['-1S0_0_', '->>>', '->>', 0]
            for v2_case in v2_list:
                # ['-1S0_0_', '->>>', '->>', 0]
                for v3_case in v3_list:
                    if v1_case != v2_case and v1_case != v3_case and v2_case != v3_case:
                        pairs.append(v1_case + v2_case + v3_case)
        return pairs
    else:
        v1_list = actions['Vehicle_1']
        v2_list = actions['Vehicle_2']
        pairs = []
        for v1_case in v1_list:
            # ['-1S0_0_', '->>>', '->>', 0]
            for v2_case in v2_list:
                # ['-1S0_0_', '->>>', '->>', 0]
                if v1_case != v2_case:
                    pairs.append(v1_case + v2_case)
        return pairs

def get_ego_model(Vehicle):
    # Sedan, SUV, Minivan, Pickup, Semi Truck
    # SVehicle, DefaultVehicle, MVehicle, LVehicle, XLVehicle
    if Vehicle['Model'] == 'Sedan':
        return 's'
    elif Vehicle['Model'] == 'SUV':
        return 'default'
    elif Vehicle['Model'] == 'Minivan':
        return 'm'
    elif Vehicle['Model'] == 'Pickup':
        return 'l'
    else:
        return 'xl'

def get_npc_model(Vehicle):
    # Sedan, SUV, Minivan, Pickup, Semi Truck
    # SVehicle, DefaultVehicle, MVehicle, LVehicle, XLVehicle
    if Vehicle['Model'] == 'Sedan':
        return SVehicle
    elif Vehicle['Model'] == 'SUV':
        return DefaultVehicle
    elif Vehicle['Model'] == 'Minivan':
        return MVehicle
    elif Vehicle['Model'] == 'Pickup':
        return LVehicle
    else:
        return XLVehicle

def get_sim_time(env_info):
    day_time_mapping = {
        'Daytime': '11:00',
        'Nighttime': '20:00'
    }
    return day_time_mapping.get(env_info.get('Time'), '11:00')

def run_straight(scenario_id, Actors, Road_network, Env, road_type, ADS, result_folder_path, crash_folder_path):
    # Get action chain
    actions = action_pairs(Actors, Road_network, road_type)
    actions = action_filter(actions)
    cases_num = len(actions)
    print(f"Generated number of cases: {cases_num}")
    counter = 0
    crash_num = 0
    daytime =get_sim_time(Env)
    for case in actions:
        # Set Vehicle_1 as agent
        # Build scenario_config
        # case: ['>', '>>', '>>>', 0, '-1S0_0_', '->>>', '>>>', 0]
        scenario_config = {'map_config': {'type': 'block_sequence',
                                          'config': 'S', # Define a straight road
                                          'lane_num': (int(Road_network['Number of lanes']) + 1) // 2 # No. lanes on a single way
                                          },
                           'agent_policy': IDMPolicy if ADS == 'IDM_policy' else ExpertPolicy, # ADS under test
                           'traffic_density': 0,
                           'agent_configs': {
                               'default_agent': {
                                   'use_special_color': True,
                                   'spawn_lane_index': (case[0], case[1], case[3]),
                                   'destination': case[2],
                                   'vehicle_model': get_ego_model(Actors['Vehicle_1'])
                               }
                           },
                           'use_render': True,
                           'daytime': daytime
                           }
        env = MetaDriveEnv(scenario_config)
        frames = []
        try:
            crash_flag = 0
            set_log_level(logging.CRITICAL)
            env.reset()
            cfg = copy.deepcopy(env.config["vehicle_config"])
            cfg["navigation_module"] = NodeNetworkNavigation
            cfg['spawn_lane_index'] = (case[4], case[5], case[7])
            cfg['destination'] = case[6]
            npc_model = get_npc_model(Actors['Vehicle_2'])
            npc = env.engine.spawn_object(npc_model, vehicle_config=cfg)
            env.engine.add_policy(npc.id, IDMPolicy if ADS == 'IDM_policy' else ExpertPolicy, npc, env.engine.generate_seed())
            # Build NPC 2
            if len(Actors) == 3:
                cfg_1 = copy.deepcopy(env.config["vehicle_config"])
                cfg_1["navigation_module"] = NodeNetworkNavigation
                cfg_1['spawn_lane_index'] = (case[8], case[9], case[11])  # Extract from DSL
                cfg_1['destination'] = case[10]  # Extract from DSL
                npc_model_1 = get_npc_model(Actors['Vehicle_3'])
                npc_1 = env.engine.spawn_object(npc_model_1, vehicle_config=cfg_1)
                env.engine.add_policy(npc_1.id, IDMPolicy if ADS == 'IDM_policy' else ExpertPolicy, npc_1, env.engine.generate_seed())
            for _ in range(100):
                # NPC action
                npc.before_step(env.engine.get_policy(npc.name).act(True))
                if len(Actors) == 3:
                    npc_1.before_step(env.engine.get_policy(npc_1.name).act(True))
                p = env.engine.get_policy(env.agent.name)
                _, r, _, _, info = env.step(p.act(True))
                frame = env.render(mode="topdown",
                                   window=False,
                                   screen_size=(800, 400),
                                   draw_target_vehicle_trajectory=False,
                                   scaling=4,
                                   camera_position=None)
                frames.append(frame)
                if info['crash']:
                    crash_flag = 1
                    print('Crash detected!')
            if crash_flag == 0:
                name_gif = str(result_folder_path / f"{scenario_id}_{counter}.gif")
                generate_gif(frames, gif_name=name_gif)
                print('Simulation Gif has been saved!')
                time.sleep(1)
            else:
                crash_num += 1
                name_gif = str(crash_folder_path / f"{scenario_id}_{counter}.gif")
                generate_gif(frames, gif_name=name_gif)
                print('Simulation Gif has been saved!')
                time.sleep(1)
        finally:
            env.close()
            counter += 1
    return cases_num, crash_num

def run_curve(scenario_id, Actors, Road_network, Env, road_type, ADS, result_folder_path, crash_folder_path):
    # Get action chain
    actions = action_pairs(Actors, Road_network, road_type)
    actions = action_filter(actions)
    cases_num = len(actions)
    print(f"Generated number of cases: {cases_num}")
    counter = 0
    crash_num = 0
    daytime = get_sim_time(Env)
    for case in actions:
        # Set Vehicle_1 as agent
        # Build scenario_config
        # case: ['>', '>>', '>>>', 0, '-1S0_0_', '->>>', '>>>', 0]
        scenario_config = {'map_config': {'type': 'block_sequence',
                                          'config': 'C',  # Define a straight road
                                          'lane_num': (int(Road_network['Number of lanes']) + 1) // 2
                                          # No. lanes on a single way
                                          },
                           'agent_policy': IDMPolicy if ADS == 'IDM_policy' else ExpertPolicy,  # ADS under test
                           'traffic_density': 0,
                           'agent_configs': {
                               'default_agent': {
                                   'use_special_color': True,
                                   'spawn_lane_index': (case[0], case[1], case[3]),
                                   'destination': case[2],
                                   'vehicle_model': get_ego_model(Actors['Vehicle_1'])
                               }
                           },
                           'use_render': True,
                           'daytime': daytime
                           }
        env = MetaDriveEnv(scenario_config)
        frames = []
        try:
            crash_flag = 0
            set_log_level(logging.CRITICAL)
            env.reset()
            cfg = copy.deepcopy(env.config["vehicle_config"])
            cfg["navigation_module"] = NodeNetworkNavigation
            cfg['spawn_lane_index'] = (case[4], case[5], case[7])
            cfg['destination'] = case[6]
            npc_model = get_npc_model(Actors['Vehicle_2'])
            npc = env.engine.spawn_object(npc_model, vehicle_config=cfg)
            env.engine.add_policy(npc.id, IDMPolicy if ADS == 'IDM_policy' else ExpertPolicy, npc,
                                  env.engine.generate_seed())
            # Build NPC 2
            if len(Actors) == 3:
                cfg_1 = copy.deepcopy(env.config["vehicle_config"])
                cfg_1["navigation_module"] = NodeNetworkNavigation
                cfg_1['spawn_lane_index'] = (case[8], case[9], case[11])  # Extract from DSL
                cfg_1['destination'] = case[10]  # Extract from DSL
                npc_model_1 = get_npc_model(Actors['Vehicle_3'])
                npc_1 = env.engine.spawn_object(npc_model_1, vehicle_config=cfg_1)
                env.engine.add_policy(npc_1.id, IDMPolicy if ADS == 'IDM_policy' else ExpertPolicy, npc_1,
                                      env.engine.generate_seed())
            for _ in range(100):
                # NPC action
                npc.before_step(env.engine.get_policy(npc.name).act(True))
                if len(Actors) == 3:
                    npc_1.before_step(env.engine.get_policy(npc_1.name).act(True))
                p = env.engine.get_policy(env.agent.name)
                _, r, _, _, info = env.step(p.act(True))
                frame = env.render(mode="topdown",
                                   window=False,
                                   screen_size=(800, 400),
                                   draw_target_vehicle_trajectory=False,
                                   scaling=4,
                                   camera_position=None)
                frames.append(frame)
                if info['crash']:
                    crash_flag = 1
                    print('Crash detected!')
            if crash_flag == 0:
                name_gif = str(result_folder_path / f"{scenario_id}_{counter}.gif")
                generate_gif(frames, gif_name=name_gif)
                print('Simulation Gif has been saved!')
                time.sleep(1)
            else:
                crash_num += 1
                name_gif = str(crash_folder_path / f"{scenario_id}_{counter}.gif")
                generate_gif(frames, gif_name=name_gif)
                print('Simulation Gif has been saved!')
                time.sleep(1)
        finally:
            env.close()
            counter += 1
    return cases_num, crash_num

def run_intersection(scenario_id, Actors, Road_network, Env, road_type, ADS, result_folder_path, crash_folder_path):
    # Get action chain
    actions = action_pairs(Actors, Road_network, road_type)
    actions = action_filter(actions)
    cases_num = len(actions)
    print(f"Generated number of cases: {cases_num}")
    counter = 0
    crash_num = 0
    daytime = get_sim_time(Env)
    for case in actions:
        # Set Vehicle_1 as agent
        # Build scenario_config
        # case: ['>', '>>', '>>>', 0, '-1S0_0_', '->>>', '>>>', 0]
        scenario_config = {'map_config': {'type': 'block_sequence',
                                          'config': 'X',  # Define a straight road
                                          'lane_num': (int(Road_network['Number of lanes']) + 1) // 2
                                          # No. lanes on a single way
                                          },
                           'agent_policy': IDMPolicy if ADS == 'IDM_policy' else ExpertPolicy,  # ADS under test
                           'traffic_density': 0,
                           'agent_configs': {
                               'default_agent': {
                                   'use_special_color': True,
                                   'spawn_lane_index': (case[0], case[1], case[3]),
                                   'destination': case[2],
                                   'vehicle_model': get_ego_model(Actors['Vehicle_1'])
                               }
                           },
                           'use_render': True,
                           'daytime': daytime
                           }
        env = MetaDriveEnv(scenario_config)
        frames = []
        try:
            crash_flag = 0
            set_log_level(logging.CRITICAL)
            env.reset()
            cfg = copy.deepcopy(env.config["vehicle_config"])
            cfg["navigation_module"] = NodeNetworkNavigation
            cfg['spawn_lane_index'] = (case[4], case[5], case[7])
            cfg['destination'] = case[6]
            npc_model = get_npc_model(Actors['Vehicle_2'])
            npc = env.engine.spawn_object(npc_model, vehicle_config=cfg)
            env.engine.add_policy(npc.id, IDMPolicy if ADS == 'IDM_policy' else ExpertPolicy, npc,
                                  env.engine.generate_seed())
            # Build NPC 2
            if len(Actors) == 3:
                cfg_1 = copy.deepcopy(env.config["vehicle_config"])
                cfg_1["navigation_module"] = NodeNetworkNavigation
                cfg_1['spawn_lane_index'] = (case[8], case[9], case[11])  # Extract from DSL
                cfg_1['destination'] = case[10]  # Extract from DSL
                npc_model_1 = get_npc_model(Actors['Vehicle_3'])
                npc_1 = env.engine.spawn_object(npc_model_1, vehicle_config=cfg_1)
                env.engine.add_policy(npc_1.id, IDMPolicy if ADS == 'IDM_policy' else ExpertPolicy, npc_1,
                                      env.engine.generate_seed())
            for _ in range(100):
                # NPC action
                npc.before_step(env.engine.get_policy(npc.name).act(True))
                if len(Actors) == 3:
                    npc_1.before_step(env.engine.get_policy(npc_1.name).act(True))
                p = env.engine.get_policy(env.agent.name)
                _, r, _, _, info = env.step(p.act(True))
                frame = env.render(mode="topdown",
                                   window=False,
                                   screen_size=(800, 400),
                                   draw_target_vehicle_trajectory=False,
                                   scaling=4,
                                   camera_position=None)
                frames.append(frame)
                if info['crash']:
                    crash_flag = 1
                    print('Crash detected!')
            if crash_flag == 0:
                name_gif = str(result_folder_path / f"{scenario_id}_{counter}.gif")
                generate_gif(frames, gif_name=name_gif)
                print('Simulation Gif has been saved!')
                time.sleep(1)
            else:
                crash_num += 1
                name_gif = str(crash_folder_path / f"{scenario_id}_{counter}.gif")
                generate_gif(frames, gif_name=name_gif)
                print('Simulation Gif has been saved!')
                time.sleep(1)
        finally:
            env.close()
            counter += 1
    return cases_num, crash_num

def run_t_intersection(scenario_id, Actors, Road_network, Env, road_type, ADS, result_folder_path, crash_folder_path):
    # Get action chain
    actions = action_pairs(Actors, Road_network, road_type)
    actions = action_filter(actions)
    cases_num = len(actions)
    print(f"Generated number of cases: {cases_num}")
    counter = 0
    crash_num = 0
    daytime = get_sim_time(Env)
    for case in actions:
        # Set Vehicle_1 as agent
        # Build scenario_config
        # case: ['>', '>>', '>>>', 0, '-1S0_0_', '->>>', '>>>', 0]
        scenario_config = {'map_config': {'type': 'block_sequence',
                                          'config': 'T',  # Define a T-intersection
                                          'lane_num': (int(Road_network['Number of lanes']) + 1) // 2
                                          # No. lanes on a single way
                                          },
                           'agent_policy': IDMPolicy if ADS == 'IDM_policy' else ExpertPolicy,  # ADS under test
                           'traffic_density': 0,
                           'agent_configs': {
                               'default_agent': {
                                   'use_special_color': True,
                                   'spawn_lane_index': (case[0], case[1], case[3]),
                                   'destination': case[2],
                                   'vehicle_model': get_ego_model(Actors['Vehicle_1'])
                               }
                           },
                           'use_render': True,
                           'daytime': daytime
                           }
        env = MetaDriveEnv(scenario_config)
        frames = []
        try:
            crash_flag = 0
            set_log_level(logging.CRITICAL)
            env.reset()
            cfg = copy.deepcopy(env.config["vehicle_config"])
            cfg["navigation_module"] = NodeNetworkNavigation
            cfg['spawn_lane_index'] = (case[4], case[5], case[7])
            cfg['destination'] = case[6]
            npc_model = get_npc_model(Actors['Vehicle_2'])
            npc = env.engine.spawn_object(npc_model, vehicle_config=cfg)
            env.engine.add_policy(npc.id, IDMPolicy if ADS == 'IDM_policy' else ExpertPolicy, npc,
                                  env.engine.generate_seed())
            # Build NPC 2
            if len(Actors) == 3:
                cfg_1 = copy.deepcopy(env.config["vehicle_config"])
                cfg_1["navigation_module"] = NodeNetworkNavigation
                cfg_1['spawn_lane_index'] = (case[8], case[9], case[11])  # Extract from DSL
                cfg_1['destination'] = case[10]  # Extract from DSL
                npc_model_1 = get_npc_model(Actors['Vehicle_3'])
                npc_1 = env.engine.spawn_object(npc_model_1, vehicle_config=cfg_1)
                env.engine.add_policy(npc_1.id, IDMPolicy if ADS == 'IDM_policy' else ExpertPolicy, npc_1,
                                      env.engine.generate_seed())
            for _ in range(100):
                # NPC action
                npc.before_step(env.engine.get_policy(npc.name).act(True))
                if len(Actors) == 3:
                    npc_1.before_step(env.engine.get_policy(npc_1.name).act(True))
                p = env.engine.get_policy(env.agent.name)
                _, r, _, _, info = env.step(p.act(True))
                frame = env.render(mode="topdown",
                                   window=False,
                                   screen_size=(800, 400),
                                   draw_target_vehicle_trajectory=False,
                                   scaling=4,
                                   camera_position=None)
                frames.append(frame)
                if info['crash']:
                    crash_flag = 1
                    print('Crash detected!')
            if crash_flag == 0:
                name_gif = str(result_folder_path / f"{scenario_id}_{counter}.gif")
                generate_gif(frames, gif_name=name_gif)
                print('Simulation Gif has been saved!')
                time.sleep(1)
            else:
                crash_num += 1
                name_gif = str(crash_folder_path / f"{scenario_id}_{counter}.gif")
                generate_gif(frames, gif_name=name_gif)
                print('Simulation Gif has been saved!')
                time.sleep(1)
        finally:
            env.close()
            counter += 1
    return cases_num, crash_num

def run_merging(scenario_id, Actors, Road_network, Env, road_type, ADS, result_folder_path, crash_folder_path):
    # Get action chain
    actions = action_pairs(Actors, Road_network, road_type)
    actions = action_filter(actions)
    cases_num = len(actions)
    print(f"Generated number of cases: {cases_num}")
    counter = 0
    crash_num = 0
    daytime = get_sim_time(Env)
    for case in actions:
        # Set Vehicle_1 as agent
        # Build scenario_config
        # case: ['>', '>>', '>>>', 0, '-1S0_0_', '->>>', '>>>', 0]
        scenario_config = {'map_config': {'type': 'block_sequence',
                                          'config': 'r',  # Define an in ramp (merging)
                                          'lane_num': (int(Road_network['Number of lanes']) + 1) // 2
                                          # No. lanes on a single way
                                          },
                           'agent_policy': IDMPolicy if ADS == 'IDM_policy' else ExpertPolicy,  # ADS under test
                           'traffic_density': 0,
                           'agent_configs': {
                               'default_agent': {
                                   'use_special_color': True,
                                   'spawn_lane_index': (case[0], case[1], case[3]),
                                   'destination': case[2],
                                   'vehicle_model': get_ego_model(Actors['Vehicle_1'])
                               }
                           },
                           'use_render': True,
                           'daytime': daytime
                           }
        env = MetaDriveEnv(scenario_config)
        frames = []
        try:
            crash_flag = 0
            set_log_level(logging.CRITICAL)
            env.reset()
            cfg = copy.deepcopy(env.config["vehicle_config"])
            cfg["navigation_module"] = NodeNetworkNavigation
            cfg['spawn_lane_index'] = (case[4], case[5], case[7])
            cfg['destination'] = case[6]
            npc_model = get_npc_model(Actors['Vehicle_2'])
            npc = env.engine.spawn_object(npc_model, vehicle_config=cfg)
            env.engine.add_policy(npc.id, IDMPolicy if ADS == 'IDM_policy' else ExpertPolicy, npc,
                                  env.engine.generate_seed())
            # Build NPC 2
            if len(Actors) == 3:
                cfg_1 = copy.deepcopy(env.config["vehicle_config"])
                cfg_1["navigation_module"] = NodeNetworkNavigation
                cfg_1['spawn_lane_index'] = (case[8], case[9], case[11])  # Extract from DSL
                cfg_1['destination'] = case[10]  # Extract from DSL
                npc_model_1 = get_npc_model(Actors['Vehicle_3'])
                npc_1 = env.engine.spawn_object(npc_model_1, vehicle_config=cfg_1)
                env.engine.add_policy(npc_1.id, IDMPolicy if ADS == 'IDM_policy' else ExpertPolicy, npc_1,
                                      env.engine.generate_seed())
            for _ in range(100):
                # NPC action
                npc.before_step(env.engine.get_policy(npc.name).act(True))
                if len(Actors) == 3:
                    npc_1.before_step(env.engine.get_policy(npc_1.name).act(True))
                p = env.engine.get_policy(env.agent.name)
                _, r, _, _, info = env.step(p.act(True))
                frame = env.render(mode="topdown",
                                   window=False,
                                   screen_size=(800, 400),
                                   draw_target_vehicle_trajectory=False,
                                   scaling=4,
                                   camera_position=None)
                frames.append(frame)
                if info['crash']:
                    crash_flag = 1
                    print('Crash detected!')
            if crash_flag == 0:
                name_gif = str(result_folder_path / f"{scenario_id}_{counter}.gif")
                generate_gif(frames, gif_name=name_gif)
                print('Simulation Gif has been saved!')
                time.sleep(1)
            else:
                crash_num += 1
                name_gif = str(crash_folder_path / f"{scenario_id}_{counter}.gif")
                generate_gif(frames, gif_name=name_gif)
                print('Simulation Gif has been saved!')
                time.sleep(1)
        finally:
            env.close()
            counter += 1
    return cases_num, crash_num

def get_dsl(dsls, id):
    for dsl in dsls:
        if dsl['Scenario'] == id:
            return dsl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dsl_path',
                        default=r'remember to setup')
    parser.add_argument('--meta_msg',
                        default=r'remember to setup')
    parser.add_argument('--ADS', default='IDM_policy') # Two policies IDMPolicy & ExpertPolicy
    args = parser.parse_args()

    with open(args.dsl_path,'rb') as file:
        # [{DSL1},{DSL2},{DSL3},...{DSLN}]
        DSLs = pickle.load(file)

    with open(args.meta_msg,'rb') as file:
        # [['Straight', 2, 'opposite direction', '128697'], ..., ['T-intersection', 2, 'crossing traffic', '119839']]
        meta_msg = pickle.load(file)

    # Create result folder
    current_path = Path.cwd()
    parent_path = current_path.parent.parent
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_folder = parent_path / 'Experiment_results' / f"MetaDrive_{args.ADS}_results_{current_time}"
    os.makedirs(result_folder, exist_ok=True)

    crash_folder = result_folder / f"Detected_crash"
    os.makedirs(crash_folder, exist_ok=True)

    total_cases_num = 0
    total_crash_num = 0

    print(f'============ADS Testing ({args.ADS}) Start!============')
    start_time = time.time()

    for case in meta_msg:
        scenario_id = case[-1] # '128697'
        dsl = get_dsl(DSLs,scenario_id)
        Actors = dsl['Actors'][0]
        Road_network = dsl['Road network']
        Env = dsl['Env']
        road_type = case[0]
        print(f'========Start to simulate case {scenario_id}!========')
        if road_type == 'Straight':
            cases_num, crash_num = run_straight(scenario_id, Actors, Road_network, Env, road_type, args.ADS, result_folder, crash_folder)
            total_cases_num = total_cases_num + cases_num
            total_crash_num = total_crash_num + crash_num
        elif road_type == 'Curve':
            cases_num, crash_num = run_curve(scenario_id, Actors, Road_network, Env, road_type, args.ADS, result_folder, crash_folder)
            total_cases_num = total_cases_num + cases_num
            total_crash_num = total_crash_num + crash_num
        elif road_type == 'Intersection':
            cases_num, crash_num = run_intersection(scenario_id, Actors, Road_network, Env, road_type, args.ADS, result_folder, crash_folder)
            total_cases_num = total_cases_num + cases_num
            total_crash_num = total_crash_num + crash_num
        elif road_type == 'T-intersection':
            cases_num, crash_num = run_t_intersection(scenario_id, Actors, Road_network, Env, road_type, args.ADS, result_folder, crash_folder)
            total_cases_num = total_cases_num + cases_num
            total_crash_num = total_crash_num + crash_num
        elif road_type == 'Merging':
            cases_num, crash_num = run_merging(scenario_id, Actors, Road_network, Env, road_type, args.ADS, result_folder, crash_folder)
            total_cases_num = total_cases_num + cases_num
            total_crash_num = total_crash_num + crash_num
        print(f'-----Case {scenario_id} finished!-----')
        print(f'Number of scenarios generated under case: {scenario_id} - ', str(cases_num))
        print(f'Number of crashes detected under case: {scenario_id} - ', str(crash_num))

    print(f'============ADS Testing ({args.ADS}) Finished!============')
    print('Total number of generated cases: ', str(total_cases_num))
    print('Total number of detected crash cases: ', str(total_crash_num))
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Program execution time: {execution_time:.6f} s")


if __name__=='__main__':
    main()