import argparse
import pickle
from datetime import datetime
import os
import shutil

def generate_straight(car_num, init_direction, scenario_id, folder_path):
    # generate a result folder
    prompts_folder = f"{folder_path}/{scenario_id}"
    os.makedirs(prompts_folder, exist_ok=True)
    prompts_folder = os.path.abspath(prompts_folder)

    # system prompt
    source_path = './Knowledge_base/Crash_dataset/Straight/Prompts/system.txt'
    destination_path = f'{prompts_folder}/system.txt'
    shutil.copy(source_path, destination_path)

    # user prompt 1
    # same direction or opposite direction
    if init_direction == 'same direction':
        source_path = './Knowledge_base/Crash_dataset/Straight/same_direction/User_1.txt'
        destination_path = f'{prompts_folder}/user1.txt'
        shutil.copy(source_path, destination_path)
    else:
        source_path = './Knowledge_base/Crash_dataset/Straight/opposite_direction/User_1.txt'
        destination_path = f'{prompts_folder}/user1.txt'
        shutil.copy(source_path, destination_path)


def generate_curve(car_num, init_direction, scenario_id, folder_path):
    # generate a result folder
    prompts_folder = f"{folder_path}/{scenario_id}"
    os.makedirs(prompts_folder, exist_ok=True)
    prompts_folder = os.path.abspath(prompts_folder)

    # System prompt
    source_path = './Knowledge_base/Crash_dataset/Curve/Prompts/system.txt'
    destination_path = f'{prompts_folder}/system.txt'
    shutil.copy(source_path, destination_path)

    # user prompt 1
    source_path = './Knowledge_base/Crash_dataset/Curve/Prompts/User_1.txt'
    destination_path = f'{prompts_folder}/user1.txt'
    shutil.copy(source_path, destination_path)

def generate_intersection(car_num, init_direction, scenario_id, folder_path):
    # generate a result folder
    prompts_folder = f"{folder_path}/{scenario_id}"
    os.makedirs(prompts_folder, exist_ok=True)
    prompts_folder = os.path.abspath(prompts_folder)

    # System prompt
    source_path = './Knowledge_base/Crash_dataset/Intersection/Prompts/system.txt'
    destination_path = f'{prompts_folder}/system.txt'
    shutil.copy(source_path, destination_path)

    # user prompt 1
    source_path = './Knowledge_base/Crash_dataset/Intersection/Prompts/User_1.txt'
    destination_path = f'{prompts_folder}/user1.txt'
    shutil.copy(source_path, destination_path)



def generate_t_intersection(car_num, init_direction, scenario_id, folder_path):
    # generate a result folder
    prompts_folder = f"{folder_path}/{scenario_id}"
    os.makedirs(prompts_folder, exist_ok=True)
    prompts_folder = os.path.abspath(prompts_folder)

    # System prompt
    source_path = './Knowledge_base/Crash_dataset/T-intersection/Prompts/system.txt'
    destination_path = f'{prompts_folder}/system.txt'
    shutil.copy(source_path, destination_path)

    # user prompt 1
    source_path = './Knowledge_base/Crash_dataset/T-intersection/Prompts/User_1.txt'
    destination_path = f'{prompts_folder}/user1.txt'
    shutil.copy(source_path, destination_path)


def generate_merging(car_num, init_direction, scenario_id, folder_path):
    # generate a result folder
    prompts_folder = f"{folder_path}/{scenario_id}"
    os.makedirs(prompts_folder, exist_ok=True)
    prompts_folder = os.path.abspath(prompts_folder)

    # System prompt
    source_path = './Knowledge_base/Crash_dataset/Merging/Prompts/system.txt'
    destination_path = f'{prompts_folder}/system.txt'
    shutil.copy(source_path, destination_path)

    # user prompt 1
    source_path = './Knowledge_base/Crash_dataset/Merging/Prompts/User_1.txt'
    destination_path = f'{prompts_folder}/user1.txt'
    shutil.copy(source_path, destination_path)


def main():
    """
    Input: Meta message (from Meta_Message_Extraction)
    :return:
    Scenario ID,
    Prompts: System, User 1, Assistant 1, User 2
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='', type=str)
    args = parser.parse_args()

    with open(args.data,'rb') as file:
        data = pickle.load(file)

    # Create result folder
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_folder = f"./Experiment_results/Prompts_generation_results_{current_time}"
    os.makedirs(result_folder, exist_ok=True)
    folder_path = os.path.abspath(result_folder)

    for crash_case in data:
        road_type = crash_case[0]
        car_num = crash_case[1]
        init_direction = crash_case[2]
        scenario_id = crash_case[3]
        if road_type == 'Straight':
            generate_straight(car_num, init_direction, scenario_id, folder_path)
        elif road_type == 'Curve':
            generate_curve(car_num, init_direction, scenario_id, folder_path)
        elif road_type == 'Intersection':
            generate_intersection(car_num, init_direction, scenario_id, folder_path)
        elif road_type == 'T-intersection':
            generate_t_intersection(car_num, init_direction, scenario_id, folder_path)
        elif road_type == 'Merging':
            generate_merging(car_num, init_direction, scenario_id, folder_path)


if __name__ == '__main__':
    main()