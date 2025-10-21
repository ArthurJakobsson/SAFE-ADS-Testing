import time
from openai import OpenAI
import argparse
from datetime import datetime
import os
import base64
import re
import pickle

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_results(message):
    road_type_match = re.search(r'["\']Road type["\']:\s*["\']([^"\']+)["\']', message)
    car_num_match = re.search(r'["\']Number of cars["\']:\s*(\d+)', message)
    drive_dir_match = re.search(r'["\']Driving direction["\']:\s*["\']([^"\']+)["\']', message)

    road_type = road_type_match.group(1) if road_type_match else None
    car_num = int(car_num_match.group(1)) if car_num_match else None
    drive_dir = drive_dir_match.group(1) if drive_dir_match else None

    return [road_type, car_num, drive_dir]

def message_validation(system_info, meta_msg, sketch, summary, record, folder_path, model):
    client = OpenAI()
    input_msg = f"Message: ['Road type': {meta_msg[0]}, 'Number of cars': {meta_msg[1]}, 'Driving direction': {meta_msg[2]}]"
    response = client.chat.completions.create(
        model=model,
        messages=
        [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_info
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{input_msg}\nSketch:\n"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{sketch}"
                        }
                    },
                    {
                        "type": "text",
                        "text": f"Summary:\n{summary}"
                    }
                ]
            }
        ],
        response_format={
            "type": "text"
        },
        temperature=1,
        max_completion_tokens=512,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    model_output = response.choices[0].message.content
    # save raw results
    file_name = f"{record}_msg_validate.txt"
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(model_output)
    # judge whether consistent
    if re.search(r'\bnot consistent\b', model_output, re.IGNORECASE):
        return 0
    else:
        return 1

def meta_msg_extraction(sketch, summary, example_sketch, example_summary, user_prompts_1, user_prompts_2, record, folder_path,
                         model):
    # Multi-round dialog, COT prompting, Few shots learning
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an experienced road engineering expert who is skilled in identifying road types[Straight, Curve, Intersection, T-intersection, Merging] from map sketches and vehicle behavior description."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompts_1
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Sure, please go ahead and provide the example so I can better understand the task and assist you accordingly."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Example sketch: \n"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{example_sketch}"
                        }
                    },
                    {
                        "type": "text",
                        "text": f"\nExample Summary: \n{example_summary}\n"
                    },
                    {
                        "type": "text",
                        "text": user_prompts_2
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Got it! Please provide a new crash case and I'll follow the process to extract the required information."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please analyse this case and give your answer:\nSketch:\n"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{sketch}"
                        }
                    },
                    {
                        "type": "text",
                        "text": f"\nSummary: \n{summary}"
                    }
                ]
            }
        ],
        # control creativity and randomness
        response_format={
            "type": "text"
        },
        temperature=1,
        max_completion_tokens=512,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    model_output = response.choices[0].message.content
    # save raw results
    file_name = f"{record}_meta_msg.txt"
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(model_output)
    # extract meta results
    meta = extract_results(model_output)
    print('Extracted meta info: ', meta)
    return meta

def main():
    project_path = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default=os.path.join(project_path, 'Crash_dataset'), type=str,
                        help='Path of crash dataset.')
    parser.add_argument('--gpt', default='gpt-4o')
    args = parser.parse_args()

    # Create result folder
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_folder = f"./Experiment_results/Meta_Message_results_{current_time}"
    os.makedirs(result_folder, exist_ok=True)
    folder_path = os.path.abspath(result_folder)

    ID_list = os.listdir(args.data_path)
    records = [str(ID) for ID in ID_list]

    # ----------for RQ 1

    # Prompts preparation
    example_sketch = encode_image(r'./Knowledge_base/Meta_prompts/example_sketch.jpg')  # Curve example 109536
    with open(
            r'./Knowledge_base/Meta_prompts/example_summary.txt',
            'r', encoding='utf-8') as file:
        example_summary = file.read()
    with open(
            r'./Knowledge_base/Meta_prompts/p1.txt',
            'r', encoding='utf-8') as file:
        user_prompts_1 = file.read()
    with open(
            r'./Knowledge_base/Meta_prompts/p2.txt',
            'r', encoding='utf-8') as file:
        user_prompts_2 = file.read()
    with open(
            r'./Knowledge_base/Meta_prompts/system.txt',
            'r', encoding='utf-8') as file:
        system_info = file.read()

    # Information Extraction
    Ext_final_res = []
    for record in records:
        # Prepare record
        record_path = os.path.join(args.data_path,record)
        # Encode crash sketch
        sketch = encode_image(os.path.join(record_path, 'Sketch.jpg'))
        # Get crash summary
        with open(os.path.join(record_path, 'Summary.txt'), 'r', encoding='utf-8') as file:
            summary = file.read()

        meta_msg = meta_msg_extraction(sketch,
                             summary,
                             example_sketch,
                             example_summary,
                             user_prompts_1,
                             user_prompts_2,
                             record,
                             folder_path,
                             args.gpt)
        time.sleep(1)
        # validation
        vali_result = message_validation(system_info, meta_msg,sketch, summary, record, folder_path, args.gpt)
        if vali_result == 1:
            print(f"Case {record} passed validation!")
            print(f"Case {record} finished!")
            meta_msg.append(str(record))
            Ext_final_res.append(meta_msg)
            time.sleep(1)
        elif vali_result == 0:
            print(f"Case {record} failed validation~")
            print('Extract meta message again!')
            meta_msg = meta_msg_extraction(sketch,
                                           summary,
                                           example_sketch,
                                           example_summary,
                                           user_prompts_1,
                                           user_prompts_2,
                                           record,
                                           folder_path,
                                           args.gpt)
            meta_msg.append(str(record))
            Ext_final_res.append(meta_msg)
            print(f"Case {record} finished!")

    with open(os.path.join(folder_path, 'meta_data_results.pkl'), 'wb') as f:
        pickle.dump(Ext_final_res, f)
    print(f"Results have been saved!")

if __name__ == '__main__':
    main()