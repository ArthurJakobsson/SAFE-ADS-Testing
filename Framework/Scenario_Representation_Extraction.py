import time
from openai import OpenAI
import argparse
from datetime import datetime
import os
import base64
import re
import json
import pickle

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def _extract_json_block(text):
    """Extract a JSON object from an LLM response, tolerant of formatting.

    GPT-4o reliably wraps output in a ```json fence; open VLMs (Qwen2.5-VL etc.) are less
    consistent, so we fall back to any ``` fence and finally to the first balanced {...}.
    """
    m = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r'```\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        return m.group(1)
    start = text.find('{')
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    raise ValueError("No JSON object found in model output:\n" + text[:500])

def get_dsl(record, prompts_folder, road_type, direction, model, results_path):
    # Original data
    # Get Sketch
    Sketch = encode_image(f'./Crash_dataset/{record}/Sketch.jpg')
    # Get Summary
    with open(
            f'./Crash_dataset/{record}/Summary.txt',
            'r', encoding='utf-8') as file:
        Summary = file.read()

    # Get System prompts
    with open(
            f'{prompts_folder}/{record}/system.txt',
            'r', encoding='utf-8') as file:
        System = file.read()
    # Get User 1
    with open(
            f'{prompts_folder}/{record}/user1.txt',
            'r', encoding='utf-8') as file:
        User1 = file.read()

    # Get Example Sketch
    if road_type == 'Straight':
        if direction == 'same direction':
            Example_sketch = encode_image('./Knowledge_base/Crash_dataset/Straight/same_direction/100343/Sketch.jpg')
            # Example_summary
            with open(
                    './Knowledge_base/Crash_dataset/Straight/same_direction/100343/Summary.txt',
                    'r', encoding='utf-8') as file:
                Example_summary = file.read()
        else:
            Example_sketch = encode_image('./Knowledge_base/Crash_dataset/Straight/opposite_direction/109525/Sketch.jpg')
            # Example_summary
            with open(
                    './Knowledge_base/Crash_dataset/Straight/opposite_direction/109525/Summary.txt',
                    'r', encoding='utf-8') as file:
                Example_summary = file.read()
    elif road_type == 'Curve':
        Example_sketch = encode_image('./Knowledge_base/Crash_dataset/Curve/99817/Sketch.jpg')
        # Example_summary
        with open(
                './Knowledge_base/Crash_dataset/Curve/99817/Summary.txt',
                'r', encoding='utf-8') as file:
            Example_summary = file.read()
    elif road_type == 'Intersection':
        Example_sketch = encode_image('./Knowledge_base/Crash_dataset/Intersection/100237/Sketch.jpg')
        # Example_summary
        with open(
                './Knowledge_base/Crash_dataset/Intersection/100237/Summary.txt',
                'r', encoding='utf-8') as file:
            Example_summary = file.read()
    elif road_type == 'T-intersection':
        Example_sketch = encode_image('./Knowledge_base/Crash_dataset/T-intersection/100271/Sketch.jpg')
        # Example_summary
        with open(
                './Knowledge_base/Crash_dataset/T-intersection/100271/Summary.txt',
                'r', encoding='utf-8') as file:
            Example_summary = file.read()
    elif road_type == 'Merging':
        Example_sketch = encode_image('./Knowledge_base/Crash_dataset/Merging/103341/Sketch.jpg')
        # Example_summary
        with open(
                './Knowledge_base/Crash_dataset/Merging/103341/Summary.txt',
                'r', encoding='utf-8') as file:
            Example_summary = file.read()

    client = OpenAI()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": System
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Here's an example with detailed analysis for you to better understand this job: \nSketch: \n"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{Example_sketch}"
                        }
                    },
                    {
                        "type": "text",
                        "text": f"\nSummary: \n{Example_summary}\n"
                    },
                    {
                        "type": "text",
                        "text": User1
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
                            "url": f"data:image/jpeg;base64,{Sketch}"
                        }
                    },
                    {
                        "type": "text",
                        "text": f"\nSummary: \n{Summary}"
                    }
                ]
            }
        ],
        # control creativity and randomness
        response_format={
            "type": "text"
        },
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    model_output = response.choices[0].message.content
    # save raw results
    file_name = f"{record}_dsl_raw.txt"
    file_path = os.path.join(results_path, file_name)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(model_output)

    json_match = _extract_json_block(model_output)
    json_data = json.loads(json_match)

    return model_output, json_data

def dsl_validation(dsl, record, model, results_path):
    # Original data
    # Get Sketch
    Sketch = encode_image(f'./Crash_dataset/{record}/Sketch.jpg')
    # Get Summary
    with open(
            f'./Crash_dataset/{record}/Summary.txt',
            'r', encoding='utf-8') as file:
        Summary = file.read()

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful assistant, I need you to help me verify whether the analysis results are consistent with the original data. "
                                "The original data includes an accident summary and an accident sketch. "
                                "The accident summary contains a detailed description of the accident, road section, and weather. "
                                "The accident sketch draws the road scene and vehicle trajectory from a bird's-eye view."
                                "You should answer me consistent or not consistent."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Here's the analysis results: \n{dsl} \nHere's the crash sketch: \n"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{Sketch}"
                        }
                    },
                    {
                        "type": "text",
                        "text": f"\nHere's the crash summary: \n{Summary}\n Please help me verify whether the analysis results are consistent with the original data."
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
    file_name = f"{record}_dsl_validate.txt"
    file_path = os.path.join(results_path, file_name)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(model_output)
    # judge whether consistent
    if re.search(r'\bnot consistent\b', model_output, re.IGNORECASE):
        return 0
    else:
        return 1

def extract_conflict(record, model, results_path):
    """(#3) Extract an explicit collision descriptor so the BEV can place the crash
    deterministically instead of inferring it. Returns a dict (or {} if unparseable)."""
    Sketch = encode_image(f'./Crash_dataset/{record}/Sketch.jpg')
    with open(f'./Crash_dataset/{record}/Summary.txt', 'r', encoding='utf-8') as file:
        Summary = file.read()
    client = OpenAI()
    system = (
        "You are a crash reconstruction expert. From the crash summary and the bird's-eye "
        "sketch, identify the collision. Respond with ONLY a JSON object (no prose) of the form:\n"
        '{"at_fault_vehicle": "Vehicle_1" or "Vehicle_2", '
        '"struck_vehicle": "Vehicle_1" or "Vehicle_2", '
        '"impact_type": one of ["head-on","rear-end","angle","sideswipe","T-bone","single-vehicle"], '
        '"point_of_impact": "<short phrase>", "description": "<one sentence>"}\n'
        "at_fault_vehicle is the one that initiated the conflict (crossed the centreline, turned "
        "across the other's path, merged improperly, ran a control device, etc.). Vehicle_1 is the "
        "case vehicle (V1).")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [
                {"type": "text", "text": "Sketch:\n"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{Sketch}"}},
                {"type": "text", "text": f"\nSummary:\n{Summary}"}]}],
        response_format={"type": "text"}, temperature=1, max_completion_tokens=300,
        top_p=1, frequency_penalty=0, presence_penalty=0)
    out = response.choices[0].message.content
    with open(os.path.join(results_path, f"{record}_conflict.txt"), 'w', encoding='utf-8') as file:
        file.write(out)
    try:
        return json.loads(_extract_json_block(out))
    except Exception:
        return {}

def main():
    project_path = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='', type=str,
                        help='Path of crash dataset.')
    parser.add_argument('--prompts', default='', type=str)
    parser.add_argument('--meta_message', default='')
    parser.add_argument('--gpt', default='gpt-4o')
    args = parser.parse_args()

    # Create result folder
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_folder = f"./Experiment_results/DSL_results_{current_time}"
    os.makedirs(result_folder, exist_ok=True)
    folder_path = os.path.abspath(result_folder)

    ID_list = os.listdir(args.prompts)  # 105222, 109204, 109536, ...
    records = [str(ID) for ID in ID_list]

    with open(args.meta_message,'rb') as file:
        meta_data = pickle.load(file)
    meta_data = {sublist[-1]: sublist[:-1] for sublist in meta_data}

    DSL = []

    # DSL extraction
    for record in records:
        print(f'Work on case: {record}!')
        road_type = meta_data[record][0]
        direction = meta_data[record][-1]
        raw_msg, dsl = get_dsl(record, args.prompts, road_type, direction, args.gpt, folder_path)
        print('First round DSL extraction finished! --> go to self validation')
        time.sleep(1)
        # Self-validation
        vali_results = dsl_validation(raw_msg, record, args.gpt, folder_path)

        if vali_results == 1:
            print(f"Case {record} passed validation!")
            print(f"Case {record} finished!")
            print('-----------------')
            dsl['Scenario'] = record
            dsl['Conflict'] = extract_conflict(record, args.gpt, folder_path)
            DSL.append(dsl)
            time.sleep(1)
        elif vali_results == 0:
            print(f"Case {record} failed validation~")
            print('Extract DSL again!')
            raw_msg, dsl = get_dsl(record, args.prompts, road_type, direction, args.gpt, folder_path)
            dsl['Scenario'] = record
            dsl['Conflict'] = extract_conflict(record, args.gpt, folder_path)
            DSL.append(dsl)
            print(f"Case {record} finished!")
            print('-----------------')

    with open(os.path.join(folder_path, 'DSL_extraction_results.pkl'), 'wb') as f:
        pickle.dump(DSL, f)
    print(f"Results have been saved!")

if __name__ == '__main__':
    main()