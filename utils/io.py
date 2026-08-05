"""
input/output ops
"""

import json
import os
import yaml


def load_file(file_path):
    """
    :param file_path: .json, path of file
    :return: dict/list, file
    """
    # load file
    file_path = os.path.join(file_path)
    print(f"Parsing {file_path}")
    with open(file_path, 'r') as f:
        file_json = json.load(f)
    return file_json

def dict_to_yaml(data, file_path):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            yaml.dump(data, file, allow_unicode=True)
        print(f"yaml file is saved to: {file_path}")
    except Exception as e:
        print(f"save yaml file error: {e}")

def calu_time_cost(root_path):
    """
    debug time cost
    :param root_path: the root path of time file
    """
    for exp_name in os.listdir(root_path):
        exp_path = os.path.join(root_path, exp_name)
        print(f'----------------{exp_name}-----------------')
        for file_name in os.listdir(exp_path):
            file_path, frame_time = os.path.join(exp_path, file_name), []
            with open(file_path, 'r') as f:
                time_cost = json.load(f)
            frame_time = np.array([time for _, time in time_cost.items()])
            print(f'Module {file_name.split(".")[0]} spend time: {np.mean(frame_time) * 1000} ms')
