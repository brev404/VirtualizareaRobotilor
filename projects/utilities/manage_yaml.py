import yaml
from interfaces.robot_interface import Robot

import os


def write_to_yaml(data, filename, data_type):
    # Check if directory exists, if not create it
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Check if file exists
    if os.path.exists(filename):
        # Load existing data
        with open(filename, 'r') as file:
            existing_data = yaml.safe_load(file) or {}
    else:
        # Create new file with empty data
        existing_data = {}

    # Check if 'positions' and 'params' keys exist, if not create them
    if 'positions' not in existing_data:
        existing_data['positions'] = {}
    if 'params' not in existing_data:
        existing_data['params'] = {}
    if 'rotated_boxes' not in existing_data:
        existing_data['rotated_boxes'] = {}

    # Convert coordinates to single line if data_type is 'position'
    if data_type == 'position':
        for key in data:
            data[key] = ', '.join(map(str, data[key]))
        existing_data['positions'].update(data)
    elif data_type == 'params':
        existing_data['params'].update(data)
    elif data_type == 'rotated_boxes':
        for key in data:
            data[key] = ', '.join(map(str, data[key]))
        existing_data['rotated_boxes'] = data

    # Write updated data back to the file
    with open(filename, 'w') as file:
        yaml.dump(existing_data, file)
