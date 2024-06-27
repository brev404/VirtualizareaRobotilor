from projects.utilities.manage_yaml import write_to_yaml
def get_params(width, length, number_of_layers, box_width, box_length, box_height):
    data = {
        'width': width,
        'length': length,
        'number_of_layers': number_of_layers,
        'box_width': box_width,
        'box_length': box_length,
        'box_height': box_height
    }
    write_to_yaml(data, './resources/parameters.yaml', 'params')