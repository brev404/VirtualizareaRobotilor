import yaml
from interfaces.robot_interface import Robot
from projects.utilities.manage_yaml import write_to_yaml


def getApproach():
    robot = Robot('UR5', 'data.xlsx')
    approach = robot.get_coords()
    approachJ = robot.get_joints()
    data = {'approach': approach, 'approachJ': approachJ}
    write_to_yaml(data, 'resources/parameters.yaml', 'position')


def getPickup():
    robot = Robot('UR5', 'data.xlsx')
    pickup = robot.get_coords()
    pickupJ = robot.get_joints()
    data = {'pickup': pickup, 'pickupJ': pickupJ}
    write_to_yaml(data, 'resources/parameters.yaml', 'position')


def getOrigin():
    robot = Robot('UR5', 'data.xlsx')
    origin = robot.get_coords()
    originJ = robot.get_joints()
    data = {'origin': origin, 'originJ': originJ}
    write_to_yaml(data, 'resources/parameters.yaml', 'position')


def getHome():
    robot = Robot('UR5', 'data.xlsx')
    home = robot.get_coords()
    homeJ = robot.get_joints()
    data = {'home': home, 'homeJ': homeJ}
    write_to_yaml(data, 'resources/parameters.yaml', 'position')
