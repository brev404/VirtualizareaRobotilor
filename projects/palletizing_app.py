import csv
import json
import math
import time
from depalletizing_app import get_rotated_boxes
import yaml
import pandas as pd
from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap

from robots.UR5.urx import robotiq_two_finger_gripper as gripper
from interfaces.robot_interface import Robot
from matplotlib import pyplot as plt
import numpy as np

from robots.UR5.urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper


#######################################################
def solve_pack(palletWidth, palletLength, boxWidth, boxLength, thresholdPercent=0.00):
    pack = {}

    boxW = boxWidth * (1 + thresholdPercent)
    boxL = boxLength * (1 + thresholdPercent)

    maxNoBoxes = 0

    for coefWW in range(0, int(palletWidth / boxW) + 1):
        maxNoBoxesZoneOne = 0

        ansZoneOne = {
            'WW': 0,
            'LL': 0,
            'LW': 0
        }

        if coefWW:  # Zone One exists
            coefLW = 0
            coefLL = 1

            maxLW = (palletLength - coefLL * boxL - coefLW * boxW) / boxW
            for coefLW in range(0, int(maxLW) + 1):
                coefLL = 1

                coefLL += int((palletLength - coefLL * boxL - coefLW * boxW) / boxL)

                noBoxes = coefLL * coefWW
                noBoxes += coefLW * int(coefWW * boxW / boxL)

                if noBoxes > maxNoBoxesZoneOne:
                    maxNoBoxesZoneOne = noBoxes
                    ansZoneOne = {
                        'WW': coefWW,
                        'LL': coefLL,
                        'LW': coefLW
                    }

        coefWL = int((palletWidth - coefWW * boxW) / boxL)
        maxNoBoxesZoneTwo = 0
        ansZoneTwo = {
            'WL': 0,
            'LL': 0,
            'LW': 0
        }

        if coefWL:  # Zone Two exists
            coefLW = 1
            coefLL = 0

            maxLW = int((palletLength - coefLL * boxL) / boxW)
            while coefLW <= maxLW:
                coefLL = 0
                coefLL = int((palletLength - coefLW * boxW) / boxL)

                noBoxes = coefLW * coefWL
                noBoxes += coefLL * int(coefWL * boxL / boxW)

                if noBoxes > maxNoBoxesZoneTwo:
                    maxNoBoxesZoneTwo = noBoxes
                    ansZoneTwo = {
                        'WL': coefWL,
                        'LL': coefLL,
                        'LW': coefLW
                    }

                coefLW += 1

        if maxNoBoxesZoneOne + maxNoBoxesZoneTwo > maxNoBoxes:
            maxNoBoxes = maxNoBoxesZoneOne + maxNoBoxesZoneTwo
            pack = {
                'ZoneOne': ansZoneOne,
                'ZoneTwo': ansZoneTwo,
                'noBoxesZoneOne': maxNoBoxesZoneOne,
                'noBoxesZoneTwo': maxNoBoxesZoneTwo
            }

    return pack

def generate_box_data(packing_solution, boxWidth, boxLength, palletWidth, palletLength):
    box_data = []
    box_id = 1

    # Process Zone One
    for orientation, count in packing_solution['ZoneOne'].items():
        for i in range(count):
            if orientation == 'WW':
                x = i * boxWidth
                y = 0
                box_orientation = 'widthwise'
            elif orientation == 'LL':
                x = 0
                y = i * boxLength
                box_orientation = 'lengthwise'
            else:  # orientation == 'LW'
                x = 0
                y = i * boxWidth
                box_orientation = 'lengthwise'

            box_data.append({
                'id': box_id,
                'x': x,
                'y': y,
                'orientation': box_orientation
            })

            box_id += 1

    # Process Zone Two
    for orientation, count in packing_solution['ZoneTwo'].items():
        for i in range(count):
            if orientation == 'WL':
                x = i * boxLength
                y = packing_solution['ZoneOne']['WW'] * boxWidth  # Start after Zone One
                box_orientation = 'widthwise'
            else:  # orientation == 'LL' or 'LW'
                x = packing_solution['ZoneOne']['LL'] * boxLength  # Start after Zone One
                y = i * boxLength if orientation == 'LL' else i * boxWidth
                box_orientation = 'lengthwise'

            box_data.append({
                'id': box_id,
                'x': x,
                'y': y,
                'orientation': box_orientation
            })

            box_id += 1

    return box_data

def plot_pallet(palletWidth, palletLength, boxWidth, boxLength, packedBoxes, centroids, thresholdPercent=0.00):
    packedBoxesZoneOne = packedBoxes['ZoneOne']
    packedBoxesZoneTwo = packedBoxes['ZoneTwo']

    boxW = boxWidth * (1 + thresholdPercent)
    boxL = boxLength * (1 + thresholdPercent)

    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    if boxW > boxL:
        boxW, boxL = boxL, boxW

    # Plot boxes in Zone One
    # Plot boxes on width
    for i in range(packedBoxesZoneOne['WW']):
        for j in range(packedBoxesZoneOne['LL']):
            ax.add_patch(
                plt.Rectangle(xy=(i * boxW, j * boxL), width=boxW, height=boxL, facecolor='orange',
                              edgecolor='white', linewidth=2))

    # Plot boxes on length
    offsetY = packedBoxesZoneOne['LL'] * boxL
    maxB = packedBoxesZoneOne['WW'] * boxW / boxL
    for i in range(int(maxB)):
        for j in range(packedBoxesZoneOne['LW']):
            ax.add_patch(plt.Rectangle(xy=(i * boxL, offsetY + j * boxW), width=boxL, height=boxW,
                                       facecolor='blue', edgecolor='white', linewidth=2))

    # Plot boxes in Zone Two
    # Plot boxes on width
    offsetX = packedBoxesZoneOne['WW'] * boxW
    for i in range(packedBoxesZoneTwo['WL']):
        for j in range(packedBoxesZoneTwo['LW']):
            ax.add_patch(plt.Rectangle(xy=(offsetX + i * boxL, j * boxW), width=boxL, height=boxW,
                                       facecolor='purple', edgecolor='white', linewidth=2))

    offsetY = packedBoxesZoneTwo['LW'] * boxW
    # Plot boxes on length
    maxB = packedBoxesZoneTwo['WL'] * boxL / boxW
    for i in range(int(maxB)):
        for j in range(packedBoxesZoneTwo['LL']):
            ax.add_patch(
                plt.Rectangle(xy=(offsetX + i * boxW, offsetY + j * boxL), width=boxW, height=boxL,
                              facecolor='red', edgecolor='white', linewidth=2))

    x_val = [centroid[0] for centroid in centroids]
    y_val = [centroid[1] for centroid in centroids]

    ax.scatter(x_val, y_val, color='black')

    ax.set_xlim(0, palletWidth)
    ax.set_ylim(0, palletLength)

    plt.show()
    fig.savefig('./outputs/pallet.png')


def get_centroids(boxWidth, boxLength, packedBoxes, thresholdPercent=0.00):
    boxW = boxWidth * (1 + thresholdPercent)
    boxL = boxLength * (1 + thresholdPercent)

    if boxW > boxL:
        boxW, boxL = boxL, boxW

    packedBoxesZoneOne = packedBoxes['ZoneOne']
    packedBoxesZoneTwo = packedBoxes['ZoneTwo']
    centers = []

    # Place boxes in Zone One
    # Place boxes on width
    for i in range(packedBoxesZoneOne['WW']):
        for j in range(packedBoxesZoneOne['LL']):
            centers.append((i * boxW + boxW / 2, j * boxL + boxL / 2, False))

    # Plot boxes on length
    offsetY = packedBoxesZoneOne['LL'] * boxL
    maxB = packedBoxesZoneOne['WW'] * boxW / boxL
    for i in range(int(maxB)):
        for j in range(packedBoxesZoneOne['LW']):
            centers.append((i * boxL + boxL / 2, offsetY + j * boxW + boxW / 2, True))

    # Place boxes in Zone Two
    # Place boxes on width
    offsetX = packedBoxesZoneOne['WW'] * boxW
    for i in range(packedBoxesZoneTwo['WL']):
        for j in range(packedBoxesZoneTwo['LW']):
            centers.append((offsetX + i * boxL + boxL / 2, j * boxW + boxW / 2, True))

    offsetY = packedBoxesZoneTwo['LW'] * boxW
    # Place boxes on length
    maxB = packedBoxesZoneTwo['WL'] * boxL / boxW
    for i in range(int(maxB)):
        for j in range(packedBoxesZoneTwo['LL']):
            centers.append((offsetX + i * boxW + boxW / 2, offsetY + j * boxL + boxL / 2, False))

    return np.asarray(centers)


def plot_centroids(palletWidth, palletLength, centroids):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    x_val = [centroid[0] for centroid in centroids]
    y_val = [centroid[1] for centroid in centroids]

    plt.scatter(x_val, y_val)
    ax.set_xlim(0, palletWidth)
    ax.set_ylim(0, palletLength)
    ax.grid(True)

    plt.show()


def check_dim(width, length):
    if width > length:
        width, length = length, width

    return width, length


def get_all_positions(filename):
    """
    This method reads the YAML file and returns all position parameters as a list.

    Returns:
        list: A list containing the values of all position parameters.
    """
    # Read the YAML file
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)

    # Get the positions dictionary
    positions = data.get('positions', {})

    # Convert the string values to lists of floats
    for key in positions:
        positions[key] = list(map(float, positions[key].split(',')))

    # Return the values of all position parameters as a list
    return [positions.get('home', None), positions.get('homeJ', None), positions.get('origin', None),
            positions.get('originJ', None), positions.get('approach', None), positions.get('approachJ', None),
            positions.get('pickup', None), positions.get('pickupJ', None)]


def get_all_params(filename):
    """
    This method reads the YAML file and returns all box parameters as a list.

    Returns:
        list: A list containing the values of all box parameters.
    """
    # Read the YAML file
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)

    # Get the params dictionary
    params = data.get('params', {})

    # Return the values of all box parameters as a list
    return [float(params.get('width', None)), float(params.get('length', None)),
            int(params.get('number_of_layers', None)), float(params.get('box_width', None)),
            float(params.get('box_length', None)), float(params.get('box_height', None))]


def print_img(imageLabel):
    image_path = "../outputs/pallet.png"
    pixmap = QPixmap(image_path)
    pixmap = pixmap.scaled(500, 500, QtCore.Qt.KeepAspectRatio)
    imageLabel.setPixmap(pixmap)
    imageLabel.setScaledContents(True)


def createEmptyFigure(palletWidth, palletLength):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(0, palletWidth)
    ax.set_ylim(0, palletLength)
    return fig, ax


def addBoxToFigure(ax, centroid, isRotated, boxW, boxL):
    x, y = centroid
    if isRotated:
        ax.add_patch(
            plt.Rectangle(xy=(x - boxL / 2, y - boxW / 2), width=boxL, height=boxW, facecolor='red', edgecolor='white',
                          linewidth=2))
    else:
        ax.add_patch(
            plt.Rectangle(xy=(x - boxW / 2, y - boxL / 2), width=boxW, height=boxL, facecolor='green',
                          edgecolor='white',
                          linewidth=2))

def rearrange_solution(solution, box_width, box_length, pallet_width, pallet_length):
    number_of_boxes = len(solution)
    total_box_width = number_of_boxes * box_width
    total_box_length = number_of_boxes * box_length

    total_gap_width = pallet_width - total_box_width
    total_gap_length = pallet_length - total_box_length

    gap_width = total_gap_width / (number_of_boxes + 1)
    gap_length = total_gap_length / (number_of_boxes + 1)

    half_gap_width = gap_width / 2
    half_gap_length = gap_length / 2

    new_solution1 = []
    new_solution2 = []

    for i in range(number_of_boxes):
        new_center1 = [(i + 1) * (box_width + gap_width) - box_width / 2,
                       (i + 1) * (box_length + gap_length) - box_length / 2]
        new_center2 = [new_center1[0] + half_gap_width, new_center1[1] + half_gap_length]

        new_solution1.append(new_center1)
        new_solution2.append(new_center2)

    packedBoxes = {
        'ZoneOne': {'WL': len(new_solution1), 'LL': 0, 'LW': 0},
        'ZoneTwo': {'WL': len(new_solution2), 'LL': 0, 'LW': 0},
    }

    return packedBoxes, new_solution1, new_solution2

def showFigure(fig):
    #plt.show()
    #fig.savefig("./outputs/pallet" + str(index) + ".png")
    #image_path = "./outputs/pallet" + str(index) + ".png"
    print("saving figure")
    fig.savefig("./outputs/pallet.png")
    # try:
    #     mainWindow.updateImageGraphics(image_path)
    # except Exception as e:
    #     print(e)

    # mainWindow.updateImageGraphics(image_path)
    # pixmap = QPixmap(image_path)
    # pixmap = pixmap.scaled(500, 500, QtCore.Qt.KeepAspectRatio)
    # imageLabel.setPixmap(pixmap)
    # imageLabel.setScaledContents(True)

def main():
    palletLength, palletWidth, numLayers, boxWidth, boxLength, boxHeight = 1200, 800, 3, 200, 300, 200
    boxWidth, boxLength = check_dim(boxWidth, boxLength)
    palletWidth, palletLength = check_dim(palletWidth, palletLength)

    result = solve_pack(palletWidth, palletLength, boxWidth, boxLength)
    print(json.dumps(result, indent=4))

    fig, ax = createEmptyFigure(palletWidth, palletLength)

    centroids = get_centroids(boxWidth, boxLength, result)
    #plot_pallet(palletWidth, palletLength, boxLength, boxWidth, result, centroids, thresholdPercent)

    packed, sol1, sol2 = rearrange_solution(centroids, boxWidth, boxLength, palletWidth, palletLength)
    print(json.dumps(packed, indent=4))
    plot_pallet(palletWidth, palletLength, boxLength, boxWidth, packed, sol1)
    

def trial(): # to be changed back to main
    print("Starting palletizing app from main.py...")
    
    robot = Robot('UR5', 'test')
    thresholdPercent = 0.12
    # Opening the gripper
    robot.gripper_action(80)

    homel, homej, originl, originj, approach, approachj, pickUp, pickUpJ = get_all_positions(
        './resources/parameters.yaml')
    palletLength, palletWidth, numLayers, boxWidth, boxLength, boxHeight = get_all_params(
        './resources/parameters.yaml')
    pickupPoints = get_rotated_boxes('./resources/parameters.yaml')  # value[0], value[1], key

    # Convert the dictionary to a list of tuples
    pickupPointsList = [(k, *map(float, v.split(', '))) for k, v in pickupPoints.items()]
    
    boxWidth, boxLength = check_dim(boxWidth, boxLength)
    palletWidth, palletLength = check_dim(palletWidth, palletLength)

    result = solve_pack(palletWidth, palletLength, boxWidth, boxLength, thresholdPercent)
    print(json.dumps(result, indent=4))

    fig, ax = createEmptyFigure(palletWidth, palletLength)

    centroids = get_centroids(boxWidth, boxLength, result, thresholdPercent)
    #plot_pallet(palletWidth, palletLength, boxLength, boxWidth, result, centroids, thresholdPercent)

    packed, sol1, sol2 = rearrange_solution(centroids, boxWidth, boxLength, palletWidth, palletLength)
    plot_pallet(palletWidth, palletLength, boxLength, boxWidth, packed, sol1)

    boxesNum = len(centroids)
    centroids = np.asarray(centroids)
    dest = originl * boxesNum
    dest = np.asarray(dest).reshape((boxesNum, 6))

    dest[:, :2] = (originl[:2]) + centroids[:, :2]
    # move to home and update points
    robot.move_joints(homej, velocity=250, acceleration=100)

    pickupHeight = pickUp[2]
    fallDistance = approach[2]
    currentHeight = 0

    approachListJ = approach * (numLayers + 1)
    approachListL = approach * (numLayers + 1)
    approachListL[0] = approach
    robot.move_joints(approachj, velocity=250, acceleration=100)
    # save approach points in j

    for layer in range(numLayers):
        approach[2] = approach[2] + boxHeight
        robot.move_to_coords(approach, velocity=250, acceleration=100)
        approachListL[layer + 1] = robot.get_coords()
        approachListJ[layer + 1] = robot.get_joints()

    index = 0
    
    robot.move_joints(homej, velocity=250, acceleration=100)

    for layer in range(numLayers):
        contor_centroid = 0
        # parcurgem coordonatele cutiilor
        for d in dest:
            addBoxToFigure(ax, centroids[contor_centroid, :2], centroids[contor_centroid, 2], boxWidth, boxLength)
            showFigure(fig)
            #mainWindow.update()
            print(pickupPointsList)
            if index > len(pickupPointsList) - 1:
                robot.move_joints([0, -90, 90, -90, -90, 0], velocity=250, acceleration=100)
                return
            
            # go to safe pose above boxes
            robot.move_joints([0, -90, 90, -90, -90, 0], velocity=250, acceleration=100)
            
            targetX, targetY, targetAngle = robot.convertCameraToRobot(pickupPointsList[index][1], pickupPointsList[index][2], pickupPointsList[index][0])
            
            pickupPose = [targetX, targetY, -200, 180, 0, 0]
            
            # go above the next box to be pickup
            robot.move_to_coords(pickupPose, velocity=250, acceleration=100)
            
            jPosePickup = robot.get_joints()
            jPosePickup[5] = jPosePickup[0] + targetAngle
            
            # match the angle of the box
            robot.move_joints(jPosePickup, velocity=250, acceleration=100)
            
            wPose = robot.get_coords()
            wPose[2] = wPose[2] - 220
            
            # go down to pickup the box
            robot.move_to_coords(wPose, velocity=250, acceleration=100)
            Zdrop = wPose[2]
            
            # close gripper - get the box
            robot.gripper_action(125)
            
            wPose[2] = wPose[2] + 220
            
            # go up with the box
            robot.move_to_coords(wPose, velocity=250, acceleration=100)
            
            # move to approach
            robot.move_joints(approachListJ[layer + 1], velocity=250, acceleration=100)

            #print("Moved to approach[layer + 1]")

            d[2] = approachListL[layer + 1][2]
            print(d)
            currentj = []
            # only rotate here
            if centroids[contor_centroid, 2] == 1:
                currentj = robot.get_joints()
                currentj[5] = currentj[0]
                currentj[5] += math.degrees(1.5708)
                robot.move_joints(currentj, velocity=250, acceleration=100)
                currentl = robot.get_coords()
                print(d)
                d[3:6] = currentl[3:6]
                print(d)
            # print("Dest and origin")
            # print(dest)
            # print(originl)

            robot.move_to_coords(d, velocity=250, acceleration=100)
            d[2] = Zdrop + 10 + (boxHeight - 20)* (layer)
            robot.move_to_coords(d, acceleration=250, velocity=100)
            robot.gripper_action(80)
            d[2] += 210
            robot.move_to_coords(d, velocity=250, acceleration=100)
            contor_centroid += 1
            
            index += 1

    robot.move_joints(homej, velocity=50, acceleration=50)

    robot.disconnect()

def previous():
    print("Starting palletizing app from main.py...")
    
    robot = Robot('UR5', 'test')

    robot.gripper_open()
    robot.gripper_close()
    robot.gripper_open()
    robot.gripper_close()

    homel, homej, originl, originj, approach, approachj, pickUp, pickUpJ = get_all_positions(
        './resources/parameters.yaml')
    palletLength, palletWidth, numLayers, boxWidth, boxLength, boxHeight = get_all_params(
        './resources/parameters.yaml')
    
    #pickup points
    

    boxWidth, boxLength = check_dim(boxWidth, boxLength)
    palletWidth, palletLength = check_dim(palletWidth, palletLength)

    result = solve_pack(palletWidth, palletLength, boxWidth, boxLength)
    print(json.dumps(result, indent=4))

    fig, ax = createEmptyFigure(palletWidth, palletLength)

    centroids = get_centroids(boxWidth, boxLength, result)
    plot_pallet(palletWidth, palletLength, boxLength, boxWidth, result, centroids)

    boxesNum = len(centroids)
    centroids = np.asarray(centroids)
    # print(boxesNum)
    # print(originl)
    dest = originl * boxesNum
    # print(dest)
    dest = np.asarray(dest).reshape((boxesNum, 6))

    dest[:, :2] = (originl[:2]) + centroids[:, :2]
    # print(originl)
    # print(dest)
    # move to home and update points
    robot.move_joints(homej, velocity=50, acceleration=50)

    pickupHeight = pickUp[2]
    fallDistance = approach[2]
    currentHeight = 0

    approachListJ = approach * (numLayers + 1)
    approachListL = approach * (numLayers + 1)
    approachListL[0] = approach
    robot.move_joints(approachj, velocity=50, acceleration=50)
    # save approach points in j

    for layer in range(numLayers):
        approach[2] = approach[2] + boxHeight
        robot.move_to_coords(approach, velocity=50, acceleration=50)
        approachListL[layer + 1] = robot.get_coords()
        approachListJ[layer + 1] = robot.get_joints()

    index = 0

    for layer in range(numLayers):
        contor_centroid = 0
        # parcurgem coordonatele cutiilor
        for d in dest:
            addBoxToFigure(ax, centroids[contor_centroid, :2], centroids[contor_centroid, 2], boxWidth, boxLength)
            showFigure(fig)
            #mainWindow.update()
            index += 1
            # move to get boxes from pickup
            robot.move_joints(pickUpJ, velocity=50, acceleration=50)

            current = robot.get_coords()
            current[2] = current[2] + (-410 - current[2])  # -410
            robot.move_to_coords(current, velocity=50, acceleration=50)

            # print("Go down")

            # close gripper
            # gripper.gripper_action(120)

            pickUp[2] = approachListL[layer + 1][2]
            # box in gripper, get to z of approach
            robot.move_to_coords(pickUp, velocity=50, acceleration=50)

            # move to approach
            robot.move_joints(homej, velocity=50, acceleration=50)
            robot.move_joints(approachListJ[layer + 1], velocity=50, acceleration=50)

            print("Moved to approach[layer + 1]")

            d[2] = approachListL[layer + 1][2]
            print(d)
            currentj = []
            # only rotate here
            if centroids[contor_centroid, 2] == 1:
                currentj = robot.get_joints()
                currentj[5] += math.degrees(1.5708)
                robot.move_joints(currentj, velocity=50, acceleration=50)
                currentl = robot.get_coords()
                print(d)
                d[3:6] = currentl[3:6]
                print(d)
            # print("Dest and origin")
            # print(dest)
            # print(originl)

            robot.move_to_coords(d, velocity=50, acceleration=50)
            d[2] -= 81
            robot.move_to_coords(d, acceleration=50, velocity=50)
            # gripper.gripper_action(100)
            d[2] += 81
            robot.move_to_coords(d, velocity=50, acceleration=50)
            contor_centroid += 1

    robot.move_joints(homej, velocity=50, acceleration=50)

    robot.disconnect()
    
if __name__ == "__main__":
    main()  