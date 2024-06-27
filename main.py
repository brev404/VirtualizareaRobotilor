import json
from matplotlib import pyplot as plt
import numpy as np
import torch, torchvision
from interfaces.robot_interface import Robot
import yaml
from depalletizing_app import getBoxesFromImage
from palletizing_app import solve_pack, plot_pallet, get_centroids, check_dim

from PyQt5.QtCore import QThread

#   PEPSI 10l - 14 CUTII - forma diferita
thrs = 0.0
boxWidth, boxLength, palletWidth, palletLength = 200, 280, 800, 1200

# Pepsi 5l - 26 CUTII - aceeasi forma
# thrs = 0.0
# boxWidth, boxLength, palletWidth, palletLength = 154, 217, 800, 1200

# # Magura glazurata - 24 buc - 21 CUTII - forma diferita
# thrs = 0.0
# boxWidth, boxLength, palletWidth, palletLength = 190, 230, 800, 1200

# # Magura robot var 2
# thrs = 0.0
# boxWidth, boxLength, palletWidth, palletLength = 202, 231, 800, 1200

# # Mgura rulada + 3 cereala 24 buc - 20 CUTII aceeasi forma
# thrs = 0.0
# boxWidth, boxLength, palletWidth, palletLength = 200, 230, 800, 1200

# # Tymbark 200ml - 16 cutii
# thrs = 0.0
# boxWidth, boxLength, palletWidth, palletLength = 160, 361, 800, 1202

# # Tymbark 200ml - 17 cutii
# thrs = 0.0
# boxWidth, boxLength, palletWidth, palletLength = 160, 335, 800, 1200

# # Ursus 330 - 5 cutii
# thrs = 0.0
# boxWidth, boxLength, palletWidth, palletLength = 240, 365, 605, 800

# # Ursus 500 - 4 cutii
# thrs = 0.0
# boxWidth, boxLength, palletWidth, palletLength = 269, 406, 600, 812

### BOXES

boxWidth, boxLength = check_dim(boxWidth, boxLength)
palletWidth, palletLength = check_dim(palletWidth, palletLength)

result = solve_pack(palletWidth, palletLength, boxWidth, boxLength, thresholdPercent=thrs)
print(json.dumps(result, indent=4))

centroids = get_centroids(boxWidth, boxLength, result, thresholdPercent=thrs)
plot_pallet(palletWidth, palletLength, boxLength, boxWidth, result, centroids, thresholdPercent=thrs)

#robot = Robot('UR5', 'data.xlsx')
#robot.move_joints([0, -90, 90, -90, -90, 0], velocity=250, acceleration=100)
# lista = get_rotated_boxes('./resources/parameters.yaml')

# print(lista)

# robot.move_joints([0, 0, 0, 0, 0, 0], velocity=250, acceleration=100)

# robot.move_joints([0, -90, 90, 45, 0, 0], velocity=250, acceleration=100)

# robot.move_joints([0, -90, 90, 45, -60, 0], velocity=250, acceleration=100)

# for key, value in lista.items():
#     goToCamera(value[0], value[1], key)

# robot = Robot('UR5', 'data.xlsx')

# class GripperOpenThread(QThread):
#     def run(self):
#         robot.gripper_open()
        
# class GripperCloseThread(QThread):  
#     def run(self):
#         robot.gripper_close()

# gripper_open_thread = GripperOpenThread()
# gripper_close_thread = GripperCloseThread()

# robot.gripper_close()
# robot.move_joints([0, -90, 90, 0, 0, 0], velocity=250, acceleration=100)
# gripper_open_thread.start()

# if gripper_open_thread.isFinished():
#     print("Gripper open thread has finished.")
# else:
#     print("Gripper open thread is still running.")
    
# robot.move_joints([90, -90, 90, 0, 0, 0], velocity=250, acceleration=100)