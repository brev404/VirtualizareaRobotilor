import math
from PIL import Image
import torch
import torchvision
import argparse
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import pyrealsense2 as rs
from matplotlib import pyplot as plt
from interfaces.robot_interface import Robot
import interfaces.camera_interface as camera_interface

# robot = Robot('CRX10', 'data.xlsx')
# robot.connect()


coco_names = [
    '__background__','person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

np.random.seed(42)

# CREATE DIFFERENT COLORS FOR EACH CLASS
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

# DEFINE THE TORCHVISION IMAGE TRANSFORM
transform = transforms.Compose([
    transforms.ToTensor(),
])

def get_model(device='cpu', model_name='v2'):
    # Load the model.
    if model_name == 'v2':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights='DEFAULT'
        )
    elif model_name == 'v1':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights='DEFAULT'
        )
    # Load the model onto the computation device.
    model = model.eval().to(device)
    return model

# path_img = 'D://Desktop//FRCNN//test//image.jpeg'

def predict(image, model, device, detection_threshold):
    """
    Predict the output of an image after forward pass through
    the model and return the bounding boxes, class names, and
    class labels.
    """
    # Transform the image to tensor.
    image = transform(image).to(device)
    # Add a batch dimension.
    image = image.unsqueeze(0)
    # Get the predictions on the image.
    with torch.no_grad():
        outputs = model(image)

    # Get score for all the predicted objects.
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()

    # Get all the predicted bounding boxes.
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    # Get boxes above the threshold score.
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
    labels = outputs[0]['labels'][:len(boxes)]
    # Get all the predicited class names.
    pred_classes = [coco_names[i] for i in labels.cpu().numpy()]

    return boxes, pred_classes, labels

def draw_boxes(boxes, classes, labels, image):
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)  # Line width.
    tf = max(lw - 1, 1) # Font thickness.

    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            img=image,
            pt1=(int(box[0]), int(box[1])),
            pt2=(int(box[2]), int(box[3])),
            color=color[::-1],
            thickness=lw
        )
        cv2.putText(
            img=image,
            text=classes[i],
            org=(int(box[0]), int(box[1]-5)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=lw / 3,
            color=color[::-1],
            thickness=tf,
            lineType=cv2.LINE_AA
        )
    return image

def calculate_center(boxes, classes, depths):
    # ROTATION MATRIX
    a = -0.0199686
    b = 0.464328
    c = 0.973301
    d = -0.3562
    tx = -586.241
    ty = 5.74667

    # AN EMPTY LIST
    center_coordinates = []

    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = box

        # CALCULATE THE CENTER COORDINATES
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        # APPLY THE ROTATION MATRIX
        center_new_x = a * center_x + b * center_y + tx
        center_new_y = c * center_x + d * center_y + ty

        # ADD CENTER COORDINATES
        center_coordinates.append([center_new_x, center_new_y, depths.pop(0)])

    # AN EMPTY MAP
    result_map = {}

    # ADD THE RESULTS IN THE MAP LIST
    if len(classes) == len(center_coordinates):
        for class_, center in zip(classes, center_coordinates):
            if class_ not in result_map:
                result_map[class_] = []
            result_map[class_].append(center)
    else:
        print("Listele nu au aceeași lungime!")

    return result_map

def detect(color_image, depth_data, depth_scale, depth_intrinsics, threshold=0.7, model_type='v2'):
    # DEFINE THE COMPUTATION DEVICE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(device, model_type)

    # # CONVERT COLOR IMAGE TO PIL IMAGE
    # color_pil_image = Image.fromarray(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

    # DETECT OUTPUTS
    with torch.no_grad():
        boxes, classes, labels = predict(color_image, model, device, threshold)
    # EXTRACT DEPTH FOR DETECTED OBJECTS
    depths = []
    for box in boxes:
        ymin, xmin, ymax, xmax = box
        center_x = int((xmin + xmax) / 2)
        center_y = int((ymin + ymax) / 2)


        # Ensure the calculated coordinates are within the valid range for the color image
        if 0 <= center_y < color_image.shape[0] and 0 <= center_x < color_image.shape[1]:
            # Use color image coordinates to retrieve depth value
            cropped_depth = np.asanyarray(depth_data.get_data())
            cropped_depth = cropped_depth[xmin:xmax, ymin:ymax].astype(float)
            cropped_depth = (cropped_depth * depth_scale) * 1000  # Convert to millimeters
            dist, _, _, _ = cv2.mean(cropped_depth)
            depths.append(dist)
        else:
            print(f"Warning: Coordinates ({center_x}, {center_y}) out of bounds for color image, appended 0.0.")
            depths.append(0.0)
        # contor +=1
    # DRAW BOUNDING BOXES AND SHOW THEM
    result = calculate_center(boxes, classes, depths)  # THE FINAL RESULT
    transformed_result = camera_interface.transform_coordinates(result, depth_intrinsics, depth_scale)



    # image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    image = draw_boxes(boxes, classes, labels, color_image)
    save_name = f"sorting_output_{str(threshold).replace('.', '_')}_{model_type}"
    # cv2.imshow('Image', image)
    plt.imshow(image)
    plt.show()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"../outputs/{save_name}.jpg", image)
    # cv2.waitKey(0)

    return transformed_result

pipline, profile = camera_interface.init_camera()
color_image, depth_frame, depth_scale, depth_intrinsics = camera_interface.capture_aligned_images(pipline, profile)
result_list = detect(color_image, depth_frame, depth_scale, depth_intrinsics)
print(result_list)


# def play ():
#     robot = Robot('UR5', 'data.xlsx')
#
#     home = [150.0, -443.0, 301.0, math.degrees(2.380), math.degrees(2.098), math.degrees(-0.112)]
#     print(home)
#     robot.move_to_coords(home, velocity=50, acceleration=50)
#
#     extendes_list = [179.9, 0, 10]
#     for key in result_list:
#         if key == 'cell phone' or key == 'bottle' or key == 'spoon':
#             for index in result_list[key]:
#                 centroids = index
#                 centroids.extend(extendes_list)
#                 centroids[2] = -188
#                 centroids_current_object = centroids.copy()
#                 robot.move_to_coords(centroids_current_object, velocity=50, acceleration=50)
#                 # robot.set_dout(107, True)
#                 robot.move_to_coords([578.0, 103.00, -187, math.degrees(2.188), math.degrees(2.162), math.degrees(0.063)], velocity=50, acceleration=50) # change by area
#                 # robot.move_to_coords([1011.974, 533.286, -470, 179, 0, 0]) # change by area
#                 # robot.set_dout(107, False)
#     robot.move_to_coords(home, velocity=50, acceleration=50)

