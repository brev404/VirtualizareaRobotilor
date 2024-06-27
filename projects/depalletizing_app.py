import os
import cv2
import torch
import torchvision
import yaml
import supervision as sv
import numpy as np
import pyrealsense2
from projects.utilities.manage_yaml import write_to_yaml
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import pyrealsense2 as rs
import numpy as np
import cv2

import pyrealsense2 as rs
import numpy as np
import cv2

def getBoxesFromImage():
    HOME = os.getcwd()
    print("HOME:", HOME)

    CHECKPOINT_PATH = os.path.join(HOME, "projects\weights", "sam_vit_h_4b8939.pth")
    print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))

    #create data directory
    DATA_DIR = os.path.join(HOME, "projects\data")
    os.makedirs(DATA_DIR, exist_ok=True)

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_h"

    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(0))
    print(torchvision.__version__)
    print(torch.__version__)

    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)

    mask_generator = SamAutomaticMaskGenerator(sam)

    # Usage
    try:
        capture_photo('projects\data\poza.png')
        print("Photo captured successfully!")
    except RuntimeError as e:
        print(e)


    IMAGE_NAME = "poza.png"
    IMAGE_PATH = os.path.join(HOME, "projects\data", IMAGE_NAME)

    image_bgr = cv2.imread(IMAGE_PATH)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    print("We are generating the mask for the image, please wait...")
    sam_result = mask_generator.generate(image_rgb)
    print("Mask generated successfully!")

    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

    detections = sv.Detections.from_sam(sam_result=sam_result)

    annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

    corners = [(323, 142), (600, 142), (600, 446), (323, 446)]
    lista = {}
    # Iterate over the sam_result
    for mask in sam_result:
        # Get the mask
        segmentation = mask['segmentation']
        bounding_box = mask['bbox']
        bbox_area = bounding_box[2] * bounding_box[3]

        pixels = (mask['bbox'][0], mask['bbox'][1])

        is_in_corners = is_point_in_polygon(pixels, corners)
        area_condition = 3500 < mask['area'] < 5000
        bbox_area_condition = bbox_area < 10_000
        if is_in_corners:  # banc de lucru
            print(f"Bounding box: {bounding_box}, Bounding box area: {bbox_area}")
            print(f"Mask area: {mask['area']}")
            if area_condition and bbox_area_condition:
                center = find_center(bounding_box)
            # print(center, mask['area'])
            # Find the contours of the mask
                contours, _ = cv2.findContours(segmentation.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw the contours on the image
                cv2.drawContours(annotated_image, contours, -1, (0, 255, 0), 2)
                # cv2.rectangle(annotated_image, (int(bounding_box[0]), int(bounding_box[1])),
                #               (int(bounding_box[0] + bounding_box[2]), int(bounding_box[1] + bounding_box[3])), (0, 255, 0), 2)
                cv2.circle(annotated_image, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)

                # Calculate the angle of rotation
                rect = cv2.minAreaRect(contours[0])
                angle = rect[-1]
                width = int(rect[1][0])
                height = int(rect[1][1])

                if width < height:
                    angle = 90 - angle
                else:
                    angle = 180 - angle
                print(f"Angle of rotation: {angle} degrees")


                lista[int(angle)] = center

                cv2.putText(annotated_image, f'{int(angle)}', (int(bounding_box[0]), int(bounding_box[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    #Plot the images
    sv.plot_images_grid(
        images=[image_bgr, annotated_image],
        grid_size=(1, 2),
        titles=['source image', 'segmented image']
    )

    #Write the angles and the centers to the YAML file
    write_to_yaml(lista, './resources/parameters.yaml', 'rotated_boxes')

def capture_photo(filename='poza.png', auto_exposure=True):
    
    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    # 1280 x 720 color image
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Get the sensor and enable/disable auto-exposure
    sensor = profile.get_device().query_sensors()[1]
    sensor.set_option(rs.option.exposure, 100.000)
    #sensor.set_option(rs.option.enable_auto_exposure, int(auto_exposure))

    try:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("No color frame received")

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        # Save image
        cv2.imwrite(filename, color_image)

    finally:
        # Stop streaming
        pipeline.stop()

    # Check if the image was saved successfully
    if not os.path.isfile(filename):
        raise RuntimeError(f"Failed to capture photo and save it as {filename}")


def get_rotated_boxes(filename):
    # Read the YAML file
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)

    # Get the positions dictionary
    rotations_center = data.get('rotated_boxes', {})

    # Create a dictionary to store the results
    results = {}

    # Convert the string values to lists of floats
    for key in rotations_center:
        angle = int(key)
        coordinates = rotations_center[key]

        # Store the angle and coordinates in the results dictionary
        results[angle] = coordinates

    # Return the results dictionary
    return results

def is_point_in_polygon(point, polygon):
    from shapely.geometry import Point, Polygon
    point = Point(point)
    polygon = Polygon(polygon)
    return polygon.contains(point)


def find_center(bbox):
    x, y, w, h = bbox
    center_x = x + w / 2
    center_y = y + h / 2
    return [center_x, center_y]


# print(lista)

# results = get_rotated_boxes('../resources/parameters.yaml')

# # Print the returned values
# print(results)
