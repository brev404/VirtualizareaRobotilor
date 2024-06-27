from abc import ABC, abstractmethod
import pandas as pd
import pyrealsense2 as rs
import numpy as np
import cv2
from matplotlib import pyplot as plt


def init_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    profile = pipeline.start(config)
    return pipeline, profile

def stream_camera(pipeline, new_frame, color_depth):
    """
    new_frame = True --> The stream will open in a cv2 window
    new_frame = False --> The stream will return the color data and you have to use WHILE with this function
    color_depth = True ->> The stream will show colorized depth
    pipeline will be returned from init_camera()
    """
    colorizer = rs.colorizer()
    while True:
        # Wait for a new set of frames
        frames = pipeline.wait_for_frames()

        # Get the depth frame from frameset
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not (depth_frame or color_frame):
            continue

        # Get the depth data
        color_data = np.asanyarray(color_frame.get_data())

        # Align depth and color frames
        align = rs.align(rs.stream.color)
        frames = align.process(frames)

        # Update color and depth frames
        aligned_depth_frame = frames.get_depth_frame()
        colorized_depth= np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
        if new_frame == False and color_depth == True:
            return color_data, colorized_depth
        elif new_frame == False and color_depth == False:
            return color_data
        cv2.imshow("Color Frame", color_data)
        cv2.imshow("Depth Frame", colorized_depth)
        # Break the loop if 'q' is pressed
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
def capture_aligned_images(pipeline, profile):
    # Get the depth sensor
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    for x in range(5):
        pipeline.wait_for_frames()

    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()

    # Get aligned color and depth frames
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    pipeline.stop()

    # RGB Data
    color = np.asanyarray(color_frame.get_data())
    # plt.imshow(color)
    # plt.show()

    # Colorizer for depth visualization (not aligned)
    # colorizer = rs.colorizer()
    # colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

    # Align depth and color frames
    align = rs.align(rs.stream.color)
    frames = align.process(frames)

    # Update color and depth frames
    aligned_depth_frame = frames.get_depth_frame()
    # colorized_depth= np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

    # Get depth intrinsics for converting pixel coordinates to real-world coordinates
    depth_intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

    # cv2.imshow('Depth image after align', colorized_depth)
    # plt.imshow(colorized_depth)
    # plt.show()

    return color, aligned_depth_frame, depth_scale, depth_intrinsics

def transform_coordinates(result_list, depth_intrinsics, depth_scale):
    transformed_result = {}

    for label, coordinates_list in result_list.items():
        transformed_coordinates_list = []

        for coordinates in coordinates_list:
            x, y, depth_value = coordinates
            depth_point = pixel_to_metric_coordinates(x, y, depth_value, depth_intrinsics)

            # Append the transformed coordinates to the list
            transformed_coordinates_list.append(depth_point)

        # Update the result dictionary with transformed coordinates
        transformed_result[label] = transformed_coordinates_list

    return transformed_result

def pixel_to_metric_coordinates(x, y, depth_value, depth_intrinsics):
    # Convert pixel coordinates (x, y) and depth value to real-world coordinates
    depth_point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth_value)
    return depth_point

# class Camera(ABC):
#     @abstractmethod
#     def __init__(self, data):
#         pass
#
#     @staticmethod
#     def getParams(filename):
#         data = pd.read_excel(filename)
#         camera_type = data['type'][1]
#         if camera_type == 'IntelRealSense':
#             return IntelRS(data)
#         elif camera_type == 'Altceva':
#             return Camera2(data)
#         else:
#             raise ValueError('Invalid camera type')
#
#     @abstractmethod
#     def capture_image(self):
#         pass
#
#     def capture_depth(self):
#         pass
#
#     @abstractmethod
#     def start_recording(self):
#         pass
#
#     @abstractmethod
#     def stop_recording(self):
#         pass
#
#
# class IntelRS(Camera):
#     def __init__(self, data):
#         self.data = data
#         self.pipeline = rs.pipeline()
#         self.config = rs.config()
#         self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
#         self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
#         self.recording = False
#
#     def capture_image(self):
#         self.pipeline.start(self.config)
#         frames = self.pipeline.wait_for_frames()
#         color_frame = frames.get_color_frame()
#         color_image = np.asanyarray(color_frame.get_data())
#         self.pipeline.stop()
#         return color_image
#
#     def capture_depth(self):
#         self.pipeline.start(self.config)
#         frames = self.pipeline.wait_for_frames()
#         depth_frame = frames.get_depth_frame()
#         depth_image = np.asanyarray(depth_frame.get_data())
#         self.pipeline.stop()
#         return depth_image
#
#     def start_recording(self):
#         if not self.recording:
#             self.pipeline.start()
#             self.recording = True
#
#     def stop_recording(self):
#         if self.recording:
#             self.pipeline.stop()
#             self.recording = False
#
#
# class Camera2(Camera):
#     def __init__(self, data):
#         self.data = data
#
#     def capture_image(self):
#         pass
#
#     def start_recording(self):
#         pass
#
#     def stop_recording(self):
#         pass
