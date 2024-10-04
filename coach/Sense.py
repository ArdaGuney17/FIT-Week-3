import cv2
import mediapipe as mp
import math
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode


# Sense Component: Detect joints using the camera
# Things you need to improve: Make the skeleton tracking smoother and robust to errors.
class Sense:

    def __init__(self):
        base_options = python.BaseOptions(model_asset_path='models/pose_landmarker_full.task') # delegate=mp.tasks.BaseOptions.Delegate.GPU on linux only
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=VisionTaskRunningMode.VIDEO, # TODO Should be LIVESTREAM
            output_segmentation_masks=True)
        self.detector = vision.PoseLandmarker.create_from_options(options)

        # Initialize the Mediapipe Pose object to track joints

        # self.mp_drawing = mp.solutions.drawing_utils
        # self.mp_pose = mp.solutions.pose.Pose()

        # used later for having a moving avergage
        # self.angle_window = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        # self.previous_angle = -1

    def detect_joints(self, frame, capture):
        return self.detector.detect_for_video(frame, int(capture.get(cv2.CAP_PROP_POS_MSEC)))
