# Act Component: Provide feedback to the user
import math
import threading
import time

import mediapipe as mp
import cv2
import numpy as np
import random
import pyttsx3
import queue
import random


# Act Component: Visualization to motivate user, visualization such as the skeleton and debugging information.
# Things to add: Other graphical visualization, a proper GUI, more verbal feedback
class Act:

    def __init__(self):
        self.popped_count = 0
        self.finish_time = None
        self.engine = pyttsx3.init()
        self.speech_queue = queue.Queue()
        self.motivating_utterances = ['keep on going', 'you are doing great. I see it', 'only a few left',
                                      'that is awesome', 'you have almost finished the exercise']

        self.stage = 0
        self.location = (500, 100)
        self.current_balloon = 0
        self.limb_list = [0, 1, 2, 3]
        # Handles balloon inflation and reset after explosion

        t = threading.Thread(target=self._speech_thread, args=())
        t.start()

    def speak_text(self, text):
        """
        Speaks the given text using pyttsx3 text-to-speech engine.
        :param text: The text to be spoken
        """
        self.speech_queue.put(text)

    def _speech_thread(self):
        while True:
            try:
                # Try to get an item with a small timeout to avoid blocking indefinitely
                item = self.speech_queue.get(timeout=1)
                self.engine.say(item)
                self.engine.runAndWait()
                self.speech_queue.task_done()
            except queue.Empty:
                time.sleep(1)
    def provide_feedback(self, decision, frame, joints, distance, elapsed_time):
        """
        Displays the skeleton and some text using open cve.

        :param decision: The decision in which state the user is from the think component.
        :param frame: The currently processed frame form the webcam.
        :param joints: The joints extracted from mediapipe from the current frame.
        :param elbow_angle_mvg: The moving average from the left elbow angle.

        """

        mp.solutions.drawing_utils.draw_landmarks(frame, joints.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

        # Define the number and text to display
        text = ""
        distance_text = ""

        # Correctly check the distance range
        if 0.4 < distance < 0.5:
            distance_text = "You are in range."
        elif 0.4 > distance:
            distance_text = f"You are out of range! Move closer to the camera."
        elif 0.5 < distance:
            distance_text = f"You are out of range! Move away from the camera."

        # Draw the text on the image
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255),
                    2)  # White color for contrast
        cv2.putText(frame, distance_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255),
                    2)  # Adjusted y-coordinate
        # Format the elapsed time to display
        elapsed_time_text = f"Duration: {elapsed_time:.2f} seconds"
        cv2.putText(frame, elapsed_time_text, (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        # Display the frame
        # cv2.imshow('Sport Coaching Program', frame)

    def show_balloon(self, type, frame):
        # Choose image
        balloon_paths = {
            0: "Left_hand",
            1: "Left_knee",
            2: "Right_hand",
            3: "Right_knee"
        }
        balloon_colors = {
            0: "green",
            1: "yellow",
            2: "blue",
            3: "red"
        }
        if 0 <= self.stage <= 5:
            base_path = f"images/{balloon_paths[type]}/{balloon_colors[type]}_balloon"
            if self.stage == 0:
                path = f"{base_path}.png"
            else:
                path = f"{base_path}_popping{self.stage}.png"
        overlay_img = cv2.imread(path)
        overlay_img = cv2.resize(overlay_img, (100, 100))
        overlay_height, overlay_width, _ = overlay_img.shape
        overlay_pos = self.location
        overlay_rect = (
            overlay_pos[0], overlay_pos[1], overlay_pos[0] + overlay_width, overlay_pos[1] + overlay_height)  # hit box
        frame[overlay_pos[1]:overlay_pos[1] + overlay_height,
        overlay_pos[0]:overlay_pos[0] + overlay_width] = overlay_img  # put in frame
        return overlay_rect

    def random_location(self, frame_width, frame_height):
        x, y = 0, 0
        if self.current_balloon == 0:
            x1lim, x2lim = 0, int(frame_width / 2) - 100
            y1lim, y2lim = 0, int(frame_height / 2) - 100
        elif self.current_balloon == 1:
            x1lim, x2lim = 0, int(frame_width / 2) - 100
            y1lim, y2lim = int(frame_height / 2), frame_height - 100
        elif self.current_balloon == 2:
            x1lim, x2lim = int(frame_width / 2), frame_width - 100
            y1lim, y2lim = 0, int(frame_height / 2) - 100
        elif self.current_balloon == 3:
            x1lim, x2lim = int(frame_width / 2), frame_width - 100
            y1lim, y2lim = int(frame_height / 2), frame_height - 100
        # return (0, frame_height-100)
        return random.randrange(x1lim, x2lim, 1), random.randrange(y1lim, y2lim, 1)

    def enlarge(self, frame_width, frame_height):
        if self.stage == 5:
            self.stage = 0
            self.current_balloon = random.choice([x for x in self.limb_list if x != self.current_balloon])
            self.location = self.random_location(frame_width, frame_height)
            self.popped_count += 1
        else:
            self.stage += 1

# class Bubble:
#     def __init__(self, overlay_pos, overlay_rect, overlay_image):
#         """
#         Initialize the ImageRectangle with its top-left corner (x, y),
#         and the width and height of the rectangle.
#         """
#         self.overlay_pos = overlay_pos
#         self.size = 100
#         self.overlay_rect = overlay_rect
#         self.overlay_image = overlay_image
