import random
import time

import cv2
import mediapipe as mp
from numpy.ma.core import nonzero

from coach import Sense
from coach import Think
from coach import Act

import numpy as np


# Main Program Loop
def main():
    """
    Main function to initialize the exercise tracking application.

    This function sets up the webcam feed, initializes the Sense, Think, and Act components,
    and starts the main loop to continuously process frames from the webcam.
    """

    # Initialize the components: Sense for input, Think for decision-making, Act for output
    sense = Sense.Sense()
    act = Act.Act()
    think = Think.Think(act)

    # Initialize the webcam capture
    cap = cv2.VideoCapture(0)  # Use the default camera (0)

    # Start the timer
    tutorial_duration = 15
    start_time = time.time() + tutorial_duration

    tutorial = cv2.imread("images/balloon_tutorial.png")

    # Main loop to process video frames
    while cap.isOpened():

        # Capture frame-by-frame from the webcam
        ret, frame = cap.read()

        frame = cv2.flip(frame, 1)

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not ret:
            print("Failed to grab frame")
            break


        # Calculate elapsed time
        elapsed_time = time.time() - start_time  # Calculate elapsed time


        if elapsed_time < 0:
            cv2.imshow("Pop The Balloons", tutorial)
            if cv2.waitKey(10) & 0xFF == ord(' '):
                start_time = time.time()
            continue

        if act.popped_count >= 10:
            if act.finish_time is None:
                act.finish_time = elapsed_time
            end_screen = cv2.imread("images/balloons_end_screen.png")
            cv2.putText(end_screen, f'{act.finish_time:.2f}', (490, 220), cv2.FONT_HERSHEY_COMPLEX, 1.4, (255, 160, 230), 4, cv2.LINE_AA)
            cv2.imshow("Pop The Balloons", end_screen)

            if cv2.waitKey(10) & 0xFF == ord(' '):
                # Restart
                act.finish_time = None
                act.popped_count = 0
                act.stage = 0
                act.location = (500, 100)
                act.current_balloon = 0
                start_time = time.time()

            continue

        # Sense: Detect joints
        joints = sense.detect_joints(frame)
        landmarks = joints.pose_landmarks

        # If landmarks are detected, calculate the elbow angle
        if landmarks:
            shoulder = sense.extract_joint_coordinates(landmarks, 'left_shoulder')
            left_knee = sense.extract_joint_coordinates(landmarks, "left_knee")
            right_knee = sense.extract_joint_coordinates(landmarks, "right_knee")
            left_wrist = sense.extract_joint_coordinates(landmarks, "left_wrist")
            right_wrist = sense.extract_joint_coordinates(landmarks, "right_wrist")
            limbs = [left_wrist, left_knee, right_wrist, right_knee]
            overlay_rect = act.show_balloon(act.current_balloon, frame)
            # print(act.current_balloon, limbs[act.current_balloon])

            # Calculate the distance from the camera
            distance = sense.calculate_distance(landmarks)

            decision = think.state

            act.provide_feedback(decision, frame=frame, joints=joints, distance=distance, elapsed_time=elapsed_time)

            if think.is_landmark_over_image(limbs[act.current_balloon], overlay_rect, frame_width, frame_height):
                act.enlarge(frame_width,frame_height)
                print("Hand is over the image!")
            mp.solutions.drawing_utils.draw_landmarks(frame, joints.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)


            cv2.imshow("Pop The Balloons", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
