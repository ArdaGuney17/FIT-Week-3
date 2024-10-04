import cv2
import mediapipe as mp
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

    # Main loop to process video frames
    while cap.isOpened():

        # Capture frame-by-frame from the webcam
        ret, frame = cap.read()
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not ret:
            print("Failed to grab frame")
            break

        # Sense: Detect joints
        joints = sense.detect_joints(frame)
        landmarks = joints.pose_landmarks

        # If landmarks are detected, calculate the elbow angle
        if landmarks:
            # Extract joint coordinates for the left arm
            # For this example, we will use specific landmark indexes for shoulder, elbow, and wrist
            # shoulder = sense.extract_joint_coordinates(landmarks, 'left_shoulder')
            # elbow = sense.extract_joint_coordinates(landmarks, 'left_elbow')
            # wrist = sense.extract_joint_coordinates(landmarks, 'left_wrist')
            left_knee = sense.extract_joint_coordinates(landmarks, "left_knee")
            right_knee = sense.extract_joint_coordinates(landmarks, "right_knee")
            left_wrist = sense.extract_joint_coordinates(landmarks, "left_wrist")
            right_wrist = sense.extract_joint_coordinates(landmarks, "right_wrist")
            overlay_rect = act.spawn_balloon(1, frame)
            # print(overlay_rect)
            if think.is_landmark_over_image(right_wrist, overlay_rect, frame_width, frame_height):
                print("Hand is over the image!")
            mp.solutions.drawing_utils.draw_landmarks(frame, joints.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
            frame = cv2.flip(frame, 1)
            cv2.imshow("Webcam Feed", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
