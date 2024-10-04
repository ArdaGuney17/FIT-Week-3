# Act Component: Provide feedback to the user
import threading
import time

import mediapipe as mp
import cv2
import numpy as np
import random
import pyttsx3
import queue


# Act Component: Visualization to motivate user, visualization such as the skeleton and debugging information.
# Things to add: Other graphical visualization, a proper GUI, more verbal feedback
class Act:

    def __init__(self):
        # Balloon size and transition tracking for visualization
        self.balloon_size = 50
        self.transition_count = 0
        self.max_transitions = 10  # Explodes after 10 transitions
        self.exploded = False  # Track whether the balloon exploded
        self.explosion_fragments = []  # Store explosion fragments
        self.explosion_frame_count = 0  # Frame counter for explosion duration
        self.explosion_duration = 30  # Number of frames to show explosion effect
        self.engine = pyttsx3.init()
        self.speech_queue = queue.Queue()

        self.motivating_utterances = ['keep on going', 'you are doing great. I see it', 'only a few left',
                                      'that is awesome', 'you have almost finished the exercise']
        # Handles balloon inflation and reset after explosion

        t = threading.Thread(target=self._speech_thread, args=())
        t.start()

    def handle_balloon_inflation(self):
        """
        Increases the size of the balloon with each successful repetition.
        """
        if not self.exploded:  # Only inflate if balloon hasn't exploded

            self.transition_count += 1
            self.balloon_size += 10  # Inflate balloon by 10 units per transition

            motivation_text = random.choice(self.motivating_utterances)
            text = "%s %s" % (self.transition_count, motivation_text)

            self.speak_text(text)

            # Check if balloon should explode

            if self.transition_count >= self.max_transitions:
                self.explode_balloon()

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

    def explode_balloon(self):
        """
        Handles the visual effect of the balloon exploding.
        """

        self.exploded = True  # Mark the balloon as exploded
        self.create_explosion_fragments()  # Generate the explosion fragments
        self.speak_text("boooom booooom booom")

    def reset_balloon(self):
        """
        Resets the balloon after it explodes.
        """

        self.transition_count = 0
        self.balloon_size = 50  # Reset balloon size
        self.exploded = False  # Reset explosion state
        self.explosion_frame_count = 0  # Reset the explosion frame counter
        self.explosion_fragments.clear()  # Clear the fragments after explosion

        self.speak_text("You did great! Let's reset the balloon.")
        # Create explosion fragments with random sizes and positions

    def create_explosion_fragments(self):
        # Generate random "fragments" for explosion effect
        for _ in range(20):
            fragment = {
                'position': (random.randint(200, 300), random.randint(200, 400)),
                'size': random.randint(5, 15),
                'color': (0, 0, 255),  # Red fragments
                'dx': random.randint(-10, 10),  # X-direction movement
                'dy': random.randint(-10, 10)  # Y-direction movement
            }
            self.explosion_fragments.append(fragment)

        # Visualization of the balloon and explosion in OpenCV

    def visualize_balloon(self):
        """
        Renders the balloon .
        """

        # Create a black background
        img = np.zeros((500, 500, 3), dtype=np.uint8)

        if not self.exploded:
            # Draw the balloon (a circle) with dynamic size if it hasn't exploded
            cv2.circle(img, (250, 300), self.balloon_size, (0, 0, 255), -1)  # Red balloon
        else:
            # Draw explosion fragments if balloon has exploded
            for fragment in self.explosion_fragments:
                x, y = fragment['position']
                size = fragment['size']
                color = fragment['color']

                # Move fragments in random directions
                x += fragment['dx']
                y += fragment['dy']
                fragment['position'] = (x, y)

                # Draw each fragment as a small circle
                cv2.circle(img, (x, y), size, color, -1)

            self.explosion_frame_count += 1

            # Reset the balloon after the explosion effect finishes
            if self.explosion_frame_count >= self.explosion_duration:
                self.reset_balloon()

        cv2.putText(img, f'Repeat flexing/bending your left arm to pop the balloon!', (0, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2, cv2.LINE_AA)

        # Add transition count and text
        cv2.putText(img, f'Repetitions: {self.transition_count}', (150, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f'Balloon Size: {self.balloon_size}', (150, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the image in the window
        cv2.imshow('Flex and bend your left elbow!', img)

        # Wait for 1 ms and check if the window should be closed
        cv2.waitKey(1)

    def provide_feedback(self, decision, frame, joints, elbow_angle_mvg):
        """
        Displays the skeleton and some text using open cve.

        :param decision: The decision in which state the user is from the think component.
        :param frame: The currently processed frame form the webcam.
        :param joints: The joints extracted from mediapipe from the current frame.
        :param elbow_angle_mvg: The moving average from the left elbow angle.

        """

        mp.solutions.drawing_utils.draw_landmarks(frame, joints.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

        # Define the number and text to display
        number = elbow_angle_mvg
        text = " "
        if decision == 'flexion':
            text = "You are flexing your elbow! %s" % number
        elif decision == 'extension':
            text = "You are extending your elbow! %s" % number

        # Set the position, font, size, color, and thickness for the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = .9
        font_color = (0, 0, 0)  # White color in BGR
        thickness = 2

        # Define the position for the number and text
        text_position = (50, 50)

        # Draw the text on the image
        cv2.putText(frame, text, text_position, font, font_scale, font_color, thickness)

        # Display the frame (for debugging purposes)
        cv2.imshow('Sport Coaching Program', frame)

    def spawn_balloon(self, type, frame):
        overlay_img = cv2.imread('images/hand.jpg')
        overlay_img = cv2.resize(overlay_img, (100, 100))
        overlay_height, overlay_width, _ = overlay_img.shape
        overlay_pos = (50, 50)  # Top-left position (x, y)
        overlay_rect = (overlay_pos[0], overlay_pos[1], overlay_pos[0] + overlay_width, overlay_pos[1] + overlay_height)
        frame[overlay_pos[1]:overlay_pos[1] + overlay_height,
        overlay_pos[0]:overlay_pos[0] + overlay_width] = overlay_img
        return overlay_rect
