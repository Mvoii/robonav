#!/usr/bin/python3

from math import sqrt

import cv2
import numpy as np


class VisualOdometry:
    def __init__(self, full_img_path, viewbox_size=(50, 50), step_size=10, target_threshold=20):
        """
            Initialize the visual odometry system

        Args:
            full_img_path (str): path to image map
            viewbox_size (tuple, optional): dimensions of with and height of viewport. Defaults to (50, 50).
            step_size (int): Pixels to move in each step
            target_threshold (int): Distance to consider the target reached
        """
        self.full_image = cv2.imread(full_img_path)
        if self.full_image is None:
            raise ValueError("Could not load image")

        self.viewbox_width, self.viewbox_height = viewbox_size
        self.current_pos = [0, 0]
        self.step_size = step_size
        self.target_thresold = target_threshold
        
        self.orb = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # display image size
        self.display_image_size()
        
    def display_image_size(self):
        """Print the dimensions of the full reference image"""
        height, width, channels = self.full_image.shape
        print(f"reference image size: width={width}, height={height}, channels={channels}")

    def get_current_view(self):
        """Return the current view viewport of the image"""
        x, y = self.current_pos
        return self.full_image[y:y + self.viewbox_height, x:x + self.viewbox_width]

    def move_viewbox(self, dx, dy):
        """Move the viewport by dx, dy while staying within bounds"""
        new_x = max(0, min(self.current_pos[0] + dx, self.full_image.shape[1] - self.viewbox_width))
        new_y = max(0, min(self.current_pos[1] + dy, self.full_image.shape[0] - self.viewbox_height))

        self.current_pos = [new_x, new_y]

    def get_position(self):
        return self.current_pos

    def distance_to_target(self, target_pos):
        """calculate Eucledean distance to target position"""
        dx = target_pos[0] - self.current_pos[0]
        dy = target_pos[1] - self.current_pos[1]
        return sqrt(dx ** 2 + dy ** 2)

    def navigate_to_target(self, target_pos, display=True):
        """
        Navigate the viewbox to teh target position using only visul information

        Args:
            target_pos (list): [x, y] coordinates of a target position
            display (bool): whether to show visualization

        Returns:
            bool: True if the target was reached
        """
        if not (0 <= target_pos[0] <= self.full_image.shape[1] and 0 <= target_pos[1]):
            raise ValueError("Target position out of bounds")
        
        print(f"Starting point {self.current_pos} to {target_pos}")
        max_steps = 1000  # prevent infinite loopy loops
        steps_taken = 0

        while self.distance_to_target(target_pos) > self.target_thresold:
            if steps_taken >= max_steps:
                print("Maximum steps reached without getting to target")
                return False

            # calculate the direction to target
            dx = target_pos[0] - self.current_pos[0]
            dy = target_pos[1] - self.current_pos[1]

            # normalize teh direction and multiply by step size
            distance = sqrt(dx*dx + dy*dy)
            if distance > 0:
                dx = int((dx / distance) * self.step_size)
                dy = int((dy / distance) * self.step_size)

            self.move_viewbox(dx, dy)

            # display current view and position if requested
            if display:
                self.display_navigation(target_pos)

            steps_taken += 1

        print(f"Navigation target reached in {steps_taken} steps")
        return True

    def display_navigation(self, target_pos):
        """Display the current view and position for visualization"""
        current_view = self.get_current_view().copy()
        
        # target relative to the current view
        target_relative_x = target_pos[0] - self.current_pos[0]
        target_relative_y = target_pos[1] - self.current_pos[1]
        
        # draw target of visible in currrent view
        if 0 <= target_relative_x < self.viewbox_width and 0 <= target_relative_y < self.viewbox_height:
            cv2.circle(current_view, (target_relative_x, target_relative_y), 5, (0, 255, 0), -1)
        
        cv2.imshow("Navigation view", current_view)
        print(f"Current Position: {self.current_pos}, Distance to Target: {self.distance_to_target(target_pos):.2f}")
        
        # small dela8y for visualization
        if cv2.waitKey(50) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            exit()

def main():
    # init VO
    vo = VisualOdometry("new_ref_img.jpg", viewbox_size=(20, 20), step_size=15)

    # set the start point
    vo.move_viewbox(100, 100)

    # set target
    target_position = [400, 300]

    success = vo.navigate_to_target(target_position)

    # conditional for if reached taget
    if success:
        print("Successfully reached the target")
    else:
        print("Failed to reach target")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
