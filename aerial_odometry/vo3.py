#!/usr/bin/python3

from math import sqrt

import cv2
import numpy as np


class VisualOdometry:
    def __init__(self, full_img_path, viewbox_size=(50, 50)):
        # previous init code
        self.full_image = cv2.imread(full_img_path)
        if self.full_image is None:
            raise ValueError("Could not load image")

        self.viewbox_width, self.viewbox_height = viewbox_size
        self.current_pos = [0, 0]
        self.orb = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # nav params
        self.step_size = 10  # pixels to move in each step
        self.target_thresold = 20  # how closewe need to get to the target

    def get_current_view(self):
        x, y = self.current_pos
        return self.full_image[y : y + self.viewbox_height, x : x + self.viewbox_width]

    def move_viewbox(self, dx, dy):
        new_x = self.current_pos[0] + dx
        new_y = self.current_pos[1] + dy

        # ensure we stay within img bounds
        new_x = max(0, min(new_x, self.full_image.shape[1] - self.viewbox_width))
        new_y = max(0, min(new_y, self.full_image.shape[0] - self.viewbox_height))

        self.current_pos = [new_x, new_y]

    def get_position(self):
        return self.current_pos

    def distance_to_target(self, target_pos):
        """calculate Eucledean distance to target position"""
        dx = target_pos[0] - self.current_pos[0]
        dy = target_pos[1] - self.current_pos[1]
        return sqrt(dx * dx + dy * dy)

    def navigate_to_target(self, target_pos, display=True):
        """
        Navigate the viewbox to teh target position using only visul information

        Args:
            target_pos: [x, y] coordinates of a target position
            display: whether to show visualization

        Returns:
            bool: True if the target was reached
        """
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
                dx = int((dx / distance) + self.step_size)
                dy = int((dy / distance) * self.step_size)

            self.move_viewbox(dx, dy)

            # display current view and position if requested
            if display:
                current_view = self.get_current_view()

                # draw target indicator if target is visible
                target_relative_x = target_pos[0] - self.current_pos[0]
                target_relative_y = target_pos[1] - self.current_pos[1]

                if (0 <= target_relative_x , self.viewbox_width and
                    0 <= target_relative_y, self.viewbox_height):
                    cv2.circle(current_view, (target_relative_x, target_relative_y), 5, (0, 255, 0), -1)

                # show current view
                cv2.imshow("Navigation iew", current_view)

                # show teh position information
                print(f"Currrent position: {self.current_pos}, Distance to target: {self.distance_to_target(target_pos):.2f}")

                # small delay for visualization
                if cv2.waitKey(50) & 0xFF == ord('q'):
                    break

            steps_taken += 1

        print(f"Navigation completed in {steps_taken} steps")
        return True


def main():
    # init VO
    vo = VisualOdometry("aerial.jpg", viewbox_size=(20, 20))

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
