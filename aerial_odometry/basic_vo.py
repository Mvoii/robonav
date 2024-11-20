#!/usr/bin/python3

import cv2
import numpy as np


class VisualOdometry:
    def __init__(self, full_img_path, viewbox_size=(200, 200)):
        """
            Init the visual odometry system
        Args:
            full_img_path: path to the reference image
            viewbox_size: Tuple of )width, height) for the viewbox
        """
        # load and store the full reference image
        self.full_image = cv2.imread(full_img_path)
        if self.full_image is None:
            raise ValueError("Could not load teh image")

        # store teh viewbox dimensions
        self.viewbox_width, self.viewbox_height = viewbox_size

        # init the current position
        self.current_pos = [0, 0]

        # init feature detector
        self.orb = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def get_current_view(self):
        """Extract the current viewbox from the full image"""
        x, y = self.current_pos

        return self.full_image[y : y + self.viewbox_height, x : x + self.viewbox_width]

    def update_position(self, new_view):
        """
        Update the position based on feature matching betwwen current and new view

        Args:
            new_view: New viewbox imaga to compare against the current position

        Returns:
            bool: Ture if position was successfully updated
        """
        # get current view for comparison
        current_view = self.get_current_view()

        # detect features in both views
        kp1, des1 = self.orb.detectAndCompute(current_view, None)
        kp2, des2 = self.orb.detectAndCompute(new_view, None)

        # Check of enough features were found
        if des1 is None or des2 is None or (len(kp1) < 10) or (len(kp2) < 10):
            return False

        # match features
        matches = self.matcher.match(des1, des2)

        # calculate avg movement
        total_dx = 0
        total_dy = 0
        good_matches = 0

        for match in matches:
            # Get the coordinate of matched key_point
            x1, y1 = kp1[match.queryIdx].pt
            x2, y2 = kp2[match.trainIdx].pt

            # calculate displacement
            dx = x2 - x1
            dy = y2 - y1

            # simple outlier rejection
            if abs(dx) < 50 and abs(dy) < 50:   # threshold for max movement
                total_dx += dx
                total_dy += dy
                good_matches += 1


        if good_matches > 0:
            # calculate avg movement
            avg_dx = int(total_dx / good_matches)
            avg_dy = int(total_dy / good_matches)

            # update position
            new_x = self.current_pos[0] - avg_dx
            new_y = self.current_pos[1] - avg_dy

            # Ensure we stay within image bounds
            new_x = max(0, min(new_x, self.full_image.shape[1] - self.viewbox_width))
            new_y = max(0, min(new_y, self.full_image.shape[0] - self.viewbox_height))

            self.current_pos = [new_x, new_y]
            return True

        return False


    def get_position(self):
        """Return current position"""
        return self.current_pos


def main():
    # init vo with reference image
    vo = VisualOdometry("aerial.jpg", viewbox_size=(200, 200))

    # create a window to display the window
    cv2.namedWindow("viewbox")

    while True:
        # get current view
        current_view = vo.get_current_view()

        # display current view
        cv2.imshow("viewbox", current_view)

        # show current position
        print(f"Current Position: {vo.get_position()}")

        # key press to simulate movement
        key = cv2.waitKey(0)

        # handle key presses for movement
        if key == ord('q'):
            break
        elif key == ord('w'):
            new_view = vo.get_current_view()
            vo.update_position(new_view)
        elif key == ord('s'):
            new_view = vo.get_current_view()
            vo.update_position(new_view)
        elif key == ord('a'):
            new_view = vo.get_current_view()
            vo.update_position(new_view)
        elif key == ord('d'):
            new_view = vo.get_current_view()
            vo.update_position(new_view)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

