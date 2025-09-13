#!/usr/bin/python3

from math import sqrt

import cv2
import numpy as np


class FeatureBasedNavigation:
    def __init__(
        self, full_img_path, viewbox_size=(50, 50), step_size=10, target_threshold=20
    ):
        """
        Init the feature based navigation system

        Args:
            full_img_path (str): Path to the image map
            viewbox_size (tuple): Dimensions (width, height) of the viewport
            step_size (int): Pixels to move in each step
            target_threshold (int): Distance to consider the target reached
        """
        self.full_image = cv2.imread(full_img_path, cv2.IMREAD_GRAYSCALE)
        if self.full_image is None:
            raise ValueError("Could not load image at given path")

        self.viewbox_width, self.viewbox_height = viewbox_size
        self.current_pos = [20, 45]
        self.step_size = step_size
        self.target_threshold = target_threshold

        # ORB detector and matcher
        self.orb = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        counter = 0

    def get_current_view(self):
        """Return the current viewport of the image"""
        x, y = self.current_pos
        return self.full_image[y: y + self.viewbox_height, x: x + self.viewbox_width]

    def detect_features(self, image):
        """Detect keypoints and descriptors in the given image"""
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        return keypoints, descriptors

    def match_features(self, desc1, desc2):
        """Match descriptors using BFMatcher"""
        matches = self.matcher.match(desc1, desc2)
        return sorted(matches, key=lambda x: x.distance)

    def estimate_motion(self, matches, kp1, kp2):
        """Estimate motion using feature matches"""
        if len(matches) < 0:
            raise ValueError("Not enough matche sto estimate motion")

        # estimate matched points
        src_points = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_points = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # find homograpghy or Essential matrix
        print(f"finding Essential matrix: {counter}")
        H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
        counter += 1

        return H, mask

    def move_viewbox(self, dx, dy):
        """Move the viewport by dx, dy while staying within bounds"""
        new_x = max(
            0,
            min(
                self.current_pos[0] + dx, self.full_image.shape[1] - self.viewbox_width
            ),
        )
        new_y = max(
            0,
            min(
                self.current_pos[1] + dy, self.full_image.shape[0] - self.viewbox_height
            ),
        )
        self, current_pos = [new_x, new_y]

    def navigate_to_target(self, target_pos, display=True):
        """
        Navigate to the target position using feature based matching

        Args:
            target_pos (list): [x, y] coordinates of the target position
            display (bool): whether to display the navigation process
        """
        if not (
            0 <= target_pos[0] <= self.full_image.shape[1]
            and 0 <= target_pos[1] <= self.full_image.shape[0]
        ):
            raise ValueError("Target position is out of bounds")

        print(f"navigation from {self.current_pos} to {target_pos}")
        previous_view = self.get_current_view()
        prev_kp, prev_desc = self.detect_features(previous_view)

        max_steps = 1000
        steps_taken = 0

        while steps_taken < max_steps:
            current_view = self.get_current_view()
            cur_kp, cur_desc = self.detect_features(current_view)

            matches = self.match_features(prev_desc, cur_desc)

            # estimate motion
            try:
                H, mask = self.estimate_motion(matches, prev_kp, cur_kp)
                motion_vector = H[:2, 2]  # extraact translation matrix
                dx, dy = int(motion_vector[0]), int(motion_vector[1])
            except ValueError as e:
                print(f"Error in motion estimation: {e}")
                break

            self.move_viewbox(dx, dy)
            steps_taken += 1

            # check if within target threshold
            if (
                sqrt(
                    (target_pos[0] - self.current_pos[0]) ** 2
                    + (target_pos[1] - self.current_pos[1]) ** 2
                )
                < self.target_threshold
            ):
                print(f"Target reached in {steps_taken} steps")
                break

            if display:
                self.display_navigation(target_pos, cur_kp, matches)

        print(f"Navigation complet in {steps_taken} steps")

    def display_navigation(self, target_pos, keypoints, matches):
        """Visualize the navigation with keypoints and matches"""
        current_view = self.get_current_view()
        matched_view = cv2.drawMatches(
            self.full_image,
            keypoints,
            current_view,
            keypoints,
            matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        cv2.imshow("Feature based nav", matched_view)

        if cv2.waitKey(50) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            exit()


def main():
    nav = FeatureBasedNavigation(
        "new_ref_img.jpg", viewbox_size=(100, 100), step_size=15
    )
    nav.move_viewbox(100, 100)  # start pos
    nav.navigate_to_target([2400, 1300])  # target pos
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
