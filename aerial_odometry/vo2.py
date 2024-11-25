#!/usr/bin/python3

import cv2
import numpy as np


class VisualOdometry:
    def __init__(self, full_img_path, viewbox_size=(20 ,20)):
        """Init the Visual Odometry system

        Args:
            full_img_path (string): _description_
            viewbox_size (tuple of (width and height)): viewbox dimensions. Defaults to (200 ,200).
        """
        # Load ajnd stores the full image
        self.full_image = cv2.imread(full_img_path)
        if self.full_image is None:
            raise ValueError("Could not load the image")
        
        # store viewbox dimensions
        self.viewbox_width, self.viewbox_height = viewbox_size
        
        # init teh current position (top-left corner of viewbox)
        self.current_pos = [0, 0]
        
        # init feature detector
        self.orb = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


    def get_current_view(self):
        """Extract the current viewbox from the full image"""
        x, y = self.current_pos
        return self.full_image[y: y+self.viewbox_height, x: x+self.viewbox_width]
    
    
    def move_viewbox(self, dx, dy):
        """Move te viewox by specified delta x and y

        Args:
            dx (integer): Change in x position
            dy (integer): Change in y position
        """
        new_x = self.current_pos[0] + dx
        new_y = self.current_pos[1] + dy
        
        # ensure we are within the image bounds
        new_x = max(0, min(new_x, self.full_image.shape[1] - self.viewbox_width))
        new_y = max(0, min(new_y, self.full_image.shape[0] - self.viewbox_height))
        
        self.current_pos = [new_x, new_y]
        
    
    def get_position(self):
        """Returns the current position"""
        return self.current_pos
    

def main():
    vo = VisualOdometry("aerial.jpg", viewbox_size=(20, 20))
    
    # create window
    cv2.namedWindow("viewbox")
    
    while True:
        # get current view
        current_view = vo.get_current_view()
        
        # display current view
        cv2.imshow("viewbox", current_view)
        
        # show current position
        print(f"Current position: {vo.get_position()}")
        
        # wait keys
        key = cv2.waitKey(0)
        
        # handle key presses for manual movement
        if key == ord("q"):
            break
        elif key == ord("w"):
            vo.move_viewbox(0, -10)
        elif key == ord("s"):
            vo.move_viewbox(0, 10)
        elif key == ord("a"):
            vo.move_viewbox(-10, 0)
        elif key == ord("d"):
            vo.move_viewbox(10, 0)
            
    cv2.destroyAllWindows()
    
    
if __name__ == "__main__":
    main()
