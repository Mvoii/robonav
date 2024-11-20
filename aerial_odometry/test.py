#!/usr/bin/python3

import cv2

# Load image
image = cv2.imread("map.jpg", cv2.IMREAD_GRAYSCALE)

# init the orb detectot
orb = cv2.ORB_create(nfeatures=1000)

# detect key points and descriptors
keypoints, descriptors = orb.detectAndCompute(image, None)

# draw keypoints on the image for visualization
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))


# feature matching from teh viewport
# brute force matcher, maybe flann

# init brute force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors btwn two imgs
matches = bf.match(descriptors_current_viewpoert, descriptors_map)

# sort matches based on distance (lower distance = better match)
matches = sorted(matches, key=lambda x: x.distance)


# movement and path planning
# a* to calc optimal path
# implement a cost function
# pseudocode
def a_star_search(start, goal, grid):
    open_list = set([start])
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_list:
        current = min(open_list, key=lambda x: f_score[x])
        if current == goal:
            return reconstruct_path(came_from, current)

        open_list.remove(current)
        for neighbour in get_neihbours(current, grid):
            tentative_g_score = g_score[current] + cost(current, neighbour)
            if tentative_g_score < g_score.get(neighbour, float("inf")):
                came_from[neighbour] = current
                g_score[neighbour] = tentative_g_score
                f_score[neighbour] = g_score[neighbour] + heuristic(neighbour, goal)
                open_list.add(neighbour)


# viz and sim using plt
# highlight path and current location for better debugging
import matplotlib.pyplot as plt

plt.imshow(image, cmap="gray")
plt.scatter([start_x, end_x], [start_y, end_y], c=["green", "red"])
plt.show()
