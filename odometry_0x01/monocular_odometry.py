import os
import numpy as np
import cv2
import random
from matplotlib import pyplot as plt

from tqdm import tqdm

class VisualOdometry():
    def __init__(self, data_dir):
        self.K, self.P = self._load_calib(os.path.join(data_dir, 'calib.txt'))
        self.gt_poses = self._load_poses(os.path.join(data_dir, 'poses.txt'))
        self.images = self._load_images(os.path.join(data_dir, 'image_1'))
        self.orb = cv2.ORB_create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
        
    @staticmethod
    def _load_calib(filepath):
        """Loads teh calibration of the camera

        Args:
            filepath (str): The file path to teh camera file
            
        Returs:
            K (ndarray): Intrinsic parameters
            P (ndarray): Projection matrix
        """
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readable(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K, P
    
    @staticmethod
    def _load_poses(filepath):
        """Loads the Gt poses

        Args:
            filepath (str): the file path to the poses file
            
        Returns:
            poses (ndarray): The GT poses
        """
        poses = []
        with open(filepath, "r") as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=" ")
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
        return poses
    
    @staticmethod
    def _load_images(filepath):
        """Loads the images

        Args:
            filepath (str_): The path to the image dir

        Returns:
            imaes (list): grayscale images
        """
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]
    
    @staticmethod
    def _form_transf(R, t):
        """ makes a transformation matrix from the given
            rotational matrix and translational vector

        Args:
            R (ndarray): The rotational matrix
            t (list): The translation vector
        
        Returns:
            T (ndarray): The transformation matrix
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_matches(self, i):
        """teh function detects and computes keypoints
            and descriptors from teh i-1 'th and i'th image
            using the

        Args:
            i (int): The current frame
        
        Returns:
            q1 (ndarray): the good keypoints matches position in i-1'th image
            q2 (ndarray): The good keypoints matche sposition in the i'th image
        """
        keypoints1, descriptors1 = self.orb.detectAndCompute(self.images[i - 1], None)
        keypoints2, descriptors2 = self.orb.detectAndCompute(self.images[i - 1], None)
        
        matches = self.flann.knnMatch(descriptors1, descriptors2, k=2)
        
        # store all teh good matche sas per Lowe's ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good.append(m)
        
        q1 = np.float32([ keypoints1[m.queryIdx].pt for m in good ])
        q2 = np.float32([ keypoints2[m.trainIdx].pt for m in good ])
        
        draws_params = dict(matchColor = -1, # draw matches in green
                            singlePointColor = None,
                            matchesmask = None, # draw only inliers
                            flags = 2)
        
        img3 = cv2.drawMatches(self.images[i], keypoints1, self.images[i - 1], keypoints2, good, None, **draws_params)
        cv2.imshow("image", img3)
        cv2.waitKey(750)
        
        plt.imshow(img3, 'gray'), plt.show()
        plt.imshow(self.images[i]), plt.show()
        plt.imshow(self.images[i - 1]), plt.show()
        
        return q1, q2
    
    def get_pose(self, q1, q2):
        """Calculates the transformation matrix

        Args:
            q1 (ndarray): The good keypoints matche sposition in i-1'th image
            q2 (ndarray): The good keypoints matches position in i'th image
        
        Returns:
            tranformation_matrix (ndarray): The transformation matrix
        """
        
        Essential, mask = cv2.findEssentialMat(q1, q2, self.K)
        
        R, t = self.decomp_essential_mat(Essential, q1, q2)
        
        return self._form_transf(R, t)
    
    def decomp_essential_mat(self, E, q1, q2):
        """Decompose the essential matrix

        Args:
            E (ndarray): Essential matrix
            q1 (ndarray): The good keypoints matches position in the i-1'th image
            q2 (ndarray): The good keypoints matches position in the i'th image
        
        Returns:
            right_pair (list): Conatins the rotation matrix and translation vector
        """
        
        R1, R2, t = cv2.decomposeEssentialMat(E)
        T1 = self._form_transf(R1, np.ndarray.flatten(t))
        T2 = self._form_transf(R2, np.ndarray.flatten(t))
        T3 = self._form_transf(R1, np.ndarray.flatten(t))
        T4 = self._form_transf(R2, np.ndarray.flatten(t))
        transformations = [T1, T2, T3, T4]
        
        # homogenize K
        K = np.concatenate((self.K, np.zeros((3, 1)) ), axis = 1)
        
        # list of projections
        projections = [K @ T1, K @ T2, K @ T3, K @ T4]
        
        np.set_printoptions(suppress=True)
        
        
        positives = []
        for P, T in zip(projections, transformations):
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            hom_Q2 = T @ hom_Q1
            # Un-homogenize
            Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            Q2 = hom_Q2[:3, :] / hom_Q2[3, :]
            
            total_sum = sum(Q2[2, :] > 0) + sum(Q1[2, :] > 0)
            relative_scale = np.mean(np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis = -1)/
                                     np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis = -1))
            positives.append(total_sum + relative_scale)
            
            # decompose the essential matrix using the inbuilt opencv function
            # form the 4 possible transformation matrix T from R1, R2, and t
            # Create projection matrix using eact T, and triangulate points hom_Q1
            # Transform hom_Q1 to second camera using T to create hom_Q2
            # Count how many pounts in hom_Q1 and hom_Q2 with positive z values
            # Return R and  t pair whoch resulted in the most points with positive z's
            
            max = np.argmax(positives)
            if (max == 2):
                # print (-t)
                return R1, np.ndarray.flatten(-t)
            elif (max == 3):
                # print(-t)
                return R2, np.ndarray.flatten(-t)
            elif (max == 0):
                # print(-t)
                return R1, np.ndarray.flatten(-t)
            elif (max == 1):
                # print(-t)
                return R2, np.ndarray.flatten(-t)

def main():
    data_dir = 'KITTI_seq_1'
    vo = VisualOdometry(data_dir)
    
    # play_trip(vo.images)
    
    gt_path = []
    estimated_path = []
    for i, gt_pose in enumerate(tqdm(vo.gt_poses, unit='pose')):
        if i == 0:
            cur_pose = gt_pose
        else:
            q1, q2 = vo.get_matches(i)
            transf = vo.get_pose(q1, q2)
            cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))
            print("\n Ground truth pose:\n" + str(gt_pose))
            print("\n Current pose:\n" + str(cur_pose))
            print("The Current pose used x,y: \n" + str(cur_pose[0, 3] + " " + str(cur_pose[2, 3])))
        gt_path.append(gt_pose[0, 3], gt_pose[2, 3])
        estimated_path((cur_pose[0, 3], cur_pose[2, 3]))
        

    plotting.visualize_paths(gt_path, estimated_path, "visual odometry", file_out=os.path.basename(data_dir))

if __name__ == "__main__":
    main()

