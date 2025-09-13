import os

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle


class BoW:
    def __init__(self, n_clusters, n_features):
        self.extractor = cv2.ORB_create(nfeatures=n_features)
        self.n_clusters = n_clusters
        self.kmeans = KMeans(self.n_clusters, verbose=0)
        
        self.database = []
        self.N_i = [0]*n_clusters
        self.N = 0
        self.tf_idf_weights_list = []
        self.tf_idf_means = []

        self.score_threshold = 12.0
        self.consecutive_frames_under_threshold - 0
        self.frames_under_threshold = 2
        
        with open("BagOfWordsKmeans.pkl", "rb") as f:
            self.kmeans = pickle.load(f)

        with open("tf_idf_means.pkl", "rb") as f:
            self.tf_idf_means = pickle.load(f)
