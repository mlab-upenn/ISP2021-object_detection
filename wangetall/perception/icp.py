import numpy as np
import math
from sklearn.neighbors import NearestNeighbors
from skimage.transform import estimate_transform
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

import sys


class ICP:
    def __init__(self):
        """
        Testing out with the params
        """
        self.max_iterations=30
        self.distance_threshold=1
        self.match_ratio_threshold = 0.5


    def run(self, reference_points, points, key = None, trackid = None):

        self.reference_points = reference_points
        self.points = points
        # nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(self.reference_points)

        for iter_num in range(self.max_iterations):
            converged =  False

            closest_point_pairs_idxs = []
            C = cdist(self.points, self.reference_points)
            try:
                _, assignment = linear_sum_assignment(C)
            except ValueError:
                print("ValueError in ICP")
            if self.points.shape[0] < self.reference_points.shape[0]:
                N = self.points.shape[0]
            else:
                N = self.reference_points.shape[0]
            validIdxs = [i for i in range(N) if C[i, assignment[i]]<self.distance_threshold]

            closest_point_pairs = np.zeros((len(validIdxs),2, 2))
            closest_point_pairs[:,:,0] = self.points[validIdxs]
            closest_point_pairs[:,:,1] = self.reference_points[assignment[validIdxs]]
            for idx in validIdxs:
                closest_point_pairs_idxs.append((idx, assignment[idx]))


            # All associations obtained in this way are used to estimate a transform that aligns the point set P to Q.
            if len(closest_point_pairs) == 0:
                #print('No better solution can be found!')
                break


            tform = estimate_transform("euclidean", closest_point_pairs[:,:,0], closest_point_pairs[:,:,1])
            closest_rot_angle = tform.rotation
            closest_translation_x, closest_translation_y = tform.translation

            #The points in P are then updated to their new positions with the estimated transform
            self.points = tform(self.points)

            match_ratio = min(len(closest_point_pairs)/self.reference_points.shape[0],len(closest_point_pairs)/self.points.shape[0])
            close_enough = abs(max(tform.translation)) < 0.1 and abs(tform.rotation) < 0.01

            if(match_ratio > self.match_ratio_threshold) or close_enough:
                converged = True
                break


        #The association upon convergence is taken as the final association, with outlier rejection from P to Q.
        # -- outliers not in points now
        return converged, closest_point_pairs_idxs


# def main():
#     icp = ICP()
#     points = sys.argv[1]
#     scans = sys.argv[2]
#     points = np.load(points)
#     scans = np.load(scans)
#
#     icp.run(points, scans)

    # plt.scatter(points[:,0], points[:,1],'r',  markersize = 30, label = "points")
    # plt.scatter(scans[:,0], scans[:,1],'b',  markersize = 15, label = "scans")
    # plt.legend()
    # plt.show()

# if __name__ == "__main__":
#     main()
