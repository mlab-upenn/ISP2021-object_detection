import numpy as np
import math
from sklearn.neighbors import NearestNeighbors
from skimage.transform import estimate_transform
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from skimage import transform
import sys


class ICP:
    def __init__(self):
        """
        Testing out with the params
        """
        self.max_iterations=5
        self.distance_threshold=0.3
        self.match_ratio_threshold = 0.5


    def run(self, reference_points, points, key = None, trackid = None):

        self.reference_points = reference_points-np.mean(reference_points, axis = 0)
        self.points = points-np.mean(reference_points, axis = 0)
        # nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(self.reference_points)

        for iter_num in range(self.max_iterations):
            converged =  False

            closest_point_pairs_idxs = []
            C = cdist(self.points, self.reference_points)
            try:
                _, assignment = linear_sum_assignment(C)
            except ValueError:
                print("ValueError in ICP")
                breakpoint()
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
            if closest_point_pairs.shape[0]<=2:
                #print('No better solution can be found!')
                break


            tform = estimate_transform("euclidean", closest_point_pairs[:,:,0], closest_point_pairs[:,:,1])
            closest_rot_angle = tform.rotation
            closest_translation_x, closest_translation_y = tform.translation
            #The points in P are then updated to their new positions with the estimated transform
            mini_tform = transform.EuclideanTransform(
                            rotation=tform.rotation*0.1,
                            translation = tuple(tform.translation*0.1)
                            )
            self.points = mini_tform(self.points)

            match_ratio = min(len(closest_point_pairs)/self.reference_points.shape[0],len(closest_point_pairs)/self.points.shape[0])
            close_constraint = abs(max(tform.translation)) < 0.3 and abs(tform.rotation) < 1
            close_enough = abs(max(tform.translation)) < 0.1 and abs(tform.rotation) < 0.1
            if(match_ratio > self.match_ratio_threshold and close_constraint) or close_enough:
                # if close_enough:
                #     print("Close enough!")
                #     print(tform)
                #     print("Iter num {}".format(iter_num))
                # if (match_ratio > self.match_ratio_threshold and close_constraint):
                #     print("Match ratio met.")
                converged = True
                break


        #The association upon convergence is taken as the final association, with outlier rejection from P to Q.
        # -- outliers not in points now
        return converged, closest_point_pairs_idxs

def convert_to_SE2(x):
    ret = np.hstack((x, np.ones((x.shape[0], 1))))
    return ret


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
