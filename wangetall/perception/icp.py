import numpy as np
import math
from sklearn.neighbors import NearestNeighbors
from skimage.transform import estimate_transform
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

import sys

class ICP:
    """
    Class Based on the ICP implementation of https://github.com/richardos/icp/blob/master/icp.py and Besl and
    McKay, 1992 -
    http://www-evasion.inrialpes.fr/people/Franck.Hetroy/Teaching/ProjetsImage/2007/Bib/besl_mckay-pami1992.pdf

    An implementation of the Iterative Closest Point algorithm that matches a set of M 2D points to another set
    of N 2D (reference) points.
    :param reference_points: the reference point set as a numpy array (N x 2)
    :param points: the point that should be aligned to the reference_points set as a numpy array (M x 2)
    :param max_iterations: the maximum number of iteration to be executed
    :param distance_threshold: the distance threshold between two points in order to be considered as a pair
    :param convergence_translation_threshold: the threshold for the translation parameters (x and y) for the
                                              transformation to be considered converged
    :param convergence_rotation_threshold: the threshold for the rotation angle (in rad) for the transformation
                                               to be considered converged
    :param point_pairs_threshold: the minimum number of point pairs that should exist
    :return: aligned points as a numpy array M x 2
    """
    def __init__(self):
        """
        Testing out with the params
        """
        self.max_iterations=30
        self.distance_threshold=1.4
        self.convergence_translation_threshold=1e-3
        self.convergence_rotation_threshold=1e-4
        self.match_ratio_threshold = 0.5


    def run(self, reference_points, points, key = None, trackid = None):

        self.reference_points = reference_points
        self.points = points
        # nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(self.reference_points)

        for iter_num in range(self.max_iterations):
            #print('------ iteration', iter_num, '------')
            converged =  False

            # closest_point_pairs = []
            closest_point_pairs_idxs = []
            C = cdist(self.points, self.reference_points)
            try:
                _, assignment = linear_sum_assignment(C)
            except ValueError:
                print("ValueError in ICP")
                print(C)
                #breakpoint()
            if self.points.shape[0] < self.reference_points.shape[0]:
                N = self.points.shape[0]
            else:
                N = self.reference_points.shape[0]
            validIdxs = [i for i in range(N) if C[i, assignment[i]]<self.distance_threshold]
            # validIdxs = [i for i in range(N)]

            closest_point_pairs = np.zeros((len(validIdxs),2, 2))
            closest_point_pairs[:,:,0] = self.points[validIdxs]
            closest_point_pairs[:,:,1] = self.reference_points[assignment[validIdxs]]
            for idx in validIdxs:
                closest_point_pairs_idxs.append((idx, assignment[idx]))

            # if only few point pairs, stop process
            #print('number of pairs found:', len(closest_point_pairs))
            # if key == 25 and trackid == 1:
            #     print("PINGGGG!!!")
            #     plt.figure()

            #     breakpoint()


            # All associations obtained in this way are used to estimate a transform that aligns the point set P to Q.
            if len(closest_point_pairs) == 0:
                #print('No better solution can be found!')
                break


            tform = estimate_transform("euclidean", closest_point_pairs[:,:,0], closest_point_pairs[:,:,1])
            closest_rot_angle = tform.rotation
            closest_translation_x, closest_translation_y = tform.translation

            #T he points in P are then updated to their new positions with the estimated transform

            self.points = tform(self.points)

            # rangepts = np.max(self.points, axis = 0)-np.min(self.points, axis = 0)
            # rangerefs = np.max(self.reference_points, axis = 0)-np.min(self.reference_points, axis = 0)
            # and the loop continues until convergence


            #Karel's idea: reject outliers based on scoring num points in dist, penalize num points out of dist
            #Possible implementation: do min(len(closest_point_pairs)/self.reference_points.shape[0],len(closest_point_pairs)/self.points.shape[0])
            #If min < threshold, reject?
            # print("len(closest_point_pairs):",len(closest_point_pairs))
            # print("self.reference_points.shape[0]",self.reference_points.shape[0])
            # print("self.points.shape[0]",self.points.shape[0])
            match_ratio = min(len(closest_point_pairs)/self.reference_points.shape[0],len(closest_point_pairs)/self.points.shape[0])
            # print("Match ratio {}".format(match_ratio))
            # #print(C)
            #plt.plot(self.points[:,0], self.points[:,1],'bo', markersize = 10)
            # plt.plot(self.reference_points[:,0], self.reference_points[:,1],'rs',  markersize = 7)
            # for p in range(N):
            #     plt.plot([self.points[p,0], self.reference_points[assignment[p],0]], [self.points[p,1], self.reference_points[assignment[p],1]], 'k')
            # plt.show()
            # print("Match ratio {}".format(match_ratio))
            #breakpoint()
            if (abs(closest_rot_angle) < self.convergence_rotation_threshold) \
                    and (abs(closest_translation_x) < self.convergence_translation_threshold) \
                    and (abs(closest_translation_y) < self.convergence_translation_threshold) \
                    and match_ratio > self.match_ratio_threshold:

                converged = True
                # if key == 190 and trackid == 2:
                #     plt.figure()
                #     plt.plot(self.points[:,0], self.points[:,1],'bo', markersize = 10)
                #     plt.plot(self.reference_points[:,0], self.reference_points[:,1],'rs',  markersize = 7)
                #     for p in range(N):
                #         plt.plot([self.points[p,0], self.reference_points[assignment[p],0]], [self.points[p,1], self.reference_points[assignment[p],1]], 'k')
                #     plt.show()
                #     breakpoint()

                break


        #The association upon convergence is taken as the final association, with outlier rejection from P to Q.
        # -- outliers not in points now
        return converged, closest_point_pairs_idxs


    def point_based_matching(self, point_pairs):
        """
        This function is based on the paper "Robot Pose Estimation in Unknown Environments by Matching 2D Range Scans"
        by F. Lu and E. Milios.
        :param point_pairs: the matched point pairs [((x1, y1), (x1', y1')), ..., ((xi, yi), (xi', yi')), ...]
        :return: the rotation angle and the 2D translation (x, y) to be applied for matching the given pairs of points
        """
        x_mean = 0
        y_mean = 0
        xp_mean = 0
        yp_mean = 0
        n = len(point_pairs)

        if n == 0:
            return None, None, None

        for pair in point_pairs:

            (x, y), (xp, yp) = pair

            x_mean += x
            y_mean += y
            xp_mean += xp
            yp_mean += yp

        x_mean /= n
        y_mean /= n
        xp_mean /= n
        yp_mean /= n

        s_x_xp = 0
        s_y_yp = 0
        s_x_yp = 0
        s_y_xp = 0
        for pair in point_pairs:

            (x, y), (xp, yp) = pair

            s_x_xp += (x - x_mean)*(xp - xp_mean)
            s_y_yp += (y - y_mean)*(yp - yp_mean)
            s_x_yp += (x - x_mean)*(yp - yp_mean)
            s_y_xp += (y - y_mean)*(xp - xp_mean)

        rot_angle = math.atan2(s_x_yp - s_y_xp, s_x_xp + s_y_yp)
        translation_x = xp_mean - (x_mean*math.cos(rot_angle) - y_mean*math.sin(rot_angle))
        translation_y = yp_mean - (x_mean*math.sin(rot_angle) + y_mean*math.cos(rot_angle))

        return rot_angle, translation_x, translation_y
