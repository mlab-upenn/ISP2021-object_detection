import numpy as np
import math
from sklearn.neighbors import NearestNeighbors
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
        self.distance_threshold=5
        self.convergence_translation_threshold=1e-3
        self.convergence_rotation_threshold=1e-4
        self.point_pairs_threshold=0

    def run(self, reference_points, points):
        self.reference_points = reference_points
        self.points = points
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(self.reference_points)

        for iter_num in range(self.max_iterations):
            #print('------ iteration', iter_num, '------')
            static=  False

            closest_point_pairs = []
            closest_point_pairs_idxs = []

              # list of point correspondences for closest point rule
            distances, indices = nbrs.kneighbors(self.points)
            # Step 1: A point in P is associated to its nearest neighbour in Q if their distance is within a certain threshold,
            for nn_index in range(len(distances)):
                if distances[nn_index][0] < self.distance_threshold: # ELSE OUTLIER?
                    closest_point_pairs.append((self.points[nn_index], self.reference_points[indices[nn_index][0]]))
                    closest_point_pairs_idxs.append((nn_index, indices[nn_index][0]))
                    # otherwise it is discarded as an outlier for this iteration and become unassociated to any point in Q.

            # if only few point pairs, stop process
            #print('number of pairs found:', len(closest_point_pairs))
            if len(closest_point_pairs) < self.point_pairs_threshold:
                #print('No better solution can be found (very few point pairs)!')
                break

            # All associations obtained in this way are used to estimate a transform that aligns the point set P to Q.
            closest_rot_angle, closest_translation_x, closest_translation_y = self.point_based_matching(closest_point_pairs)

            if closest_rot_angle is None or closest_translation_x is None or closest_translation_y is None:
                #print('No better solution can be found!')
                break

            #T he points in P are then updated to their new positions with the estimated transform
            c, s = math.cos(closest_rot_angle), math.sin(closest_rot_angle)
            rot = np.array([[c, -s],
                            [s, c]])
            aligned_points = np.dot(self.points, rot.T)
            aligned_points[:, 0] += closest_translation_x
            aligned_points[:, 1] += closest_translation_y

            self.points = aligned_points

            # and the loop continues until convergence
            if (abs(closest_rot_angle) < self.convergence_rotation_threshold) \
                    and (abs(closest_translation_x) < self.convergence_translation_threshold) \
                    and (abs(closest_translation_y) < self.convergence_translation_threshold):
                static= True
                break
        #The association upon convergence is taken as the final association, with outlier rejection from P to Q.
        # -- outliers not in points now
        return static, closest_point_pairs_idxs


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
