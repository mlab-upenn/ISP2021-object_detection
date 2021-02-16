import numpy as np
import math
from sklearn.neighbors import NearestNeighbors

class ICP:
    """
    Class Based on the ICP implementation of https://github.com/richardos/icp/blob/master/icp.py and Besl and McKay, 1992 -
    http://www-evasion.inrialpes.fr/people/Franck.Hetroy/Teaching/ProjetsImage/2007/Bib/besl_mckay-pami1992.pdf
    """
    def __init__(self, reference_points, points):
        """
        Testing out with the default params from  https://github.com/richardos/icp/blob/master/icp.py, probably
        needs to be adjusted later.
        """
        self.max_iterations=100
        self.distance_threshold=0.3
        self.convergence_translation_threshold=1e-3
        self.convergence_rotation_threshold=1e-4
        self.point_pairs_threshold=10

    def run(self, reference_points, points):
        self.reference_points = reference_points
        self.points = points
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(self.reference_points)

        for iter_num in range(self.max_iterations):
            #print('------ iteration', iter_num, '------')

            closest_point_pairs = []  # list of point correspondences for closest point rule

            distances, indices = nbrs.kneighbors(self.points)
            for nn_index in range(len(distances)):
                if distances[nn_index][0] < self.distance_threshold:
                    closest_point_pairs.append((self.points[nn_index], self.reference_points[indices[nn_index][0]]))

            # if only few point pairs, stop process
            #print('number of pairs found:', len(closest_point_pairs))
            if len(closest_point_pairs) < self.point_pairs_threshold:
                print('No better solution can be found (very few point pairs)!')
                break

            # compute translation and rotation using point correspondences
            closest_rot_angle, closest_translation_x, closest_translation_y = self.point_based_matching(closest_point_pairs)

            if closest_rot_angle is None or closest_translation_x is None or closest_translation_y is None:
                #print('No better solution can be found!')
                break

            # transform 'points' (using the calculated rotation and translation)
            c, s = math.cos(closest_rot_angle), math.sin(closest_rot_angle)
            rot = np.array([[c, -s],
                            [s, c]])
            aligned_points = np.dot(self.points, rot.T)
            aligned_points[:, 0] += closest_translation_x
            aligned_points[:, 1] += closest_translation_y

            # update 'points' for the next iteration
            self.points = aligned_points

            # check convergence
            if (abs(closest_rot_angle) < self.convergence_rotation_threshold) \
                    and (abs(closest_translation_x) < self.convergence_translation_threshold) \
                    and (abs(closest_translation_y) < self.convergence_translation_threshold):
                print('Converged!')
                break

        return self.points


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
