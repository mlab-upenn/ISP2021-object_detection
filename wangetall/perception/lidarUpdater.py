import numpy as np
from scipy.linalg import block_diag

from perception.cluster import Cluster
from coarse_association import Coarse_Association
from perception.jcbb import JCBB
from perception.helper import Helper
from perception.init_and_merge import InitAndMerge
from cleanupstates import CleanUpStates

import cv2
import sys
import matplotlib.pyplot as plt
class lidarUpdater:
    def __init__(self):
        self.jcbb = JCBB()
        self.Updater = Updater()
        self.InitAndMerge = InitAndMerge()
        # self.clean_up_states = CleanUpStates()
        self.num_beams = 1080
        self.fov = 4.7
        self.theta_init = np.linspace(-self.fov/2., self.fov/2., num=self.num_beams)
        self.i = 0
        self.i2 = 0
        self.frame_no = 0



    def update(self, dt, data, state):
        self.state = state
        self.theta = self.theta_init+self.state.xs[2] #do I need to correct for current heading?
        # self.theta = self.theta_init
        self.polar_laser_points = np.zeros((len(data), 2))
        self.polar_laser_points[:,0] = data
        self.polar_laser_points[:,1] = self.theta

        # self.clean_up_states.run()
        self.forward(dt)
        x, y = Helper.convert_scan_polar_cartesian(np.array(data), self.theta)
        self.laserpoints= np.vstack((x, y)).T

        # self.state.laserpoints = laserpoints
        new_tracks = self.associate_and_update(data, dt)
        for key, points in new_tracks.items():
            idx = self.state.create_new_track(self.laserpoints, points)

        tracks_to_init_and_merge = []
        # print("to init: {}".format(tracks_to_init_and_merge))
        for track_id, track in self.state.dynamic_tracks.items():
            # print("Track id {}, num_viewings {}".format(track_id, track.num_viewings))
            if track.num_viewings == track.mature_threshold:
                tracks_to_init_and_merge.append(track_id)
        if len(tracks_to_init_and_merge) > 0:
            self.InitAndMerge.run(tracks_to_init_and_merge, self.state)
        self.frame_no = self.frame_no + 1
        self.prev_laserpoints = self.laserpoints

    def clean_up_states(self):
        pass

    def forward(self, dt):
        """Propagates forward all tracks
        based on current transition model"""
        F = self.calc_F(dt)
        Q = self.calc_Q(dt)

        for id, track in self.state.dynamic_tracks.items():
            track.kf.F = F
            track.kf.Q = Q
            track.kf.predict()

        #Static background doesn't move, so no need for propagation step...
        # self.state.static_background.kf.F = F
        # self.state.static_background.kf.Q = Q
        # self.state.static_background.kf.predict()


    def associate_and_update(self, data, dt):

        # for key in clusters.keys():
        #     selected_points = self.laserpoints[clusters[key]]
        #     plt.scatter(selected_points[:,0], selected_points[:,1])
        # plt.savefig("output_plots/cluster_idx{}.png".format(self.i2))
        # self.i2 += 1
        #First, do the static points.
        static_association, static_point_pairs, dynamic_association, dynamic_point_pairs, new_tracks = Coarse_Association().run(self.laserpoints, self.state)
        #check how the dynamic point pairs is working... don't want just rough associations for all the dynamic tracks.
        if len(static_point_pairs) > 0:
            # print("Stat point pairs {}".format(static_point_pairs.size))
            P_static_sub = self.state.static_background.kf.P
            tgt_points = []
            for key, value in static_association.items():
                tgt_points = tgt_points+value
            print("Tgt pts shape {}".format(len(tgt_points)))
            pairs = np.array([*static_point_pairs]).T
            print("pairs shape {}".format(pairs.shape))
            initial_association = np.zeros((2, len(tgt_points)))
            initial_association[0] = np.arange(len(tgt_points))
            #tiebreaker?
            initial_association[1, pairs[:,0]] = pairs[:,1]
            self.jcbb.assign_values(xs = self.state.xs, scan_data = self.polar_laser_points[tgt_points], track=None, P = P_static_sub, static=True, psi=self.state.xs[2])
            association = self.jcbb.run(initial_association, self.state.static_background.xb)

            pairings = association[:,~np.isnan(association[1])]
            update_x, update_y = Helper.convert_scan_polar_cartesian_joint(self.polar_laser_points[pairings[0].astype(int)])
            update_points = np.vstack((update_x, update_y)).T +self.state.xs[0:2]
            self.state.static_background.xb[pairings[1].astype(int)] = update_points

            new_pts = set(tgt_points)-set(pairings[0].astype(int))
            new_pts_x, new_pts_y = Helper.convert_scan_polar_cartesian_joint(self.polar_laser_points[list(new_pts)])
            new_pts = np.vstack((new_pts_x, new_pts_y)).T+self.state.xs[0:2]
            self.state.static_background.xb = np.concatenate((self.state.static_background.xb, new_pts))
        #then, do dynamic tracks
        for track_id, dyn_association in dynamic_association.items():
            if dyn_association != {}:
                track = self.state.dynamic_tracks[track_id]
                # print("Track id {}, Track boundary std {}".format(track_id, np.std(track.xp, axis = 0)))
                print("track:",track)
                track.update_num_viewings()
                tgt_points = []
                for value in dyn_association.values():
                    tgt_points = tgt_points+value
                print(dynamic_point_pairs)
                pairs = np.array([*dynamic_point_pairs[track_id]])
                initial_association = np.zeros((2, len(tgt_points)))
                initial_association[0] = np.arange(len(tgt_points))
                initial_association[1, pairs[:,0]] = pairs[:,1]

                self.jcbb.assign_values(xs = self.state.xs, scan_data = self.polar_laser_points[tgt_points], track = track.kf.x, P = track.kf.P[0:2,0:2], static=False, psi=self.state.xs[2])
                # if track.id == 1:
                #     scan_x, scan_y = Helper.convert_scan_polar_cartesian_joint(self.polar_laser_points[tgt_points])
                #     plt.figure()
                #     plt.scatter(scan_x, scan_y, c="b", marker="o", alpha = 0.5, label="Scan Data")
                #     plt.scatter(track.xp[:,0]+track.kf.x[0], track.xp[:,1]+track.kf.x[1], c="orange", marker="o", alpha = 0.1, label="Boundary Points")
                #     plt.savefig("output_plots/{}.png".format(self.i))
                #     self.i += 1

                association = self.jcbb.run(initial_association, track.xp)
                # sys.exit()
                association[0] = tgt_points
                pairings = association[:,~np.isnan(association[1])]
                if pairings.shape[1] >= 3: #need 3 points to compute rigid transformation
                    self.Updater.assign_values(track, association, self.state, static=False)

                    selected_bndr_pts = track.xp[pairings[1].astype(int)]+track.kf.x[0:2]
                    selected_scan_pts = self.polar_laser_points[pairings[0].astype(int)]

                    selected_scan_x, selected_scan_y = Helper.convert_scan_polar_cartesian_joint(selected_scan_pts)
                    selected_scan_cartesian = np.vstack((selected_scan_x, selected_scan_y)).T+self.state.xs[0:2]
                    M = cv2.estimateAffinePartial2D(selected_bndr_pts, selected_scan_cartesian)
                    T = M[0]
                    angle= np.arctan2(T[1,0], T[0,0])
                    measurement = np.zeros((6))
                    measurement[0] = track.kf.x[0]+T[0,2]
                    measurement[1] = track.kf.x[1]+T[1,2]
                    measurement[2] = (track.kf.x[2]+angle)%np.pi
                    measurement[3] = T[0,2]/dt
                    measurement[4] = T[1,2]/dt
                    measurement[5] = angle/dt
                    if track.id == 1:
                        print("T {}".format(T))
                        print("Measurement {}".format(measurement))
                    self.Updater.run(measurement)

            # track.xp = np.append(track.xp, unassociated_boundarypts = ??) #add to track

        return new_tracks #need to feed remaining clusters into initialize and update


    def calc_F(self, dt):
        F = np.zeros((6,6))
        F[0:3,0:3] = np.eye(3)
        F[0:3,3:] = dt*np.eye(3)
        F[3:, 3:] = np.eye(3)
        return F

    def calc_Q(self, dt):
        V = self.calc_V()
        Q = np.zeros((6,6))
        Q[0:3,0:3] = (dt**3/3)*V
        Q[0:3,3:] = (dt**2/2)*V
        Q[3:,0:3] = (dt**2/2)*V
        Q[3:,3:] = dt*V
        return Q

    def calc_V(self):
        #supposed to be a 3x3 covariance matrix for the zero mean continuous linear and angular white noise acceleration
        #supposed to be found here: https://wiki.dmdevelopment.ru/wiki/Download/Books/Digitalimageprocessing/%D0%9D%D0%BE%D0%B2%D0%B0%D1%8F%20%D0%BF%D0%BE%D0%B4%D0%B1%D0%BE%D1%80%D0%BA%D0%B0%20%D0%BA%D0%BD%D0%B8%D0%B3%20%D0%BF%D0%BE%20%D1%86%D0%B8%D1%84%D1%80%D0%BE%D0%B2%D0%BE%D0%B9%20%D0%BE%D0%B1%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%BA%D0%B5%20%D1%81%D0%B8%D0%B3%D0%BD%D0%B0%D0%BB%D0%BE%D0%B2/Estimation%20with%20Applications%20to%20Tracking%20and%20Navigation/booktext@id89013302placeboie.pdf
        #but I can't find it.

        #I think it's a 3x3 diagonal...
        sigma_a = 0.1
        V = np.eye(3)*sigma_a

        return V


class Updater:
    """Can I do this in one calculation for all objects? At least
    all dynamic objects at a time?"""
    """Maybe. Explore using 3D block diagonals?"""


    """Takes in associated output boundary points. What do we do about
    boundary points that don't have association?

    Returns updated state and covariance per object (static and dynamic)"""

    def __init__(self):
        pass
    def assign_values(self, track,associated_points, state, static):
        self.state = state
        self.track = track
        self.xs = self.state.xs
        # self.psi = self.state.xs[2]
        self.psi = 0

        self.associated_points = associated_points
        #HOW DO WE USE ASSOCIATED POINTS AS MEASUREMENT?
        self.static = static

    def run(self, measurement):
        R = self.calc_R(self.associated_points)
        # g, G = self.calc_g_and_G(self.associated_points)
        # H = self.calc_Jacobian_H(x, g, G)

        #May need to come up with custom measurement model for the
        #special measurement we created..
        self.track.kf.update(measurement, self.calc_Hj, self.calc_hx, R)

    def calc_Hj(self, x):
        return np.eye(6)

    def calc_hx(self, x):
        return x



    def calc_R(self, associated_points):
        #https://dspace.mit.edu/handle/1721.1/32438#files-area
        R = np.eye(6)*0.1
        return R

    # def compute_Kalman_gain(self, H, P, R):
    #     K = P@H.T@ np.linalg.inv(H@P@H.T + R)
    #     ##Make 3D? Stack K's on top of each other for every dynamic object.
    #     return K

    # def calc_Jacobian_H(self, x, g, G):
    #     U = self.calc_U(x, g)
    #     H = U.T @ G
    #     return H

    # def calc_U(self, x, g):
    #     r = np.sqrt(g[:,0]**2+g[:,1]**2)
    #     U = (np.array([[r*g[:,0], r*g[:,1]],[-g[:,1], g[:,0]]]))/r**2

    #     U_matrices = tuple([U[:,:,i] for i in range(U.shape[2])])
    #     U =  block_diag(*U_matrices)
    #     ###Make 3D? Stack U's on top of each other for every dynamic object.
    #     return U


    # def calc_g_and_G(self, associated_points):
    #     g = np.zeros((associated_points.shape[0], 2))
    #     #make 3d?
    #     G = np.zeros((associated_points.shape[0]*2, 2))
    #     #make 3d?

    #     alpha = self.xs[0]
    #     beta = self.xs[1]
    #     alpha_beta_arr = np.array([alpha, beta])
    #     phi = self.track.kf.x[0]
    #     gamma = self.track.kf.x[2]
    #     delta = self.track.kf.x[1]

    #     ##Make 2D? have phi, gamma, delta be 2D matrices for every object

    #     R_pi_by_2 = Helper.compute_rot_matrix(np.pi/2)
    #     R_psi = Helper.compute_rot_matrix(self.psi)
    #     R_phi = Helper.compute_rot_matrix(phi)
    #     #naive way-- with for loops. need to think how to get rid of.
    #     for index, point in enumerate(associated_points):
    #         x = point[0]
    #         y = point[1]

    #         if self.static:
    #             g[index] = R_psi.T @ np.array(np.array([x, y])- alpha_beta_arr).T
    #         else:
    #             g[index] = R_psi.T@(R_phi@np.array([x, y])+np.array([gamma, delta])-alpha_beta_arr)

    #         G[index*2:index*2+2] = -R_psi.T-R_pi_by_2@g[index]
    #     return g, G
