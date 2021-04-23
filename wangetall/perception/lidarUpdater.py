import numpy as np
from scipy.linalg import block_diag

from perception.cluster import Cluster
from coarse_association import Coarse_Association
from perception.jcbb import JCBB
from perception.helper import Helper
from perception.init_and_merge import InitAndMerge
from cleanupstates import CleanUpStates
from timeit import default_timer as timer
from skimage.transform import estimate_transform
import sys
import matplotlib.pyplot as plt
import os
import datetime as dt
import logging
from simplification.cutil import (
    simplify_coords,
    simplify_coords_idx,
    simplify_coords_vw,
    simplify_coords_vw_idx,
    simplify_coords_vwp,
)


class lidarUpdater:
    def __init__(self):
        self.cl = Cluster()
        self.jcbb = JCBB()
        self.Updater = Updater()
        self.InitAndMerge = InitAndMerge()
        self.num_beams = 1080
        self.fov = 4.7
        self.theta_init = np.linspace(-self.fov/2., self.fov/2., num=self.num_beams)
        self.i = 0
        self.i2 = 0
        self.clean_up_states = CleanUpStates()



    def update(self, dt, data, state):
        start_total = timer()
        self.state = state
        self.theta = self.theta_init+self.state.xs[2] #do I need to correct for current heading?
        # self.theta = self.theta_init
        self.polar_laser_points = np.zeros((len(data), 2))
        self.polar_laser_points[:,0] = data
        self.polar_laser_points[:,1] = self.theta
        self.forward(dt)
        x, y = Helper.convert_scan_polar_cartesian(np.array(data), self.theta)
        self.laserpoints = np.vstack((x, y)).T

        start = timer()
        self.clean_up_states.run(self.state.xs[0], self.state.xs[1], self.laserpoints, self.state)
        end = timer()
        print("Elapsed CLEANUPSTATES = %s s" % round(end - start, 2))
        # self.state.laserpoints = laserpoints

        start = timer()
        new_tracks = self.associate_and_update(data, dt)
        end = timer()
        print("Elapsed ASSOCATE_AND_UPDATE = %s s" % round(end - start, 2))
        start = timer()
        for key, points in new_tracks.items():
            idx = self.state.create_new_track(self.laserpoints, points)
            logging.info("Created new track {}".format(idx))
        end = timer()
        print("Elapsed CREATE_NEW_TRACK = %s s" % round(end - start, 2))

        tracks_to_init_and_merge = []
        # print("to init: {}".format(tracks_to_init_and_merge))
        start = timer()
        for track_id, track in self.state.dynamic_tracks.items():
            logging.info("Track id {}, num_viewings {}, last_seen {}".format(track_id, track.num_viewings, track.last_seen))
            if track.num_viewings >= track.mature_threshold:
                tracks_to_init_and_merge.append(track_id)
        if len(tracks_to_init_and_merge) > 0:
            logging.info("Tracks to init and merge {}".format(tracks_to_init_and_merge))
            self.InitAndMerge.run(tracks_to_init_and_merge, self.state)
        end = timer()
        print("Elapsed INIT_AND_MERGE = %s s" % round(end - start, 2))
        end_total = timer()
        print("ONE STEP TIME = %s s" % round(end_total - start_total, 4))
        print("-----------------------------")


    def forward(self, dt):
        """Propagates forward all tracks
        based on current transition model"""
        F = self.calc_F(dt)
        Q = self.calc_Q(dt)

        for idx, track in self.state.dynamic_tracks.items():
            if track.last_seen <= track.seen_threshold:
                track.update_seen()
                track.kf.F = F
                track.kf.Q = Q
                track.kf.predict()
            if track.last_seen > 1:
                track.update_seen()
                track.kf.P *= 1.1
                logging.info("Increasing Track {} covariance.".format(track.id))
            if track.id == 4:
                logging.warn("Track 4 cov: {}".format(track.kf.P[0,0]))
                logging.warn("Track 4 Kalman Gain: {}".format(track.kf.K))

        #Static background doesn't move, so no need for propagation step...
        # self.state.static_background.kf.F = F
        # self.state.static_background.kf.Q = Q
        # self.state.static_background.kf.predict()


    def associate_and_update(self, data, dt):
        start = timer()
        clusters = self.cl.cluster(self.laserpoints)

        a = np.array([])
        for key in clusters.keys():
            points = self.laserpoints[clusters[key]]
            if(len(points)>100):
                print("# of points BEFORE RDP:",len(self.laserpoints[clusters[key]]))
                points = simplify_coords(np.array(points, order='c'), 0.02)
                print("# of points AFTER RDP:",len(points))
            a = np.append(a, points)
            #self.laserpoints[clusters[key]] = points


        self.laserpoints = a.reshape(-1, 2)
        clusters = self.cl.cluster(self.laserpoints)
        end = timer()
        print("Elapsed CLUSTERING = %s s" % round(end - start, 2))
        # plt.savefig("output_plots/cluster_idx{}.png".format(self.i2))
        # self.i2 += 1

        #First, do the static points.
        start = timer()
        static_association, static_point_pairs, dynamic_association, dynamic_point_pairs, new_tracks = Coarse_Association(clusters).run(self.laserpoints, self.state)
        end = timer()
        print("Elapsed COARSE_ASSOCATION = %s s" % round(end - start, 2))
        #check how the dynamic point pairs is working... don't want just rough associations for all the dynamic tracks.
        if len(static_association) > 0:
            # print("Stat point pairs {}".format(static_point_pairs.size))
            P_static_sub = self.state.static_background.kf.P

            tgt_points = [point for association in list(static_association.values()) for point in association]

            # print("Tgt pts shape {}".format(len(tgt_points)))
            pairs = np.array([*static_point_pairs]).T
            # print("pairs shape {}".format(pairs.shape))

            initial_association = np.zeros((2, len(tgt_points)))
            initial_association[0] = np.arange(len(tgt_points))
            #breakpoint()
            try:
                xy, x_ind, y_ind = np.intersect1d(pairs[:,0], np.array(tgt_points), return_indices=True)
            except:
                breakpoint()
            initial_association[1, y_ind] = pairs[x_ind, 1]

            # print("Scan data static shape{}".format(self.polar_laser_points[tgt_points].shape))
            # breakpoint()

            # self.jcbb.assign_values(xs = self.state.xs, scan_data = self.polar_laser_points[tgt_points], track=None, P = P_static_sub, static=True, psi=self.state.xs[2])
            # association = self.jcbb.run(initial_association, self.state.static_background.xb)
            # association[0] = tgt_points
            # pairings = association[:,~np.isnan(association[1])]
            # update_x, update_y = Helper.convert_scan_polar_cartesian_joint(self.polar_laser_points[pairings[0].astype(int)])
            # update_points = np.vstack((update_x, update_y)).T +self.state.xs[0:2]
            # self.state.static_background.xb[pairings[1].astype(int)] = update_points

            # new_pts = set(tgt_points)-set(pairings[0].astype(int))
            # new_pts_x, new_pts_y = Helper.convert_scan_polar_cartesian_joint(self.polar_laser_points[list(new_pts)])
            # new_pts = np.vstack((new_pts_x, new_pts_y)).T+self.state.xs[0:2]
            # self.state.static_background.xb = np.concatenate((self.state.static_background.xb, new_pts))
        #then, do dynamic tracks

        for track_id, dyn_association in dynamic_association.items():
            if dyn_association != {}:
                start = timer()
                # if track_id == 1:
                #     breakpoint()

                track = self.state.dynamic_tracks[track_id]
                # print("Track id {}, Track boundary std {}".format(track_id, np.std(track.xp, axis = 0)))


                track.update_num_viewings()
                track.reset_seen()

                tgt_points = np.array([point for association in list(dyn_association.values()) for point in association])


                pairs = np.array([*dynamic_point_pairs[track_id]])

                # if tgt_points.shape[0] > 100:
                #     tgt_points = tgt_points[np.random.choice(len(tgt_points), 100, replace=False)]


                initial_association = np.zeros((2, len(tgt_points)))
                initial_association[0] = np.arange(len(tgt_points))
                try:
                    xy, x_ind, y_ind = np.intersect1d(pairs[:,0], np.array(tgt_points), return_indices=True)
                except:
                    breakpoint()
                initial_association[1, y_ind] = pairs[x_ind, 1]

                scan_data = self.polar_laser_points[tgt_points]

                #ok, so rn, there are items in pairs that are not in scan data.
                # if scan_data.shape[0] > 100:
                #     selected_scan_idxs= set(np.random.choice(scan_data.shape[0], 100, replace=False)).union(set(y_ind))
                #     scan_data = scan_data[list(selected_scan_idxs)]

                boundary_points = track.xp
                # if boundary_points.shape[0] > 100:
                #     selected_bndry_idxs= set(np.random.choice(boundary_points.shape[0], 100, replace=False)).union(set(x_ind))
                #     boundary_points = boundary_points[list(selected_bndry_idxs)]

                self.jcbb.assign_values(xs = self.state.xs, scan_data = scan_data, track = track.kf.x, P = track.kf.P[0:2,0:2], static=False, psi=self.state.xs[2])

                association = self.jcbb.run(initial_association, boundary_points)
                # sys.exit()
                association[0] = tgt_points
                pairings = association[:,~np.isnan(association[1])]
                if pairings.shape[1] == 0:
                    logging.warn("Track {}, num pairings {}".format(track.id, pairings.shape[1]))
                # if pairings.shape[1] == 0 and track.id == 4:
                #     np.save("tests/npy_files/xs.npy", self.state.xs)
                #     np.save("tests/npy_files/scan_data.npy", scan_data)
                #     np.save("tests/npy_files/track.npy", track.kf.x)
                #     np.save("tests/npy_files/P.npy", track.kf.P[0:2,0:2])

                #     np.save("tests/npy_files/psi.npy", self.state.xs[2])
                #     np.save("tests/npy_files/initial_association.npy", initial_association)
                #     np.save("tests/npy_files/boundary_points.npy", boundary_points)


                #     scan_x, scan_y = Helper.convert_scan_polar_cartesian_joint(self.polar_laser_points[tgt_points])
                #     plt.figure()
                #     # plt.xlim(-15,15)
                #     # plt.ylim(-15,15)
                #     plt.scatter(scan_x+self.state.xs[0], scan_y+self.state.xs[1], c="red", marker="o", alpha = 0.5, label="Scan Data")
                #     plt.scatter(track.xp[:,0]+track.kf.x[0], track.xp[:,1]+track.kf.x[1], c="purple", marker="o", alpha = 0.5, label="Boundary Points")
                #     plt.show()
                #     breakpoint()
                #     # plt.savefig("output_plots/{}.png".format(self.i))
                #     # self.i += 1
                end = timer()
                print("Elapsed JCBB for track_id %s = %s s" % (track_id,round(end - start, 2)))
                if pairings.shape[1] >= 3: #need 3 points to compute rigid transformation
                    #start = timer()
                    self.Updater.assign_values(track, association, self.state, static=False)

                    selected_bndr_pts = track.xp[pairings[1].astype(int)]+track.kf.x[0:2]
                    selected_scan_pts = self.polar_laser_points[pairings[0].astype(int)]

                    selected_scan_x, selected_scan_y = Helper.convert_scan_polar_cartesian_joint(selected_scan_pts)
                    selected_scan_cartesian = np.vstack((selected_scan_x, selected_scan_y)).T+self.state.xs[0:2]
                    boundaries_centroid = np.mean(selected_bndr_pts, axis = 0)
                    boundaries_adjusted = selected_bndr_pts
                    scans_adjusted = selected_scan_cartesian
                    # if track_id == 1:
                    #     np.save("scan.npy", selected_scan_cartesian)
                    #     np.save("boundaries.npy", selected_bndr_pts)
                    #     sys.exit()
                    tform = estimate_transform("euclidean", boundaries_adjusted, scans_adjusted)
                    # if track_id == 5:
                    #     print("ahhh")
                    #     plt.figure()

                    #     plt.scatter(selected_scan_cartesian[:,0],selected_scan_cartesian[:,1],alpha=0.5, s=4,c="red")
                    #     for i in range(selected_scan_cartesian.shape[0]):
                    #         plt.text(selected_scan_cartesian[i,0], selected_scan_cartesian[i,1], str(i), size = "xx-small")


                    #     plt.scatter(selected_bndr_pts[:,0],selected_bndr_pts[:,1],alpha=0.5, s = 4,c="purple")
                    #     for i in range(selected_bndr_pts.shape[0]):
                    #         plt.text(selected_bndr_pts[i,0], selected_bndr_pts[i,1], str(i), size = "xx-small")
                    #     ### plt.scatter(scans_adjusted[:,0],scans_adjusted[:,1],alpha=0.5, c="red")
                    #     ### plt.scatter(boundaries_adjusted[:,0],boundaries_adjusted[:,1],alpha=0.5, c="purple")
                    #     plt.show()
                    #     breakpoint()

                    # breakpoint()

                    angle= tform.rotation
                    measurement = np.zeros((6))
                    measurement[0] = track.kf.x[0]+tform.translation[0]
                    measurement[1] = track.kf.x[1]+tform.translation[1]
                    measurement[2] = (track.kf.x[2]+angle)%np.pi
                    measurement[3] = tform.translation[0]/dt
                    measurement[4] = tform.translation[1]/dt
                    measurement[5] = angle
                    logging.info("Track {} received a new measurement! {}".format(track.id, measurement[3:5]))
                    self.Updater.run(measurement)
                    #end = timer()
                    #print("Elapsed TRANSFORM for 1 track= %s s" % round(end - start, 2))
                    # print("Track {} new state {}".format(track.id, track.kf.x[0:4]))


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
        R = np.eye(6)*0.5
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
