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
from skimage import transform
import sys
import matplotlib.pyplot as plt
import os
import datetime as dt
import logging
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

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
            logging.info("Created new Track {}".format(idx))
        end = timer()
        print("Elapsed CREATE_NEW_TRACK = %s s" % round(end - start, 2))

        tracks_to_init_and_merge = []
        # print("to init: {}".format(tracks_to_init_and_merge))
        start = timer()
        for track_id, track in self.state.dynamic_tracks.items():
            logging.info("Track {}, num_viewings {}, last_seen {}".format(track_id, track.num_viewings, track.last_seen))
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
                # if track.id == 26 or track.id == 27:
                #     print("post")
                #     breakpoint()

                self.rotate_points(track)
            if track.last_seen > 1:
                track.update_seen()
                track.kf.P *= 1.1
                logging.info("Increasing Track {} covariance.".format(track.id))


    def rotate_points(self, track):
        tform = transform.EuclideanTransform(
                            rotation=track.kf.x[2],
                            translation = (0,0)
                            )
        track.xp = tform(track.xp)
        
    def associate_and_update(self, data, dt):
        start = timer()
        clusters = self.cl.cluster(self.laserpoints)

        a = np.array([])
        for key in clusters.keys():
            points = self.laserpoints[clusters[key]]

            if(len(points)>100):
                # print("# of points BEFORE RDP:",len(self.laserpoints[clusters[key]]))
                points = simplify_coords(np.array(points, order='c'), 0.02)
                # print("# of points AFTER RDP:",len(points))
            a = np.append(a, points)

        self.laserpoints = a.reshape(-1, 2)
        clusters = self.cl.cluster(self.laserpoints)


        end = timer()
        print("Elapsed CLUSTERING = %s s" % round(end - start, 2))

        #First, do the static points.
        start = timer()
        static_association, static_point_pairs, dynamic_association, dynamic_point_pairs, new_tracks = Coarse_Association(clusters).run(self.laserpoints, self.state)
        end = timer()
        

        print("Elapsed COARSE_ASSOCATION = %s s" % round(end - start, 2))
        if len(static_association) > 0:
            P_static_sub = self.state.static_background.kf.P

            tgt_points = [point for association in list(static_association.values()) for point in association]

            pairs = np.array([*static_point_pairs]).T

            initial_association = np.zeros((2, len(tgt_points)))
            initial_association[0] = np.arange(len(tgt_points))
            xy, x_ind, y_ind = np.intersect1d(pairs[:,0], np.array(tgt_points), return_indices=True)
            initial_association[1, y_ind] = pairs[x_ind, 1]

            self.jcbb.assign_values(xs = self.state.xs, scan_data = self.laserpoints[tgt_points], track=None, P = P_static_sub, static=True, psi=self.state.xs[2])
            association = self.jcbb.run(initial_association, self.state.static_background.xb)
            association[0] = tgt_points
            pairings = association[:,~np.isnan(association[1])]

            if pairings.shape[1] >= 2:
                selected_bndr_pts = self.state.static_background.xb[pairings[1].astype(int)]
                selected_scan_cartesian = self.laserpoints[pairings[0].astype(int)]+self.state.xs[0:2]

                boundaries_adjusted = selected_bndr_pts-np.mean(selected_bndr_pts, axis = 0)
                scans_adjusted = selected_scan_cartesian - np.mean(selected_bndr_pts, axis = 0)
                tform = estimate_transform("euclidean", boundaries_adjusted, scans_adjusted)

                angle= tform.rotation
                print("Translation Norm {}".format(np.linalg.norm(tform.translation)))
                if np.linalg.norm(tform.translation)/dt < 1:
                    self.state.static_background.xb = np.concatenate((self.state.static_background.xb, selected_scan_cartesian))

        #then, do dynamic tracks

        for track_id, dyn_association in dynamic_association.items():
            if dyn_association != {}:
                start = timer()
                track = self.state.dynamic_tracks[track_id]
                tgt_points = np.array([point for association in list(dyn_association.values()) for point in association])
                pairs = np.array([*dynamic_point_pairs[track_id]])

                # if tgt_points.shape[0] > 100:
                #     tgt_points = tgt_points[np.random.choice(len(tgt_points), 100, replace=False)]


                initial_association = np.zeros((2, len(tgt_points)))
                initial_association[0] = np.arange(len(tgt_points))
                xy, x_ind, y_ind = np.intersect1d(pairs[:,0], np.array(tgt_points), return_indices=True)
                initial_association[1, y_ind] = pairs[x_ind, 1]

                scan_data = self.laserpoints[tgt_points]

                boundary_points = track.xp

                self.jcbb.assign_values(xs = self.state.xs, scan_data = scan_data, track = track.kf.x, P = track.kf.P[0:2,0:2], static=False, psi=self.state.xs[2])

                association = self.jcbb.run(initial_association, boundary_points)
                association[0] = tgt_points
                pairings = association[:,~np.isnan(association[1])]
                if pairings.shape[1] < 2:
                    logging.warn("Track {}, num pairings {}".format(track.id, pairings.shape[1]))
                # if track_id == 7:
                #     breakpoint()
                end = timer()
                print("Elapsed JCBB for track_id %s = %s s" % (track_id,round(end - start, 2)))
                percent_associated = pairings.shape[1]/boundary_points.shape[0]

                if pairings.shape[1] >= 2:  #need 2 points to compute rigid transformation
                    #start = timer()
                    track.update_num_viewings()
                    track.reset_seen()

                    self.Updater.assign_values(track, self.state, static=False)

                    selected_bndr_pts = track.xp[pairings[1].astype(int)]+track.kf.x[0:2]
                    selected_scan_cartesian = self.laserpoints[pairings[0].astype(int)]+self.state.xs[0:2]

                    boundaries_adjusted = selected_bndr_pts-np.mean(selected_bndr_pts, axis = 0)
                    scans_adjusted = selected_scan_cartesian - np.mean(selected_bndr_pts, axis = 0)
                    
                    if len(boundaries_adjusted[:,0]) < 10:
                        bndrs_interped_x = np.linspace(np.min(boundaries_adjusted[:,0]), np.max(boundaries_adjusted[:,0]), num = len(boundaries_adjusted[:,0])*10)
                        bndrs_interped_y = np.interp(bndrs_interped_x, boundaries_adjusted[:,0], boundaries_adjusted[:,1])
                        boundaries_adjusted = np.vstack((bndrs_interped_x, bndrs_interped_y)).T

                        scans_interped_x = np.linspace(np.min(scans_adjusted[:,0]), np.max(scans_adjusted[:,0]), num = len(scans_adjusted[:,0])*10)
                        scans_interped_y = np.interp(scans_interped_x, scans_adjusted[:,0], scans_adjusted[:,1])
                        scans_adjusted = np.vstack((scans_interped_x, scans_interped_y)).T

                    
                    
                    tform = estimate_transform("euclidean", boundaries_adjusted, scans_adjusted)

                    angle= tform.rotation
                    measurement = np.zeros((6))
                    measurement[0] = track.kf.x[0]+tform.translation[0] #dx
                    measurement[1] = track.kf.x[1]+tform.translation[1] #dy
                    si = np.sign((track.kf.x[2]+angle))
                    measurement[2] = si*(abs(track.kf.x[2]+angle)%np.pi)
                    measurement[3] = tform.translation[0]/dt+track.kf.x[3] #dx/dt
                    measurement[4] = tform.translation[1]/dt +track.kf.x[4]#dy/dt
                    measurement[5] = 0
                    if tform.rotation > 0.1:
                        breakpoint()
                    logging.info("Track {} received a new measurement! {}".format(track.id, measurement))
                    # if track.id == 3:
                    #     breakpoint()
                    self.Updater.run(measurement)

                    if track.id == 50:

                        sel_scan = self.laserpoints[tgt_points]
                        scan_x = sel_scan[:,0]
                        scan_y = sel_scan[:,1]

                        sel_scan_paired = self.laserpoints[pairings[0].astype(int)]
                        scan_x_paired = sel_scan_paired[:, 0]
                        scan_y_paired = sel_scan_paired[:, 1]

                        bndr_x_paired = track.xp[pairings[1].astype(int), 0]
                        bndr_y_paired = track.xp[pairings[1].astype(int), 1]


                        plt.figure()
                        # plt.xlim(-15,15)
                        # plt.ylim(-15,15)
                        plt.scatter(scan_x+self.state.xs[0], scan_y+self.state.xs[1], c="red", marker="o", alpha = 0.5, label="Scan Data")
                        plt.scatter(track.xp[:,0]+track.kf.x[0], track.xp[:,1]+track.kf.x[1], c="purple", marker="o", alpha = 0.5, label="Boundary Points")
                        
                        plt.scatter(scan_x_paired+self.state.xs[0], scan_y_paired+self.state.xs[1], c="green", marker="o", alpha = 0.5, label="Scan Data")
                        for i in range(scan_x_paired.shape[0]):
                            plt.text((scan_x_paired+self.state.xs[0])[i], (scan_y_paired+self.state.xs[1])[i], str(i))

                        plt.scatter(bndr_x_paired+track.kf.x[0], bndr_y_paired+track.kf.x[1], c="blue", marker="o", alpha = 0.5, label="Boundary Points")
                        for i in range(bndr_x_paired.shape[0]):
                            plt.text((bndr_x_paired+track.kf.x[0])[i], (bndr_y_paired+track.kf.x[1])[i], str(i))


                        plt.scatter(boundaries_adjusted[:,0], boundaries_adjusted[:,1], c="blue", marker="o", alpha = 0.5, label="Boundary Points")
                        plt.scatter(scans_adjusted[:,0], scans_adjusted[:,1], c="green", marker="o", alpha = 0.5, label="Scan Points")
                        np.save("tests/npy_files/boundaries_adjusted.npy", boundaries_adjusted)
                        np.save("tests/npy_files/scans_adjusted.npy", scans_adjusted)
                        plt.legend()
                        plt.show()
                        breakpoint()

                    #end = timer()
                    #print("Elapsed TRANSFORM for 1 track= %s s" % round(end - start, 2))
                    # print("Track {} new state {}".format(track.id, track.kf.x[0:4]))


        return new_tracks #need to feed remaining clusters into initialize and update

    def distance_threshold(self, P):
        base_threshold = 0.8
        uncertainty = np.trace(P)
        C = 1
        threshold = C*base_threshold*uncertainty
        # breakpoint()
        #trace(P); P[0][0]
        return threshold
        
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
    def assign_values(self, track, state, static):
        self.state = state
        self.track = track
        self.xs = self.state.xs
        # self.psi = self.state.xs[2]
        self.psi = 0

        #HOW DO WE USE ASSOCIATED POINTS AS MEASUREMENT?
        self.static = static

    def run(self, measurement):
        R = self.calc_R()
        self.track.kf.update(measurement, self.calc_Hj, self.calc_hx, R)

    def calc_Hj(self, x):
        return np.eye(6)

    def calc_hx(self, x):
        return x

    def calc_R(self):
        #https://dspace.mit.edu/handle/1721.1/32438#files-area
        R = np.eye(6)*0.5
        return R
