import numpy as np
from scipy.linalg import block_diag

from perception.cluster import Cluster
from coarse_association import Coarse_Association
from perception.jcbb import JCBB
from perception.helper import Helper
from perception.init_and_merge import InitAndMerge
from cleanupstates import CleanUpStates
import time
from skimage.transform import estimate_transform
import sys
import matplotlib.pyplot as plt
import os
import datetime as dt
import logging



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
        self.state = state
        self.theta = self.theta_init+self.state.xs[2]
        self.polar_laser_points = np.zeros((len(data), 2))
        self.polar_laser_points[:,0] = data
        self.polar_laser_points[:,1] = self.theta
        self.forward()
        x, y = Helper.convert_scan_polar_cartesian(np.array(data), self.theta)
        self.laserpoints= np.vstack((x, y)).T
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='polar')
        # ax.scatter(self.polar_laser_points[:,1], self.polar_laser_points[:,0], c="blue",marker = "o", alpha=0.5, label="Boundary Points")
        # # ax.set_xlim(np.pi, 1.5*np.pi)
        # ax.set_ylim(0, 2)
        # plt.show()
        # breakpoint()


        # plt.figure()
        # plt.scatter(self.laserpoints[:,0], self.laserpoints[:,1], c = "blue", alpha= 0.5)
        # plt.xlim(0,1)
        # plt.ylim(0,1)
        # plt.show()
        # breakpoint()

        self.clean_up_states.run(self.state.xs[0], self.state.xs[1], self.laserpoints, self.state)

        # plt.scatter(self.laserpoints[:,0], self.laserpoints[:,1], c="b", label="test")
        # plt.scatter(self.state.xs[0], self.state.xs[1], c="r", label="car")
        # plt.legend()
        # plt.show()

        # self.state.laserpoints = laserpoints

        new_tracks = self.associate_and_update(data, dt)
        for key, points in new_tracks.items():
            if len(points) > 20:
                idx = self.state.create_new_track(self.laserpoints, points)
                logging.info("Created new Track {}".format(idx))

        tracks_to_init_and_merge = []
        # print("to init: {}".format(tracks_to_init_and_merge))
        for track_id, track in self.state.dynamic_tracks.items():
            logging.info("Track {}, num_viewings {}, last_seen {}".format(track_id, track.num_viewings, track.last_seen))
            if track.num_viewings >= track.mature_threshold:
                tracks_to_init_and_merge.append(track_id)
        if len(tracks_to_init_and_merge) > 0:
            logging.info("Tracks to init and merge {}".format(tracks_to_init_and_merge))
            self.InitAndMerge.run(tracks_to_init_and_merge, self.state)


    def forward(self):
        """Propagates forward all tracks
        based on current transition model"""

        for idx, track in self.state.dynamic_tracks.items():
            if track.last_seen <= track.seen_threshold:
                track.update_seen()
                track.kf.predict()
            if track.last_seen > 1:
                if np.trace(track.kf.P) < 5:
                    track.kf.P *= 2
                    logging.info("Increasing Track {} covariance.".format(track.id))

        #Static background doesn't move, so no need for propagation step...
        # self.state.static_background.kf.F = F
        # self.state.static_background.kf.Q = Q
        # self.state.static_background.kf.predict()


    def associate_and_update(self, data, dt):
        clusters = self.cl.cluster(self.laserpoints)
        # # plt.figure()
        # for key in clusters.keys():
        #     selected_points = self.laserpoints[clusters[key]]
        #     plt.scatter(selected_points[:,0], selected_points[:,1])
        # plt.xlim(-10,0)
        # plt.ylim(-3, 7)
        # # plt.close(fig)
        # plt.savefig("output_plots/cluster_idx{}.png".format(self.i2))
        # plt.clf()

        self.i2 += 1

        #First, do the static points.
        start = time.time()
        static_association, static_point_pairs, dynamic_association, dynamic_point_pairs, new_tracks = Coarse_Association(clusters).run(self.laserpoints, self.state)
        end = time.time()
        if end-start > 1:
            logging.warning("Long coarse time. Time: {}".format(end-start))
        #check how the dynamic point pairs is working... don't want just rough associations for all the dynamic tracks.
        # if len(static_association) > 0:
        #     # print("Stat point pairs {}".format(static_point_pairs.size))
        #     P_static_sub = self.state.static_background.kf.P

        #     tgt_points = [point for association in list(static_association.values()) for point in association]

        #     # print("Tgt pts shape {}".format(len(tgt_points)))
        #     pairs = np.array([*static_point_pairs]).T
        #     # print("pairs shape {}".format(pairs.shape))

        #     initial_association = np.zeros((2, len(tgt_points)))
        #     initial_association[0] = np.arange(len(tgt_points))
        #     #breakpoint()
        #     try:
        #         xy, x_ind, y_ind = np.intersect1d(pairs[:,0], np.array(tgt_points), return_indices=True)
        #     except:
        #         breakpoint()
        #     initial_association[1, y_ind] = pairs[x_ind, 1]

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
                # if track_id == 1:
                #     breakpoint()

                track = self.state.dynamic_tracks[track_id]
                # print("Track id {}, Track boundary std {}".format(track_id, np.std(track.xp, axis = 0)))

                tgt_points = np.array([point for association in list(dyn_association.values()) for point in association])


                pairs = np.array([*dynamic_point_pairs[track_id]])

                # if tgt_points.shape[0] > 100:
                #     tgt_points = tgt_points[np.random.choice(len(tgt_points), 100, replace=False)]
                

                initial_association = np.zeros((2, len(tgt_points)))
                initial_association[0] = np.arange(len(tgt_points))
                _, x_ind, y_ind = np.intersect1d(pairs[:,0], np.array(tgt_points), return_indices=True)
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
                start = time.time()
                self.jcbb.assign_values(xs = self.state.xs, scan_data = scan_data, track = track.kf.x, P = track.kf.P[0:2,0:2], static=False, psi=self.state.xs[2])
                association = self.jcbb.run(initial_association, boundary_points)
                end = time.time()
                logging.info("JCBB time {}".format(end-start))

                # sys.exit()
                association[0] = tgt_points
                pairings = association[:,~np.isnan(association[1])]
                if pairings.shape[1] == 0:
                    logging.warn("Track {}, num pairings {}".format(track.id, pairings.shape[1]))
                # if pairings.shape[1] == 0:
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
                #     plt.scatter(scan_x+self.state.xs[0], scan_y+self.state.xs[1], c="blue", marker="o", alpha = 0.5, label="Scan Data")
                #     plt.scatter(track.xp[:,0]+track.kf.x[0], track.xp[:,1]+track.kf.x[1], c="orange", marker="o", alpha = 0.5, label="Boundary Points")
                #     plt.title("No pairings :( SAD! Cov {}".format(track.kf.P[0,0]))
                #     plt.savefig("output_plots/nopairings_{}.png".format(time.time()))
                    # breakpoint()
                #     # plt.savefig("output_plots/{}.png".format(self.i))
                #     # self.i += 1
                # scan_x, scan_y = Helper.convert_scan_polar_cartesian_joint(self.polar_laser_points[tgt_points])
                # plt.figure()
                # # plt.xlim(-15,15)
                # # plt.ylim(-15,15)
                # plt.scatter(scan_x+self.state.xs[0], scan_y+self.state.xs[1], c="blue", marker="o", alpha = 0.5, label="Scan Data")
                # plt.scatter(track.xp[:,0]+track.kf.x[0], track.xp[:,1]+track.kf.x[1], c="orange", marker="o", alpha = 0.5, label="Boundary Points")
                # plt.title("Num pairings {} :( SAD! Cov {}".format(pairings.shape[1], track.kf.P[0,0]))
                # plt.show()
                percent_associated = pairings.shape[1]/boundary_points.shape[0]
                if percent_associated >= 0.3: #need 3 points to compute rigid transformation
                    track.update_num_viewings()
                    track.reset_seen()


                    selected_bndr_pts = track.xp[pairings[1].astype(int)]+track.kf.x[0:2]
                    selected_scan_pts = self.polar_laser_points[pairings[0].astype(int)]

                    selected_scan_x, selected_scan_y = Helper.convert_scan_polar_cartesian_joint(selected_scan_pts)
                    selected_scan_cartesian = np.vstack((selected_scan_x, selected_scan_y)).T+self.state.xs[0:2]
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
                    measurement = np.zeros((4))
                    measurement[0] = track.kf.x[0]+tform.translation[0]
                    measurement[1] = track.kf.x[1]+tform.translation[1]
                    measurement[2] = tform.translation[0]/dt
                    measurement[3] = tform.translation[1]/dt
                    speed = np.linalg.norm(measurement[2:4])
                    logging.info("Track {} received a new measurement: {}. Speed {}".format(track.id, measurement[2:4], speed))
                    if speed > 20:
                        logging.warn("Track {} is moving quickly. Speed {}".format(track.id, speed))
                    percent_associated = pairings.shape[1]/boundary_points.shape[0]
                    self.Updater.run(track, measurement, percent_associated)
                    # new_pts = set(tgt_points)-set(pairings[0].astype(int))
                    # new_pts_x, new_pts_y = Helper.convert_scan_polar_cartesian_joint(self.polar_laser_points[list(new_pts)])
                    # new_pts = np.vstack((new_pts_x, new_pts_y)).T+self.state.xs[0:2]-track.kf.x[0:2]
                    # track.xp = np.concatenate((track.xp, new_pts))

                    # print("Track {} new state {}".format(track.id, track.kf.x[0:4]))

        return new_tracks #need to feed remaining clusters into initialize and update




class Updater:
    """Takes in associated output boundary points. What do we do about
    boundary points that don't have association?

    Returns updated state and covariance per object (static and dynamic)"""
    def __init__(self):
        pass

    def run(self, track, measurement, percent_associated):
        #May need to come up with custom measurement model for the
        #special measurement we created..
        logging.info("Track {} percent associated {}".format(track.id, percent_associated))
        R = self.calc_R(percent_associated)
        logging.info("Pre update Track {} covariance: {}".format(track.id, track.kf.P[0,0]))
        logging.info("Pre update Track {} vel: {}".format(track.id, track.kf.x[2:4]))
        track.kf.update(measurement, self.calc_Hj, self.calc_hx, R=R)
        track.kf.P *= 1/abs(track.kf.log_likelihood)
        logging.info("Track {} update likelihood: {}".format(track.id, track.kf.log_likelihood))
        logging.info("Post update Track {} vel: {}".format(track.id, track.kf.x[2:4]))
        logging.info("Post update Track {} covariance: {}".format(track.id, track.kf.P[0,0]))
    
    def calc_Hj(self, x):
        return np.eye(4)

    def calc_hx(self, x):
        return x
    
    def calc_R(self, percent_associated):
        R = np.diag([0.2, 0.2, 0.2, 0.2])*(1-percent_associated)+np.diag([0.01, 0.01, 0.01, 0.01])
        # R = np.diag([0.1, 0.1, 0.1, 0.1])*(1-percent_associated)
        assert np.all(R>=0)
        return R
