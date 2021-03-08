import numpy as np
from scipy.sparse import block_diag

from perception.cluster import Cluster
from coarse_association import Coarse_Association
from perception.jcbb import JCBB
from perception.helper import Helper
from perception.init_and_merge import InitAndMerge
from cleanupstates import CleanUpStates

import cv2
import sys
class lidarUpdater:
    def __init__(self):
        self.cl = Cluster()
        self.jcbb = JCBB()
        self.Updater = Updater()
        self.InitAndMerge = InitAndMerge()
        # self.clean_up_states = CleanUpStates()
        self.num_beams = 1080
        self.fov = 4.7
        self.theta = np.linspace(-self.fov/2., self.fov/2., num=self.num_beams)



    def update(self, dt, data, state):
        self.polar_laser_points = np.zeros((len(data), 2))
        self.polar_laser_points[:,0] = data
        self.polar_laser_points[:,1] = self.theta
        self.state = state
        # self.clean_up_states.run()
        self.forward(dt)
        y, x = Helper.convert_scan_polar_cartesian(np.array(data), self.theta)
        self.laserpoints= np.vstack((x, y)).T
        # self.state.laserpoints = laserpoints
        new_tracks = self.associate_and_update(data, dt)
        for key, points in new_tracks.items():
            idx = self.state.create_new_track(self.laserpoints, points)
        
        tracks_to_init_and_merge = []
        print("to init: {}".format(tracks_to_init_and_merge))
        for track_id, track in self.state.dynamic_tracks.items():
            print("Track id {}, num_viewings {}".format(track_id, track.num_viewings))
            if track.num_viewings == track.mature_threshold:
                tracks_to_init_and_merge.append(track_id)
        if len(tracks_to_init_and_merge) > 0:
            self.InitAndMerge.run(tracks_to_init_and_merge, self.state)

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
        clusters = self.cl.cluster(self.laserpoints)

        #First, do the static points.

        static_association, static_point_pairs, dynamic_association, dynamic_point_pairs, new_tracks = Coarse_Association(clusters).run(self.laserpoints, self.state)
        #check how the dynamic point pairs is working... don't want just rough associations for all the dynamic tracks.
            
        if static_point_pairs.size > 0:
            # print("Stat point pairs {}".format(static_point_pairs.size))
            static_assoc_arr = np.array([*static_point_pairs]).T
            P_static_sub = self.state.static_background.kf.P
            self.jcbb.assign_values(self.state.xs, self.state.static_background.xb, None, P_static_sub, True, self.state.xs[2])
            association = self.jcbb.run(static_assoc_arr, self.state.static_background.xb)
        
        for key in static_association.keys():
            # print("Stat asso {}".format(static_association))
            new_pts_idxs = clusters[key]
            self.state.static_background.xb = np.concatenate(self.state.static_background.xb, self.laserpoints[new_pts_idxs]) 
        #then, do dynamic tracks
        for track_id, dyn_association in dynamic_association.items():
            # print("aiya")
            if dyn_association != {}:
                track = self.state.dynamic_tracks[track_id]
                track.update_num_viewings()
                initial_association = dyn_association#output of ICP that's associated with this track
                tgt_points = []
                for key, value in dyn_association.items():
                    tgt_points = tgt_points+value
                initial_association = np.zeros((2, len(tgt_points)))
                initial_association[0] = np.arange(len(tgt_points))
                initial_association[1] = 0 #temp

                self.jcbb.assign_values(xs = self.state.xs, scan_data = self.polar_laser_points[tgt_points], track = track.kf.x, P = track.kf.P[0:2,0:2], static=False, psi=self.state.xs[2])
                association = self.jcbb.run(initial_association, track.xp)
                association[0] = tgt_points
                if len(association) >= 3: #need 3 points to compute rigid transformation
                    self.Updater.assign_values(track.kf.x, association, static=False)
                    
                    pairings = association[:,~np.isnan(association[1])]
                    selected_bndr_pts = track.xp[pairings[1].astype(int)]
                    selected_scan_pts = data[pairings[0].astype(int)]

                    selected_scan_cartesian = Helper.convert_scan_polar_cartesian_joint(selected_scan_pts)
                    M = cv2.estimateRigidTransform(selected_bndr_pts, selected_scan_cartesian, fullAffine=False)
                    angle= np.atan2(M[1,0], M[0,0])
                    measurement = np.zeros((6))
                    measurement[0] = track.kf.x[0]+M[0,2]
                    measurement[1] = track.kf.x[1]+M[1,2]
                    measurement[2] = track.kf.x[1]+angle
                    measurement[3] = M[0,2]/dt
                    measurement[4] = M[1,2]/dt
                    measurement[5] = angle/dt

                    self.Updater.run(measurement)
            
            # track.xp = np.append(track.xp, unassociated_boundarypts = ??) #add to track

        return new_tracks #need to feed remaining clusters into initialize and update 
        


    def compute_H(self):
        pass

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
        sigma_a = 100
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
    def assign_values(self, track,associated_points, static):
        self.track = track
        self.xs = self.state.xs
        self.associated_points = associated_points
        #HOW DO WE USE ASSOCIATED POINTS AS MEASUREMENT?
        self.static = static

    def run(self, measurement):
        R = self.calc_R(self.associated_points)
        g, G = self.calc_g_and_G(self.associated_points)
        H = self.calc_Jacobian_H(g, G, self.associated_points)

        self.track.kf.R = R
        self.track.kf.H = R
        self.track.kf.update(measurement) #what do we pass in as measurement??


    def calc_R(self, associated_points):
        #https://dspace.mit.edu/handle/1721.1/32438#files-area
        R_indiv = np.array([[0.1, 0], [0,0.1]])

        return R

    def compute_Kalman_gain(self, H, P, R):
        K = P@H.T@ np.linalg.inv(H@P@H.T + R)
        ##Make 3D? Stack K's on top of each other for every dynamic object.
        return K

    def calc_Jacobian_H(self, g, G, associated_points):
        U = self.calc_U(g, len(associated_points))
        H = U.T @ G
        return H

    def calc_U(self, g, num_tiles):
        r = np.sqrt(g[:,0]**2+g[:,1]**2)
        U = (np.array([[r*g[:,0], r*g[:,1]],[-g[:,1], g[:,0]]]))/r**2
            
        U_matrices = tuple([U[:,:,i] for i in range(U.shape[2])])
        U =  block_diag(*U_matrices)
        ###Make 3D? Stack U's on top of each other for every dynamic object.
        return U


    def calc_g_and_G(self, associated_points):
        """inputs: xs, measured laserpoint
        
        xs is dict of measurements with xs["alpha"] = const, xs["beta"] = const maybe?
        
        measured_laserpoint is 2d matrix with one col of angles, one col of x coords, one col of y coords 
        where psi is the current rotation angle
        """


        """How to generalize to compute all objects at once, if
        the length of associated_points is not the same per object?"""
        g = np.zeros((associated_points.shape[0], 2))
        #make 3d?
        G = np.zeros((associated_points.shape[0]*2, 2))
        #make 3d?

        alpha = self.xs["alpha"]
        beta = self.xs["beta"]
        alpha_beta_arr = np.array([alpha, beta])
        phi = self.track.kf.x[0]
        gamma = self.track.kf.x[2]
        delta = self.track.kf.x[1]

        ##Make 2D? have phi, gamma, delta be 2D matrices for every object

        R_pi_by_2 = Helper.compute_rot_matrix(np.pi/2)
        R_psi = Helper.compute_rot_matrix(self.psi)
        R_phi = Helper.compute_rot_matrix(phi)
        #naive way-- with for loops. need to think how to get rid of.
        for index, point in enumerate(associated_points):
            x = point[0]
            y = point[1]

            if self.static:
                g[index] = R_psi.T @ np.array(np.array([x, y])- alpha_beta_arr).T
            else:
                g[index] = R_psi.T@(R_phi@np.array([x, y])+np.array([gamma, delta])-alpha_beta_arr)
            
            G[index*2:index*2+2] = -R_psi.T-R_pi_by_2@g[index]
        return g, G

