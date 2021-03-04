import numpy as np
from scipy.sparse import block_diag

from cluster import Cluster
from coarse_association import Coarse_Association
from jcbb_Cartesian import JCBB
from helper import Helper
from init_and_merge import InitAndMerge
from cleanupstates import CleanUpStates

import cv2

class lidarUpdater:
    def __init__(self):
        self.cl = Cluster()
        self.jcbb = JCBB()
        self.Updater = Updater()
        self.InitAndMerge = InitAndMerge()
        self.clean_up_states = CleanUpStates(Q_s_cart, agent_x, agent_y, lidar_range=30.0)

    def update(self, dt, data, state):
        self.state = state
        self.clean_up_states.run()
        self.forward(dt)

        new_tracks = self.associate_and_update(data, dt)
        for cluster in new_tracks:
            self.state.create_new_track(data, cluster)
        self.InitAndMerge.run(new_tracks, self.state)

        #Make repeatedly seen tracks mature?

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

        self.state.static_background.kf.F = F
        self.state.static_background.kf.Q = Q
        self.state.static_background.kf.predict()        


    def associate_and_update(self, data, dt):

        points_x = ranges*cos(angles)
        points_y = ranges*cos(angles)
        points = np.vstack((points_x, points_y)).T
        clusters = self.cl.cluster(data)

        #First, do the static points.
        static_association, dynamic_association, new_tracks = Coarse_Association(clusters).run(points, state)
        P_static_sub = self.state.static_background.kf.P
        static_cluster = ??
        self.jcbb.assign_values(self.state.xs, static_cluster, self.state.static_background.xs, P_static_sub, True, psi)
        association = self.jcbb.run(static_association, self.state.static_background.xb)
       
        if len(association) >= 3:
            pairings = association[:,~np.isnan(asso[1])]
            selected_bndr_pts = tracker.xp[pairings[1].astype(int)]
            selected_scan_pts = data[pairings[0].astype(int)]
            M = cv2.estimateRigidTransform(selected_bndr_pts, selected_scan_pts, fullAffine=False)
            angle= np.atan2(M[1,0], M[0,0])
            measurement = np.zeros((6))
            measurement[0] = tracker.kf.xt[0]+M[0,2]
            measurement[1] = tracker.kf.xt[1]+M[1,2]
            measurement[2] = tracker.kf.xt[1]+angle
            measurement[3] = M[0,2]/dt
            measurement[4] = M[1,2]/dt
            measurement[5] = angle/dt
            self.updater.assign_values(self.state.static_background.kf.xt, P_static_sub, self.state.xs, association, static=True)
            self.updater.run(measurement)
        
        self.state.static_background.xb = np.append(self.state.static_background.xb, unassociated_boundarypts = ??) 
        
        #then, do dynamic tracks
        for idx, track in self.state.dynamic_tracks.items():
            track.update_num_viewings()
            initial_association = dynamic_association[idx]#output of ICP that's associated with this track
            cluster = ??
            self.jcbb.assign_values(self.state.xs, cluster, track.kf.xt, track.kf.P, False, psi)
            association = self.jcbb.run(initial_association, track.xp)


            if len(association) >= 3: #need 3 points to compute rigid transformation
                self.updater.assign_values(track.kf.xt, track.kf.P, association, static=False)
                
                pairings = association[:,~np.isnan(asso[1])]
                selected_bndr_pts = tracker.xp[pairings[1].astype(int)]
                selected_scan_pts = data[pairings[0].astype(int)]

                selected_scan_cartesian = Helper.convert_scan_polar_cartesian(selected_scan_pts)
                M = cv2.estimateRigidTransform(selected_bndr_pts, selected_scan_pts, fullAffine=False)
                angle= np.atan2(M[1,0], M[0,0])
                measurement = np.zeros((6))
                measurement[0] = tracker.kf.xt[0]+M[0,2]
                measurement[1] = tracker.kf.xt[1]+M[1,2]
                measurement[2] = tracker.kf.xt[1]+angle
                measurement[3] = M[0,2]/dt
                measurement[4] = M[1,2]/dt
                measurement[5] = angle/dt

                self.updater.run(measurement)
            
            track.xp = np.append(track.xp, unassociated_boundarypts = ??) #add to track

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

    def run(self):
        R = self.calc_R(self.associated_points)
        g, G = self.calc_g_and_G(self.associated_points)
        H = self.calc_Jacobian_H(g, G, self.associated_points)

        self.track.kf.R = R
        self.track.kf.H = R
        self.track.kf.update() #what do we pass in as measurement??


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
        phi = self.track.kf.xt[0]
        gamma = self.track.kf.xt[2]
        delta = self.track.kf.xt[1]

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

