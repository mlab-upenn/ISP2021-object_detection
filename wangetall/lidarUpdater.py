import numpy as np
from scipy.sparse import block_diag

from cluster import Cluster
from icp import ICP
from jcbb_Cartesian import JCBB
from helper import Helper


class lidarUpdater:
    def __init__(self):
        self.cl = Cluster()
        self.jcbb = JCBB()
        self.ICP = ICP()
        self.Updater = Updater()

    def clean_up_states(self):
        pass

    def forward(self, xt, P, dt):
        """Propagates forward all tracks 
        based on current transition model"""

        F = self.calc_Fs(xt, dt)
        xt = F@xt #get index of xt's
        xt = xt.T

        Q = self.calc_Qs(xt, dt)

        P = None#a bunch of stuff, gotta index stuff.
        
        return xt, P


    def merge_tracks(self):
        pass

    def update(self, x, P, dt, data):
        xt  = x["xt"]
        xs = x["xs"]
        xp = x["xp"]
        self.clean_up_states()
        xt, P = self.forward(xt, P, dt)
        self.associate_and_update(xp, xt, xs, P, data)
        self.merge_tracks()

    def associate_and_update(self, xp, xt, xs, P, data):
        #DATA ASSOCIATION *IS* THE KALMAN MEASUREMENT UPDATE!
        ### STEP 1: COURSE LEVEL ASSOCIATION
        #1. Cluster: EMST-EGBIS
        #   1a. Compute EMST over collection of points
        #   1b. Throw the EMST as input graph structure to EGBIS to compute clusters
        #   1c. Edge weights of EMST (euclidean distances between points) are taken directly
            #   as dissimilarity measure

        points_x = ranges*cos(angles)
        points_y = ranges*cos(angles)
        points = np.vstack((points_x, points_y)).T
        clusters = cl.cluster(data)

        #First, do the static points.
        P_static_sub = ?? #grab appropriate submatrix from P
        initial_association = #output of ICP that's associated with static map
        boundary_points = ??
        self.jcbb.assign_values(xs, static_cluster, track, P_static_sub, True, psi)
        association = self.jcbb.run(cluster, initial_association, boundary_points)

        self.updater.assign_values(xp, P_static_sub, xs, association, static=True)
        xp_updated, P_static_sub = self.updater.run()
        
        #
        P = ?? #update P with updated static submatrix
        xp = #some function of xp_updated (updated xp), and all xp points that weren't associated
            #these points are used to extend xp.
        #


        #then, do dynamic tracks
        for idx, track in enumerate(dynamic_tracks):
            xt_sub =xt[idx]  #grab appropriate submatrix from xt
            P_sub = ?? #grab appropriate submatrix from P
            initial_association = #output of ICP that's associated with this track
            boundary_points = ??

            self.jcbb.assign_values(xs, static_cluster, xt_sub, P_sub, False, psi)
            association = self.jcbb.run(cluster, initial_association, boundary_points)

            self.updater.assign_values(xt_sub, P_sub, xs, association, static=False)
            
            xt_updated, P_sub = self.updater.run()

            #
            P = ?? #update P with updated dynamic submatrix

            xt[idx] = #some function of xt_updated, and all xt points that weren't associated
                        #these points are used to extend xt[idx].
            #


        return xp, xt, P, remaining_clusters #need to feed remaining clusters into initialize and update 
        


    def compute_H(self):
        pass

    def calc_Fs(self, xt, dt):
        F = np.zeros((6,6))
        F[0:3,0:3] = np.eye(3)
        F[0:3,3:] = dt*np.eye(3)
        F[3:, 3:] = np.eye(3)
        matrices = tuple([F for i in range(len(xt))])
        Fs = block_diag(matrices)
        return Fs

    def calc_Qs(self, xt, dt):
        V = self.calc_V()
        Q = np.zeros((6,6))
        Q[0:3,0:3] = (dt**3/3)*V
        Q[0:3,3:] = (dt**2/2)*V
        Q[3:,0:3] = (dt**2/2)*V
        Q[3:,3:] = dt*V
        matrices = tuple([Q for i in range(len(xt))])
        Qs = block_diag(matrices)
        return Qs

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
    def assign_values(self, x, P, xs, associated_points, static):
        self.x = x
        self.P = P
        self.xs = xs
        self.associated_points = associated_points
        self.static = static

    def run(self):
        R = self.calc_R(self.associated_points)
        g, G = self.calc_g_and_G(self.associated_points)
        H = self.calc_Jacobian_H(g, G, self.associated_points)
        K = self.compute_Kalman_gain(H, self.P, R)

        temp = np.eye(??)-K@H
        P = temp@P@temp.T+K@R@K.T
        x = x+K@(y-H)

        return x, P

    def calc_R(self, associated_points):
        #https://dspace.mit.edu/handle/1721.1/32438#files-area
        R_indiv = np.array([[0.1, 0], [0,0.1]])
        R_matrices = tuple([R_indiv for i in range(len(associated_points))])
        Rs = block_diag(*R_matrices)

        ###Make 3D? Stack Rs on top of each other for every dynamic object
        return Rs

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
        phi = self.xt[??]
        gamma = self.xt[??]
        delta = self.xt[??]

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

