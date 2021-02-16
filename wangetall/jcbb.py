import numpy as np
# from scipy.stats import chi2_contingency #want chi_d,alpha where d = 2
from helper import Helper
class JCBB:
    def __init__(self):
        self.chi_threshold = 5.991 #chi squared threshold for 0.05 alpha, 2 d
        
    def calc_U(self, x,y):
        return np.array([np.sqrt(x**2+y**2), np.arctan2(y, x)])
    def run(self, H, P, R):
        H = self.calc_Jacobian_H(xs, point_matrix)
        S = self.calc_s()
        C = self.calc_covariance(H, S, G, P, R)
        D = self.calc_D(C)

    def calc_g_and_G(self, xs, point_matrix,):
        """inputs: xs, measured laserpoint
        
        xs is dict of measurements with xs["alpha"] = const, xs["beta"] = const maybe?
        
        measured_laserpoint is 2d matrix with one col of angles, one col of x coords, one col of y coords 
        where psi is the current rotation angle

        """

        g = np.zeros((point_matrix.shape[0], 2))
        G  = np.zeros((point_matrix.shape[0]*2, 2))

        alpha = xs["alpha"]
        beta = xs["beta"]
        R_pi_by_2 = Helper.compute_rot_matrix(np.pi/2)
        #naive way-- with for loops. need to think how to get rid of.
        for index, point in enumerate(point_matrix):
            R = Helper.compute_rot_matrix(point[0])
            g[index] = R.T @ np.array([point[1], point[2]]- [alpha, beta]).T
            G[index*2] = -R.T-R_pi_by_2@g[index]


        return g, G



    def calc_Jacobian_H(self, xs, point_matrix):
        g, G = self.calc_g_and_G(xs, point_matrix)
        U = np.zeros((g.shape))
        H = np.zeros((g.shape))

        for idx, elem in enumerate(g):
            U[idx] = self.calc_U(elem[0], elem[1])
            H[idx] = U[idx] @ G[idx*2-2, idx*2] #am I indexiing this correctly?
        return H

    def calc_S(self, H, P):
        #check page 12 of Oxford paper
        return S







    def calc_covariance(self, H, S, G, P, R):
        for something in something:
            C = H@P@H.T + G@S@GG.T
        #figure out indexing. 
        return C

    def calc_D(self, h, C):

        return h.T@np.linalg.inv(C)@h < self.chi_threshold



##Input:
##Matrix; rows are H_i = {j_1...j_i} where j_1...j_i are 
#the potential contenders for matched points (so all the currently tracked points?)
#We'll have a separate H_i for each LiDAR datapoint?-- so those will be the rows (one row per point?)
