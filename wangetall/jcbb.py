import numpy as np
# from scipy.stats import chi2_contingency #want chi_d,alpha where d = 2
from helper import Helper
from scipy.sparse import block_diag
from queue import LifoQueue
#DISCLAIMER: THIS IS SUPER DUPER ROUGH CODE. IT WOULDN'T RUN EVEN IF YOU PAID IT.
#Note to engineer: DON'T FORGET TO IMPLEMENT THE JCBB-REFINE VERSION. THIS IS BASE JCBB.
class JCBB:
    """Given a set of clusters associated with a certain track,
    (or the static background), JCBB will execute fine level 
    association to match points in the clusters to points in the track."""
    
    #Three criteria for feasible association between cluster and track:
    #1. One to one mapping between all measurements and boundary points.
    #2. Each matching of a measrement to a boundary point is individually compatible
    #3. Overall data association is jointly compatible

    def __init__(self):
        # self.chi_threshold = 5.991 #chi squared threshold for 0.05 alpha, 2 d
        
    def calc_U(self, U, f):
        U[::2] = np.array([np.sqrt(f[:,0]**2+f[:,1]**2), np.arctan2(f[:,1], f[:,0])]).T
        return U
    def run(self, H, P, R):
        frontier = LifoQueue()
        megacluster = self.merge_clusters(clusters)

            #1. Individual compatibility (cull)
            #how do we formulate this as a seearch problem?

            #2. Joint compatibility
            H = self.calc_Jacobian_H(xs, point_matrix)
            R = self.calc_R()
            ##how to calc P??
            h = self.calc_h(xs, point_matrix)
            S = self.calc_s(H, P, R)
            association = cluster["association"] #arbitrary dict assumption rn
            D = self.calc_D(association,h, S)


    def calc_R(self):
        #https://dspace.mit.edu/handle/1721.1/32438#files-area
        R_indiv = ??
        R_matrices = tuple([R_indiv for i in range(len(??))])
        Rs = block_diag(R_matrices)
        return Rs

    def calc_h(self, xs, point_matrix):
        h = np.zeros((point_matrix.shape[0], 2))
        alpha = xs["alpha"]
        beta = xs["beta"]
        alpha_beta_arr = np.array([alpha, beta])
        psi= ?? #how to get current psi from rotating sensor? or maybe time stamped psi??
        R_psi = Helper.compute_rot_matrix(psi)

        for index, point in enumerate(point_matrix): 
            phi = point_matrix[??]#some index
            x = point_matrix[??]
            y = point_matrix[??]
            gamma = point_matrix[??]
            delta = point_matrix[??]
            R_phi = Helper.compute_rot_matrix(phi)
            h[index] = R_psi.T@(R_phi@np.array([x, y])+np.array([gamma, delta])-alpha_beta_arr)

        return h




    def calc_g_and_G(self, xs, point_matrix):
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
        G_matrices = []
        #naive way-- with for loops. need to think how to get rid of.
        for index, point in enumerate(point_matrix):
            R = Helper.compute_rot_matrix(point[0])
            g[index] = R.T @ np.array([point[1], point[2]]- [alpha, beta]).T
            G_matrices.append(-R.T-R_pi_by_2@g[index])
        
        G = block_diag(tuple(G_matrices))

        return g, G



    def calc_Jacobian_H(self, xs, point_matrix):
        #Want to compute H_indiv for every pair of datapoitns
        g, G = self.calc_g_and_G(xs, point_matrix)
        H = np.zeros((g.shape))
        U = np.zeros((G.shape[0], 2))
        U = self.calc_U(U, g)

        H = U.T @ G
        return H.T[::2]

    def calc_S(self, H, P, R):
        #check page 12 of Oxford paper
        S = H@P@H.T + R
        return S


    def calc_D(self, z, h, S):
        #D counts nmber of assigned measurements. 
        # Since many feasible associations w/ max assigned, search
        #to get lowest jNIS
        #Generates "joint normalized innovation squared-- jNIS"
        jNIS=(z-h).T@np.linalg.inv(S)@(z-h)
        D = jNIS <= self.chi_threshold
        return D,jNIS

