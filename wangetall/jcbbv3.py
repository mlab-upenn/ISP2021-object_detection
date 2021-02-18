import numpy as np
import scipy as sp
from scipy.linalg import block_diag

from helper import Helper


#https://github.com/Scarabrine/EECS568Project_Team2_iSAM/blob/master/JCBB_R.m

class JCBB:
    def __init__(self):
        pass
    
    def run(self, cluster, initial_association, scan_data, xs, boundary_points, track, P):
        # track is vector [gamma_i, delta_i, phi_i, gamma_dot_i, delta_dot_i, phi_dot_i]
        #initial association as 1D vector. Indices of 
        # vector correspond to indices of lidar scan datapoints, 
        # and values in vector correspond to indices of track datapoints
        #unassociated datapoints are replaced with NaN maybe?

        # megacluster = self.combine_clusters(clusters) #
        individual_compatibilities = self.compute_compatibility(scan_data, boundary_points, xs, track, P)
        pruned_associations = self.prune_associations(initial_association, individual_compatibilities)
        JNIS = self.calc_JNIS(pruned_associations, scan_data, xs, track, boundary_points, P)
        
        while True:
            JNIS_delta = 0
            curr_association = None
            for index, _ in enumerate(pruned_associations):
                curr_pairing = pruned_associations[index]
                pruned_associations[index] = np.nan
                JNIS_new = self.calc_JNIS(initial_association, scan_data, xs, track, boundary_points, P)
                JNIS_new_delta = JNIS-JNIS_new
                dof = np.count_nonzero(~np.isnan(pruned_associations))
                chi2 = sp.stats.chi2.ppf(0.95, df=dof)
                if JNIS_new_delta > JNIS_delta:
                    JNIS_delta = JNIS_new_delta
                    curr_association = pruned_associations
                    pruned_associations[index] = curr_pairing
            if JNIS-JNIS_delta <= chi2:
                minimal_association = curr_association
                JNIS = JNIS-JNIS_delta
                break
            else:
                pruned_associations = curr_association
        unassociated_measurements = minimal_association[np.isnan(minimal_association)]
        unassociated_matrix = np.zeros((2, unassociated_measurements.shape[0]))
        unassociated_matrix[0] = unassociated_measurements
        unassociated_matrix[1] = np.nan
        compat_boundaries = {}

        for measurement in unassociated_measurements:
            compat_boundaries[measurement] = individual_compatibilities[measurement,:]
        assigned_associations = self.branch_and_bound(unassociated_matrix, compat_boundaries, association, scan_data, xs, track, boundary_points, P)
    
        return assigned_associations
    
    def branch_and_bound(self, unassociated_matrix, compat_boundaries, association, scan_data, xs, track, boundary_points, P):
        best_JNIS = np.inf
        explored = set() #consists of tuples of (level, boundary_point)
        
        
        def DFS(level, explored, boundary_point, association, best_JNIS,scan_data, xs, track, boundary_points, P):
            explored.add(level, boundary_point)
            avail_boundaries = compat_boundaries[level]
            for next_boundary in avail_boundaries:
                if (level+1, next_boundary) not in explored:
                    test_association = association[:]
                    test_association[0,level+1] = next_boundary

                    #maybe for a more accurate JNIS calc, do I need to combine this association with the previous one?
                    JNIS = self.calc_JNIS(test_association, scan_data, xs, track, boundary_points, P)
                    joint_compat = self.check_compat(JNIS, DOF =np.count_nonzero(~np.isnan(test_association)))
                    if joint_compat and JNIS < best_JNIS:
                        best_JNIS = JNIS  
                        DFS(level+1, explored, next_boundary, test_association, best_JNIS, scan_data, xs, track, boundary_points, P)
            return association

        assigned_association = DFS(1, explored, None, unassociated_matrix, best_JNIS, scan_data, xs, track, boundary_points, P)

        return assigned_association


    def check_compat(self, JNIS, DOF):
        chi2 = sp.stats.chi2.ppf(0.95, df=DOF)
        return JNIS <= chi2
 
    def compute_compatibility(self, scan_data, boundary_points, xs, track, P):
        #returns MxN matrix of compatibility boolean
        individual_compatabilities = np.zeros((scan_data.shape[0], boundary_points.shape[0]))
        chi2_val = sp.stats.chi2.ppf(0.95, df=2)
        for i in range(scan_data.shape[0]):
            association = np.zeros((2, boundary_points.shape[0]))
            association[0] = i
            association[1] = boundary_points
            JNIS = self.calc_JNIS(association, scan_data[i], xs, track, boundary_points, P, indiv=True)
            individual_compatabilities[i, np.where(JNIS<chi2_val)] = 1
        return individual_compatabilities


    def prune_associations(self, associations, individual_compatibilies):
        #check if row in individual compatibilities contains any True's
        null_rows = np.where(~individual_compatibilies.any(axis=1))[0]
        associations[1, null_rows] = np.nan
        #find duplicates

        mults = np.count_nonzero(individual_compatibilies, axis = 1)
        associations[1, mults] = np.nan
        return associations

        
    def calc_JNIS(self, association, scan_data, xs, track, boundary_points, P, indiv= False):
        #want JNIS to output vector of JNIS's if individual
        #want single JNIS if joint.
        associated_points = boundary_points[~np.isnan(association[1])]
        h = self.calc_h(xs, track, associated_points, indiv)
        R = self.calc_R(associated_points, indiv)
        H = self.calc_Jacobian_H(xs, associated_points, indiv)
        S = self.calc_S(H,P,R, indiv)
        if indiv:
            z_hat = scan_data[association[0][~np.isnan(association[1])]].T
            a = (z_hat-h)
            b = np.linalg.inv(S)
            JNIS = np.einsum('ki,kij,kj->k', a, b, a)
        else:
            z_hat = scan_data[association[0][~np.isnan(association[1])]].flatten()
            JNIS = (z_hat-h).T@np.linalg.inv(S)@(z_hat-h)
        return JNIS

    def calc_R(self, associated_points, indiv):
        #https://dspace.mit.edu/handle/1721.1/32438#files-area
        R_indiv = np.array([[0.2, 0], [0,0.2]])
        if indiv:
            R_stacked = np.zeros((len(associated_points, 2,2)))
            R_stacked[:] = R_indiv
            return R_stacked
        else:
            R_matrices = tuple([R_indiv for i in range(len(associated_points))])
            Rs = block_diag(*R_matrices)
            return Rs

    def calc_h(self, xs, track, associated_points, indiv):
        h = np.zeros((associated_points.shape[0], 2))
        alpha = xs["alpha"]
        beta = xs["beta"]
        alpha_beta_arr = np.array([alpha, beta])
        psi= 15 #how to get current psi from rotating sensor? or maybe time stamped psi??
        R_psi = Helper.compute_rot_matrix(psi)
        phi = track[2]#some index
        gamma = track[0]
        delta = track[1]
        R_phi = Helper.compute_rot_matrix(phi)
        for index, point in enumerate(associated_points): 
            x = point[0]
            y = point[1]
            h[index] = R_psi.T@(R_phi@np.array([x, y])+np.array([gamma, delta])-alpha_beta_arr)
        if indiv:
            return h.flatten()
        else:
            return h

    def calc_S(self, H, P, R, indiv):
        #check page 12 of Oxford paper
        if indiv:
            P_stacked = np.zeros((R.shape[0], 2, 2))
            P_stacked[:] = P
            temp = np.einsum('ijk,ikl->ijl', H, P)
            S = np.einsum('ijk,ilk->ijl', temp, H)+R
        else:
            S = H@P@H.T + R
        return S

    def calc_g_and_G(self, xs, associated_points):
        """inputs: xs, measured laserpoint
        
        xs is dict of measurements with xs["alpha"] = const, xs["beta"] = const maybe?
        
        measured_laserpoint is 2d matrix with one col of angles, one col of x coords, one col of y coords 
        where psi is the current rotation angle

        """

        g = np.zeros((associated_points.shape[0], 2))
        G  = np.zeros((associated_points.shape[0]*2, 2))

        alpha = xs["alpha"]
        beta = xs["beta"]
        R_pi_by_2 = Helper.compute_rot_matrix(np.pi/2)
        G_matrices = []
        phi = 15
        R_phi = Helper.compute_rot_matrix(phi)
        #naive way-- with for loops. need to think how to get rid of.
        for index, point in enumerate(associated_points):
            g[index] = R_phi.T @ np.array([point[1], point[2]]- [alpha, beta]).T
            G_matrices.append(-R_phi.T-R_pi_by_2@g[index])
        
        G = block_diag(tuple(G_matrices))

        return g, G



    def calc_Jacobian_H(self, xs, associated_points, indiv):
        g, G = self.calc_g_and_G(xs, associated_points)
        H = np.zeros((g.shape))
        U = self.calc_U(g, len(associated_points))

        H = U.T @ G
        if indiv:
            return H.T[::2].reshape(len(associated_points, 2, 1))
        else:
            return H.T[::2]

    def calc_U(self, f, num_tiles):
        r = np.sqrt(f[:,0]**2+f[:,1]**2)
        U_indiv = (np.array([[r*f[:,0], r*f[:,1]],[-f[:,1], f[:,0]]]))/r**2

        return np.tile(U_indiv, (num_tiles,1))
