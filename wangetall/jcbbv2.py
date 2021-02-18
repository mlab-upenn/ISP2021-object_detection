import numpy as np
import scipy as sp

from helper import Helper


#https://github.com/Scarabrine/EECS568Project_Team2_iSAM/tree/master/jcbb-simulation/tools
class JCBB:
    def __init__(self):
        self.chi_threshold = 5.991
    
    def calc_U(self, U, f):
        U[::2] = np.array([np.sqrt(f[:,0]**2+f[:,1]**2), np.arctan2(f[:,1], f[:,0])]).T
        return U

    def run(self, cluster, initial_association, scan_data, track, boundary_points, P):
        # track is vector [gamma_i, delta_i, phi_i, gamma_dot_i, delta_dot_i, phi_dot_i]
        #initial association as 1D vector. Indices of 
        # vector correspond to indices of lidar scan datapoints, 
        # and values in vector correspond to indices of track datapoints
        #unassociated datapoints are replaced with NaN maybe?

        # megacluster = self.combine_clusters(clusters) #
        association = self.check_indiv_compatibility(initial_association, scan_data, xs, track, boundary_points, P)
        association = self.remove_duplicates(association)
        association, JNIS = self.calc_JNIS(initial_association, scan_data, xs, track, boundary_points, P)
        dof = np.count_nonzero(~np.isnan(association))
        chi2 = sp.stats.chi2.ppf(0.95, df=dof)

        #iteratively remove assignment that leads to the greatest jnis reduction till condition 3 is satisfied
        while True:
            JNIS_delta = 0
            curr_association = None
            for index, pairing in enumerate(association):
                curr_pairing = association[index]
                association[index] = np.nan
                association, JNIS_new = self.calc_JNIS(initial_association, scan_data, xs, track, boundary_points, P)
                JNIS_new_delta = JNIS-JNIS_new
                dof = np.count_nonzero(~np.isnan(association))
                chi2 = sp.stats.chi2.ppf(0.95, df=dof)
                if JNIS_new_delta > JNIS_delta:
                    JNIS_delta = JNIS_new_delta
                    curr_association = association
                    association[index] = curr_pairing
            if JNIS-JNIS_delta <= chi2:
                minimal_association = curr_association
                JNIS = JNIS-JNIS_delta
                break
            else:
                association = curr_association
        unassociated_measurements = minimal_association[np.isnan(minimal_association)]
        #Now, unassociated measurements are tried in turn and
        #assigned to the boundary point that gives the lowest jNIS
        #I think we can do this with branch n' bound.
        #How do we get the relevant set of boundary points? 
        # They need to be individually compatible...
        
        #Naive: double for loop.





        
        pass
    def check_indiv_compatibility(self, association, scan_data, xs, track, boundary_points, P):
        #replace non-indiv-compatible associations with NaN
        #scan_data is N by 2 array of range, theta measurements
        z_hat = scan_data[np.arange(association.shape[0])]
        associated_points = boundary_points[~np.isnan(association)]
        h = self.calc_h(xs, track, associated_points)
        R = self.calc_R(associated_points)
        H = self.calc_Jacobian_H(xs, association, indiv=True)
        S = self.calc_S(H,P,R)
        MD2 = (z_hat-h).T@np.linalg.inv(S)@(z_hat-h)

        compat_indices = MD2 <= sp.stats.chi2.ppf(0.95, df=2)
        print(compat_indices) #should be 1d array of indices


        #replace non-indiv-compatible associations with NaN
        association[~compat_indices] = np.nan

        return association

    def calc_JNIS(self, association, scan_data, xs, track, boundary_points, P):
        #replace non-indiv-compatible associations with NaN
        #scan_data is N by 2 array of range, theta measurements
        z_hat = scan_data[np.arange(association.shape[0])]
        associated_points = boundary_points[association]
        h = self.calc_h(xs, track, associated_points)
        R = self.calc_R(associated_points)
        H = self.calc_Jacobian_H(xs, association)
        S = self.calc_S(H,P,R)
        JNIS = (z_hat-h).T@np.linalg.inv(S)@(z_hat-h)

        # compat_indices = MD2 <= sp.stats.chi2.ppf(0.95, df=2)

        #replace non-indiv-compatible associations with NaN
        # association[~compat_indices] = np.nan

        return JNIS, association

 
    def calc_R(self, associated_points):
        #https://dspace.mit.edu/handle/1721.1/32438#files-area
        R_indiv = np.array([[0.2, 0], [0,0.2]])
        R_matrices = tuple([R_indiv for i in range(len(associated_points))])
        Rs = block_diag(R_matrices)
        return Rs

    def calc_h(self, xs, track, associated_points):
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
        return h

    def calc_S(self, H, P, R):
        #check page 12 of Oxford paper
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



    def calc_Jacobian_H(self, xs, associated_points, indiv=False):
        g, G = self.calc_g_and_G(xs, associated_points)
        H = np.zeros((g.shape))
        U = np.zeros((G.shape[0], 2))
        U = self.calc_U(U, g)

        H = U.T @ G
        if indiv:
            return H.T
        else:
            return H.T[::2]



if __name__ == "__main__":
    jc = JCBB()
