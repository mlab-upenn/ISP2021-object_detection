import numpy as np
import scipy as sp
from scipy.linalg import block_diag
from scipy import stats
from helper import Helper
import random
import sys
import time


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
        # print("JNIS {}".format(JNIS))
        #JNIS seems really large. Why??
        # print(pruned_associations)
        # sys.exit(0)
        # print(pruned_associations)
        while True:
            chi2 = stats.chi2.ppf(0.95, df=2)
            JNIS_delta = 0
            curr_association = pruned_associations
            # print("Pruned asso {}".format(pruned_associations))
            # print([~np.isnan(pruned_associations[1])])
            # print(np.arange(pruned_associations.shape[1])[~np.isnan(pruned_associations[1])])
            for index in np.arange(pruned_associations.shape[1])[~np.isnan(pruned_associations[1])]:
                # print(index)
                curr_pairing = pruned_associations[1, index]
                pruned_associations[1, index] = np.nan
                JNIS_new = self.calc_JNIS(curr_association, scan_data, xs, track, boundary_points, P)
                JNIS_new_delta = JNIS-JNIS_new
                dof = np.count_nonzero(~np.isnan(pruned_associations[1]))*2
                # print("dof {}".format(dof))
                chi2 = stats.chi2.ppf(0.95, df=dof)
                # print("JNIS_new_delta {}".format(JNIS_new_delta))
                # print("JNIS_delta {}".format(JNIS_delta))

                if JNIS_new_delta > JNIS_delta:
                    JNIS_delta = JNIS_new_delta
                    curr_association = np.copy(pruned_associations)
                    pruned_associations[1, index] = curr_pairing
            # print("JNIS-JNIS_delta: {}".format(JNIS-JNIS_delta))
            # print("JNIS {}".format(JNIS))
            # print("JNIS_delta {}".format(JNIS_delta))
            if JNIS-JNIS_delta <= chi2:
                # print("breaking while loop...")
                minimal_association = curr_association
                JNIS = JNIS-JNIS_delta
                break
            else:
                pruned_associations = curr_association
        # print("minimal association {}".format(minimal_association))
        unassociated_measurements = minimal_association[0, np.isnan(minimal_association[1])]
        unassociated_matrix = np.zeros((2, unassociated_measurements.shape[0]))
        unassociated_matrix[0] = unassociated_measurements
        unassociated_matrix[1] = np.nan
        compat_boundaries = {}
        
        for measurement in unassociated_measurements:
            # compat_boundaries[measurement] = np.append(individual_compatibilities[int(measurement),:],np.nan) #append null?
            boundary_idxs = np.where(individual_compatibilities[int(measurement),:] == 1)[0]
            compat_boundaries[measurement] = np.append(boundary_idxs,np.nan) #append null?

        
        assigned_associations = self.branch_and_bound(unassociated_matrix, compat_boundaries, scan_data, xs, track, boundary_points, P)
    
        return assigned_associations
    
    def branch_and_bound(self, unassociated_matrix, compat_boundaries, scan_data, xs, track, boundary_points, P):
        self.best_JNIS = np.inf
        self.best_num_associated = 0
        self.best_association = np.copy(unassociated_matrix)
        self.boundaries_taken = set()
        self.explored = set() #consists of tuples of (level, boundary_point)
        
        self.DFS(1, None, unassociated_matrix, compat_boundaries, scan_data, xs, track, boundary_points, P)
        # print(assigned_association)
        print("best jnis {}, num assoc {}".format(self.best_JNIS, self.best_num_associated))

        return self.best_association

    def DFS(self, level, boundary_point, association, compat_boundaries, scan_data, xs, track, boundary_points, P):
        self.explored.add((level, boundary_point))
        avail_boundaries = compat_boundaries[level]
        for next_boundary in avail_boundaries:
            if (level+1, next_boundary) not in self.explored and\
                 next_boundary not in self.boundaries_taken and level+1 < association.shape[1]:
                test_association = association[:]
                test_association[1,level+1] = next_boundary

                #maybe for a more accurate JNIS calc, do I need to combine this association with the previous one?
                JNIS = self.calc_JNIS(test_association, scan_data, xs, track, boundary_points, P)
                joint_compat = self.check_compat(JNIS, DOF =np.count_nonzero(~np.isnan(test_association)*2))
                print("======")
                print("JNIS {}".format(JNIS))
                print("Best JNIS {}".format(self.best_JNIS))
                print("Joint Compat {}".format(joint_compat))
                num_associated = np.count_nonzero(~np.isnan(test_association[1]))
                print("Num associations {}".format(num_associated))
                print("======")
                update = False
                if joint_compat and num_associated >= self.best_num_associated:
                    if num_associated == self.best_num_associated:
                        if JNIS <= self.best_JNIS:
                            update = True
                            self.best_JNIS = JNIS
                    else:
                        update = True
                if update:
                        # want to prioritize getting more num_associated over 
                    print("found a better one!")
                    # print(test_association)
                    self.boundaries_taken.add(next_boundary)
                    self.best_num_associated = num_associated
                    self.best_association = np.copy(test_association)
                    self.DFS(level+1, next_boundary, test_association, compat_boundaries, scan_data, xs, track, boundary_points, P)

    def check_compat(self, JNIS, DOF):
        chi2 = stats.chi2.ppf(0.95, df=DOF)
        return JNIS <= chi2
 
    def compute_compatibility(self, scan_data, boundary_points, xs, track, P):
        #returns MxN matrix of compatibility boolean
        individual_compatabilities = np.zeros((scan_data.shape[0], boundary_points.shape[0]))
        chi2_val = stats.chi2.ppf(0.95, df=2)
        for i in range(scan_data.shape[0]):
            association = np.zeros((2, boundary_points.shape[0]))
            association[0] = i
            association[1] = np.arange(boundary_points.shape[0])
            # print(scan_data.shape)
            JNIS = self.calc_JNIS(association, scan_data, xs, track, boundary_points, P, indiv=True)
            individual_compatabilities[i, np.where(JNIS<chi2_val)] = 1
        return individual_compatabilities


    def prune_associations(self, associations, individual_compatibilies):
        #check if row in individual compatibilities contains any True's
        null_rows = np.where(~individual_compatibilies.any(axis=1))[0]

        associations[1, null_rows] = np.nan
        #find duplicates

        count_horiz = np.count_nonzero(individual_compatibilies, axis = 1)
        associations[1, count_horiz > 1] = np.nan

        # print("indiv compat {}".format(individual_compatibilies))
        count_vert = np.count_nonzero(individual_compatibilies, axis = 0)
        rm_points = np.where(count_vert > 1)
        for point in rm_points[0]:
            associations[1][np.where(associations[1] == point)] = np.nan
        # print(associations)
        # sys.exit(0)

        return associations

        
    def calc_JNIS(self, association, scan_data, xs, track, boundary_points, P, indiv= False):
        #want JNIS to output vector of JNIS's if individual
        #want single JNIS if joint.
        association_idx = association[1][~np.isnan(association[1])].astype(int)
        associated_points = boundary_points[association_idx]
        # print("Asso points: {}".format(associated_points))
        h = self.calc_h(xs, track, associated_points, indiv)
        R = self.calc_R(associated_points, indiv)
        H = self.calc_Jacobian_H(xs, associated_points, indiv)
        S = self.calc_S(H,P,R, indiv)
        z_hat_idx = association[0, association_idx].astype(int)
        # print("Association {}".format(association))
        # print("z_hat_idx {}".format(z_hat_idx))
        if indiv:
            # print(~np.isnan(association[1]))
            # print(association[0])
            z_hat = scan_data[z_hat_idx]
            # print("zhat shape {}".format(z_hat.shape))
            # print("h shape {}".format(h.shape))

            a = (z_hat-h) #h is wrong shape!!
            b = np.linalg.inv(S)
            #b is supposed to be Nx2x2
            #a is supposed to be Nx2. It's not currently :/
            JNIS = np.einsum('ki,kij,kj->k', a, b, a)
        else:
            z_hat = scan_data[z_hat_idx].flatten()
            # print("S {}".format(S))
            # print("zhat {}".format(z_hat))
            # print("h {}".format(h))
            # print(z_hat.shape)
            # print(h.shape)
            # print(S.shape)
            h = h.flatten()
            JNIS = (z_hat-h).T@np.linalg.inv(S)@(z_hat-h)
        return JNIS

    def calc_R(self, associated_points, indiv):
        #https://dspace.mit.edu/handle/1721.1/32438#files-area
        R_indiv = np.array([[200, 0], [0,20]])
        if indiv:
            R_stacked = np.zeros((len(associated_points), 2,2))
            R_stacked[:] = R_indiv
            return R_stacked
        else:
            R_matrices = tuple([R_indiv for i in range(len(associated_points))])
            Rs = block_diag(*R_matrices)
            return Rs

    def calc_h(self, xs, track, associated_points, indiv):
        h = np.zeros((associated_points.shape[0], 2))
        # print("assoc points shape {}".format(associated_points.shape))
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

    def calc_S(self, H, P, R, indiv):
        #check page 12 of Oxford paper
        if indiv:
            P_stacked = np.zeros((R.shape[0], 2, 2))
            P_stacked[:] = P
            temp = np.einsum('ijk,ikl->ijl', H, P_stacked)
            S = np.einsum('ijk,ilk->ijl', temp, H)+R
        else:
            # print("H shape")
            # print(H.shape)
            P_tile = np.tile(P, (int(H.shape[0]/2), int(H.shape[1]/2)))
            # print("H shape {}".format(H.shape))
            # print("P shape {}".format(P_tile.shape))
            # print("R shape {}".format(R.shape))
            #Is H correct?
            S = H@P_tile@H.T + R
        return S

    def calc_g_and_G(self, xs, associated_points, indiv):
        """inputs: xs, measured laserpoint
        
        xs is dict of measurements with xs["alpha"] = const, xs["beta"] = const maybe?
        
        measured_laserpoint is 2d matrix with one col of angles, one col of x coords, one col of y coords 
        where psi is the current rotation angle

        """

        g = np.zeros((associated_points.shape[0], 2))
        if indiv:
            G  = np.zeros((associated_points.shape[0], 2, 2))
        else:
            G  = np.zeros((associated_points.shape[0]*2, 2))
            G_matrices = []


        alpha = xs["alpha"]
        beta = xs["beta"]
        R_pi_by_2 = Helper.compute_rot_matrix(np.pi/2)
        phi = 15 ##random. fix!
        R_phi = Helper.compute_rot_matrix(phi)
        #naive way-- with for loops. need to think how to get rid of.
        for index, point in enumerate(associated_points):
            g[index] = R_phi.T @ np.array(np.array([point[1], point[2]])- np.array([alpha, beta])).T
            if indiv:
                G[index] = -R_phi.T-R_pi_by_2@g[index]
            else:
                G_matrices.append(-R_phi.T-R_pi_by_2@g[index])
        
        if not indiv:
            G = block_diag(*tuple(G_matrices))

        return g, G



    def calc_Jacobian_H(self, xs, associated_points, indiv):
        g, G = self.calc_g_and_G(xs, associated_points, indiv)
        H = np.zeros((g.shape))
        U = self.calc_U(g, len(associated_points), indiv)
        H = U.T @ G
        if indiv:
            return H.T.reshape(len(associated_points), 2, 2)
        else:
            return H.T

    def calc_U(self, g, num_tiles, indiv):
        """Shapes are off..."""
        r = np.sqrt(g[:,0]**2+g[:,1]**2)
        U = (np.array([[r*g[:,0], r*g[:,1]],[-g[:,1], g[:,0]]]))/r**2
            
        if not indiv:
            U_matrices = tuple([U[:,:,i] for i in range(U.shape[2])])
            U =  block_diag(*U_matrices)

        # return np.tile(U_indiv, (num_tiles,1))
        return U

if __name__ == "__main__":
    jc = JCBB()
    cluster = None
    xs = {"alpha":0, "beta":0}
    n = 100
    initial_association = np.zeros((2, n))
    initial_association[0] = np.arange(n)
    initial_association[1] = np.random.randint(0, 10, n)
    scan_data = np.zeros((n,2))
    scan_data[:,0] = np.linspace(0.7, 1, n)
    scan_data[:,1] = np.random.randint(26, 34, n)
    boundary_points= np.zeros((10,3))
    boundary_points[:,0] = np.random.randint(20, 25, 10)
    boundary_points[:,1] = np.random.randint(20, 25, 10)
    boundary_points[:,2] = np.random.randint(15, 16, 10)

    track = [0, 22, 22]
    P = np.ones((2,2))
    starttime = time.time()
    asso = jc.run(cluster, initial_association, scan_data, xs, boundary_points, track, P)
    endtime = time.time()
    print(asso)
    print(endtime-starttime)
