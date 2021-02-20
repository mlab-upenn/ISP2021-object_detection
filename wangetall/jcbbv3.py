import numpy as np
import scipy as sp
from scipy.linalg import block_diag
from scipy import stats
from helper import Helper
import random
import sys
import time
import matplotlib.pyplot as plt

#https://github.com/Scarabrine/EECS568Project_Team2_iSAM/blob/master/JCBB_R.m

class JCBB:
    def __init__(self):
        self.alpha = 0.95
    
    def assign_values(self, xs, scan_data, track, P, static, psi):
        self.xs = xs
        self.scan_data = scan_data
        self.track = track
        self.P = P
        self.static = static
        self.psi = psi


    def run(self, cluster, initial_association, boundary_points):
        # track is vector [gamma_i, delta_i, phi_i, gamma_dot_i, delta_dot_i, phi_dot_i]
        #initial association as 1D vector. Indices of 
        # vector correspond to indices of lidar scan datapoints, 
        # and values in vector correspond to indices of track datapoints
        #unassociated datapoints are replaced with NaN maybe?

        # megacluster = self.combine_clusters(clusters) #
        individual_compatibilities = self.compute_compatibility(boundary_points)
        pruned_associations = self.prune_associations(initial_association, individual_compatibilities)
        # print("preWhile {}".format(pruned_associations[1]))
        JNIS = self.calc_JNIS(pruned_associations, boundary_points)
        # print("Original JNIS {}".format(JNIS))

        #JNIS seems really large. Why??
        # print(pruned_associations)
        # sys.exit(0)
        # print(pruned_associations)
        JNIS_delta = 0
        dof = np.count_nonzero(~np.isnan(pruned_associations[1]))*2
        chi2 = stats.chi2.ppf(self.alpha, df=dof)
        # print("Original chi2 {}".format(chi2))
        # sys.exit()

        while True:
            curr_association = np.copy(pruned_associations)
            # print("Pruned asso {}".format(pruned_associations))
            # print([~np.isnan(pruned_associations[1])])
            # print(np.arange(pruned_associations.shape[1])[~np.isnan(pruned_associations[1])])
            for index in np.arange(pruned_associations.shape[1])[~np.isnan(pruned_associations[1])]:
                # print(index)
                curr_pairing = pruned_associations[1, index]
                pruned_associations[1, index] = np.nan
                JNIS_new = self.calc_JNIS(pruned_associations, boundary_points)
                # print("JNIS New {}".format(JNIS_new))
                JNIS_new_delta = JNIS-JNIS_new
                # print("Old JNIS {}".format(JNIS))

                dof = np.count_nonzero(~np.isnan(pruned_associations[1]))*2
                chi2 = stats.chi2.ppf(self.alpha, df=dof)
                # print("new chi2 {}".format(chi2))
                # print("JNIS_new_delta {}".format(JNIS_new_delta))
                # print("JNIS_delta {}".format(JNIS_delta))

                if JNIS_new_delta > JNIS_delta:
                    # print("bigger!")
                    JNIS_delta = JNIS_new_delta
                    # print(JNIS_delta)
                    # print("Chi2 {}".format(chi2))

                    curr_association = np.copy(pruned_associations)
                    # print("JNIS proposed {}, Curr asso  {}".format(JNIS-JNIS_delta, curr_association[1]))
                pruned_associations[1, index] = curr_pairing
            # print("JNIS-JNIS_delta: {}".format(JNIS-JNIS_delta))
            # # print("JNIS {}".format(JNIS))
            # print("JNIS_delta {}".format(JNIS_delta))
            # print("Chi2 {}".format(chi2))
            # time.sleep(1)
            # print("curr association {}".format(curr_association[1]))
            if JNIS-JNIS_delta <= chi2 or dof ==0:
                # print("breaking while loop...")
                minimal_association = np.copy(curr_association)
                # print(minimal_association[1])
                # sys.exit()
                JNIS = JNIS-JNIS_delta
                break
            else:
                # print("nope...")
                pruned_associations = np.copy(curr_association)
        # print("minimal association {}".format(minimal_association))
        unassociated_measurements = minimal_association[0, np.isnan(minimal_association[1])]
        # unassociated_matrix = np.zeros((2, unassociated_measurements.shape[0]))
        # unassociated_matrix[0] = unassociated_measurements
        # unassociated_matrix[1] = np.nan
        compat_boundaries = {}
        for measurement in unassociated_measurements:
            # compat_boundaries[measurement] = np.append(individual_compatibilities[int(measurement),:],np.nan) #append null?
            boundary_idxs = np.where(individual_compatibilities[int(measurement),:] == 1)[0]
            selected_boundaries = set(boundary_idxs)
            selected_boundaries.add(np.nan)
            selected_boundaries = selected_boundaries-set(minimal_association[1])
            compat_boundaries[measurement] = list(selected_boundaries)
        assigned_associations = self.branch_and_bound(unassociated_measurements, minimal_association, compat_boundaries, boundary_points)
    
        return assigned_associations
    
    def branch_and_bound(self, unassociated_measurements, minimal_association, compat_boundaries, boundary_points):
        self.best_JNIS = np.inf
        self.best_num_associated = np.count_nonzero(~np.isnan(minimal_association[1]))
        self.best_association = np.copy(minimal_association)
        # print("min asso {}".format(self.best_association))
        boundaries_taken = set()
        self.explored = set() #consists of tuples of (level, boundary_point)
        # print("BEGIN DFS!")
        # print(minimal_association)
        self.unassociated_measurements = unassociated_measurements
        self.DFS(0, None, minimal_association, compat_boundaries, boundary_points, boundaries_taken)
        # print(assigned_association)
        print("best jnis {}, num assoc {}".format(self.best_JNIS, self.best_num_associated))
        # print("best asso")
        jnis = self.calc_JNIS(self.best_association, boundary_points)
        joint_compat = self.check_compat(jnis, DOF =np.count_nonzero(~np.isnan(self.best_association[1])*2))
        if joint_compat:
            # print(self.best_association)

            return self.best_association
        else:
            return np.zeros((self.best_association.shape))

    def DFS(self, level, boundary_point, association, compat_boundaries, boundary_points, boundaries_taken):
        # print("Boundaries taken {}".format(boundaries_taken))
        boundaries_taken = boundaries_taken.copy()
        self.explored.add((level, boundary_point))
        avail_boundaries = compat_boundaries[self.unassociated_measurements[level]]
        for next_boundary in avail_boundaries:
            if (level, next_boundary) not in self.explored and\
                 next_boundary not in boundaries_taken and level < self.unassociated_measurements.shape[0]:
                print("next boundary: {}, boundaries_taken: {}".format(next_boundary, boundaries_taken))
                test_association = association[:]
                test_association[1,int(self.unassociated_measurements[level])] = next_boundary

                #maybe for a more accurate JNIS calc, do I need to combine this association with the previous one?
                JNIS = self.calc_JNIS(test_association, boundary_points)
                # print("======")
                joint_compat = self.check_compat(JNIS, DOF =np.count_nonzero(~np.isnan(test_association[1])*2))
                # print("JNIS {}".format(JNIS))
                # print("Best JNIS {}".format(self.best_JNIS))
                # print("Joint Compat {}".format(joint_compat))
                num_associated = np.count_nonzero(~np.isnan(test_association[1]))
                # print("Num associations {}".format(num_associated))
                # print("======")
                update = False
                if joint_compat and num_associated >= self.best_num_associated:
                    if num_associated == self.best_num_associated:
                        if JNIS <= self.best_JNIS:
                            update = True
                            self.best_JNIS = JNIS
                    else:
                        update = True
                if update:
                    print("found a better one!")
                    # print(test_association)
                    self.best_num_associated = num_associated
                    self.best_association = np.copy(test_association)
                if level+1 <= self.unassociated_measurements.shape[0]:
                    print("goin' forward.")
                    boundaries_taken.add(next_boundary)
                    self.DFS(level+1, next_boundary, test_association, compat_boundaries, boundary_points, boundaries_taken)

    def check_compat(self, JNIS, DOF):
        chi2 = stats.chi2.ppf(self.alpha, df=DOF)
        # print("chi2: {}".format(chi2))
        return JNIS <= chi2
 
    def compute_compatibility(self, boundary_points):
        #returns MxN matrix of compatibility boolean
        individual_compatabilities = np.zeros((self.scan_data.shape[0], boundary_points.shape[0]))
        chi2_val = stats.chi2.ppf(self.alpha, df=2)
        for i in range(self.scan_data.shape[0]):
            association = np.zeros((2, boundary_points.shape[0]))
            association[0] = i
            association[1] = np.arange(boundary_points.shape[0])
            # print(scan_data.shape)
            JNIS = self.calc_JNIS(association, boundary_points, indiv=True)
            individual_compatabilities[i, np.where(JNIS<=chi2_val)] = 1
        return individual_compatabilities


    def prune_associations(self, associations, individual_compatibilies):
        #check if row in individual compatibilities contains any True's
        null_rows = np.where(~individual_compatibilies.any(axis=1))[0]
        associations[1, null_rows] = np.nan

        null_cols = np.where(~individual_compatibilies.any(axis=0))[0]
        for col in null_cols:
            associations[1][np.where(associations[1]==col)] = np.nan
        #find duplicates

        # count_horiz = np.count_nonzero(individual_compatibilies, axis = 1)
        # rm_horiz_idx = count_horiz > 1
        # associations[1, count_horiz > 1] = np.nan

        # # print("indiv compat {}".format(individual_compatibilies))
        # print(associations[1])
        u, c = np.unique(associations[1], return_counts=True)
        dup = u[c > 1]
        for item in dup:
            rm_idxs = np.where(associations[1] == item)[0]
            if len(rm_idxs) >= 2:
                rm_idxs = np.random.choice(rm_idxs, size = len(rm_idxs)-1, replace = False)
            associations[1][rm_idxs] = np.nan

        # count_vert = np.count_nonzero(individual_compatibilies, axis = 0)
        # rm_points = np.where(count_vert > 1)
        # for point in rm_points[0]:
        #     rm_idxs = np.where(associations[1] == point)[0]
        #     if len(rm_idxs) > 2:
        #         rm_idxs = np.random.choice(rm_idxs, size = len(rm_idxs)-1, replace = False)
        #     associations[1][rm_idxs] = np.nan


        return associations

        
    def calc_JNIS(self, association, boundary_points, indiv= False):
        #want JNIS to output vector of JNIS's if individual
        #want single JNIS if joint.
        bndry_points_idx = association[1][~np.isnan(association[1])].astype(int)
        z_hat_idx = association[0][~np.isnan(association[1])].astype(int)

        associated_points = boundary_points[bndry_points_idx]
        if len(associated_points) == 0:
            return np.inf

        g, G = self.calc_g_and_G(associated_points, indiv)
        h = self.calc_h(g)

        R = self.calc_R(associated_points, indiv)
        H = self.calc_Jacobian_H(g, G, associated_points, indiv)
        # print("association {}".format(association[1]))
        # print("Asso idx {}".format(association_idx))

        # print("Asso points {}".format(associated_points))
        S = self.calc_S(H,R, indiv)
        # z_hat_idx = association[0, scan_points_idx].astype(int)
        # print("Association {}".format(association))
        # print("z_hat_idx {}".format(z_hat_idx))
        if indiv:
            # print(~np.isnan(association[1]))
            # print(association[0])
            z_hat = self.scan_data[z_hat_idx]
            # print("zhat shape {}".format(z_hat.shape))
            # print("h shape {}".format(h.shape))

            a = (z_hat-h) #h is wrong shape!!
            b = np.linalg.inv(S)
            #b is supposed to be Nx2x2
            #a is supposed to be Nx2. It's not currently :/
            JNIS = np.einsum('ki,kij,kj->k', a, b, a)
        else:
            z_hat = self.scan_data[z_hat_idx].flatten()

            h = h.flatten()
            # sys.exit()
            # np.save("association_pre.npy", association)

            np.save("h.npy", h)
            np.save("z_hat.npy", z_hat)
            np.save("S.npy", S)

            # print("S {}".format(S))
            # print("Sub {}".format(np.sum(z_hat-h)))

            JNIS = (z_hat-h).T@np.linalg.inv(S)@(z_hat-h)

            # print("JNIS {}".format(JNIS))

        return JNIS

    def calc_R(self, associated_points, indiv):
        #https://dspace.mit.edu/handle/1721.1/32438#files-area
        R_indiv = np.array([[2, 0], [0,2]])
        if indiv:
            R_stacked = np.zeros((len(associated_points), 2,2))
            R_stacked[:] = R_indiv
            return R_stacked
        else:
            R_matrices = tuple([R_indiv for i in range(len(associated_points))])
            Rs = block_diag(*R_matrices)
            return Rs


    def calc_u(self, f):
        r = np.sqrt(f[:,0]**2+f[:,1]**2)
        theta = np.arctan2(f[:,1], f[:,0])
        return np.array([r,theta]).T

    def calc_h(self, g):
        h = self.calc_u(g)
        return h

    def calc_S(self, H, R, indiv):
        #check page 12 of Oxford paper
        if indiv:
            P_stacked = np.zeros((R.shape[0], 2, 2))
            P_stacked[:] = self.P
            temp = np.einsum('ijk,ikl->ijl', H, P_stacked)
            S = np.einsum('ijk,ilk->ijl', temp, H)+R
        else:
            # print("P.shape {}".format(P.shape))
            # print("H shape")
            # print(H.shape)

            # sys.exit()
            # P_matrices = tuple([self.P for i in range(H.shape[0]//2)])
            # P_block =  block_diag(*P_matrices)
            # print("H shape {}".format(H.shape))
            # print("R shape {}".format(R.shape))
            # print("P block shape {}".format(P_block.shape))
            # print("H {}".format(H))
            #ooh this may be wrong...
            S = H@P@H.T + R
            # else:
            #     S =0
        return S

    def calc_g_and_G(self, associated_points, indiv):
        """inputs: xs, measured laserpoint
        
        xs is dict of measurements with xs["alpha"] = const, xs["beta"] = const maybe?
        
        measured_laserpoint is 2d matrix with one col of angles, one col of x coords, one col of y coords 
        where psi is the current rotation angle
        """
        g = np.zeros((associated_points.shape[0], 2))
        if indiv:
            G  = np.zeros((associated_points.shape[0], 2, 2))
        else:
            G = np.zeros((associated_points.shape[0]*2, 2))
            # G_matrices = []

        alpha = self.xs["alpha"]
        beta = self.xs["beta"]
        alpha_beta_arr = np.array([alpha, beta])
        phi = self.track[2]
        gamma = self.track[0]
        delta = self.track[1]

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
            
            if indiv:
                G[index] = -R_psi.T-R_pi_by_2@g[index]
            else:
                G[index*2:index*2+2] = -R_psi.T-R_pi_by_2@g[index]
                # G_matrices.append(-R_psi.T-R_pi_by_2@g[index])
        
        # if not indiv:
        #     G = block_diag(*tuple(G_matrices))
        return g, G


    def calc_Jacobian_H(self, g, G, associated_points, indiv):
        U = self.calc_U(g, len(associated_points), indiv)
        H = U.T @ G
        return H

    def calc_U(self, g, num_tiles, indiv):
        r = np.sqrt(g[:,0]**2+g[:,1]**2)
        U = (np.array([[r*g[:,0], r*g[:,1]],[-g[:,1], g[:,0]]]))/r**2
            
        if not indiv:
            U_matrices = tuple([U[:,:,i] for i in range(U.shape[2])])
            U =  block_diag(*U_matrices)

        # return np.tile(U_indiv, (num_tiles,1))
        return U


def convert_scan_polar_cartesian(scan):
    return np.cos(scan[:,1])*scan[:,0], np.sin(scan[:,1])*scan[:,0]


def plot_association(asso):
    pairings = asso[:,~np.isnan(asso[1])]
    selected_bndr_pts = boundary_points[pairings[1].astype(int)]
    selected_scan_pts = scan_data[pairings[0].astype(int)]

    selected_scan_x, selected_scan_y = convert_scan_polar_cartesian(selected_scan_pts)
    scan_x, scan_y = convert_scan_polar_cartesian(scan_data)
    #plot!!


    #scan data points plot
    plt.scatter(scan_x, scan_y, c="b", marker="o", alpha = 0.5, label="Scan Data")
    #boundary points plot
    plt.scatter(boundary_points[:,0]+track[0], boundary_points[:,1]+track[1], c="orange", marker="o", alpha = 0.5, label="Boundary Points")


    #SELECTED/PAIRED POINTS
    plt.scatter(selected_scan_x, selected_scan_y, c="red", marker="v", label="Paired Scan Points")
    for i in range(selected_scan_x.shape[0]):
        plt.text(selected_scan_x[i], selected_scan_y[i], str(i))

    plt.scatter(selected_bndr_pts[:,0]+track[0], selected_bndr_pts[:,1]+track[1], c="black", marker="v", label="Paired Boundary Points")
    for i in range(selected_bndr_pts.shape[0]):
        plt.text(selected_bndr_pts[i,0]+track[0], selected_bndr_pts[i,1]+track[1], str(i))

    plt.legend()
    plt.title("Runtime: {}".format(runtime))
    plt.show()



if __name__ == "__main__":
    jc = JCBB()
    cluster = None
    # for i in range(100):

    xs = {"alpha":0, "beta":0}
    n = 100
    initial_association = np.zeros((2, n))
    initial_association[0] = np.arange(n)
    initial_association[1] = np.random.randint(0, 10, n)
    scan_data = np.zeros((n,2))
    sigma = 0.1
    scan_data[:,0] = np.random.uniform(28, 38, n) #1st row is ranges
    scan_data[:,1] = np.random.uniform(0.6, 1.1, n) #2nd row is angles (radians)

    # scan_data[:,0] = np.random.normal(33,sigma, n) #1st row is ranges
    # scan_data[:,1] =  np.random.normal(0.85,sigma, n) #1st row is ranges

    n_boundary = 10
    boundary_points= np.zeros((n_boundary,2))
    # boundary_points[:,0] = np.random.uniform(-3, 3, n_boundary) #x coord, relative to track coordinate
    # boundary_points[:,1] = np.random.uniform(-3, 3, n_boundary) #y coord, relative to track coordinate
    boundary_points[:,0] = np.array([0,0,0,0,0,1,2,3,4,5]) #x coord, relative to track coordinate
    boundary_points[:,1] = np.array([0,1,2,3,4,0,0,0,0,0]) #y coord, relative to track coordinate

    # boundary_points[:,2] = np.random.uniform(-1, 1, 10)


    track = [20, 25, 0]
    P = np.eye(2)*0
    static = False
    psi = 0.0
    
    jc.assign_values(xs, scan_data, track, P, static, psi)


    starttime = time.time()
    asso = jc.run(cluster, initial_association, boundary_points)
    endtime = time.time()
    runtime = endtime-starttime

    if np.any(asso):
        plot_association(asso)
    else:
        print("No associations found.")
        # plot_association(asso)

