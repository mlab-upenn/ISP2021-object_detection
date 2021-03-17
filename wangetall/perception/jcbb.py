import numpy as np
import scipy as sp
from scipy.linalg import block_diag
from scipy import stats
from perception.helper import Helper
import random
import sys
import time
import matplotlib.pyplot as plt


#Sample code:
#https://github.com/Scarabrine/EECS568Project_Team2_iSAM/blob/master/JCBB_R.m
class RecursionStop(Exception):
    pass

class JCBB:
    def __init__(self):
        self.alpha = 1-0.95

    def assign_values(self, xs, scan_data, track, P, static, psi):
        self.xs = xs
        self.scan_data = scan_data
        self.track = track
        self.P = P
        # assert self.P[0,0] < 100
        self.static = static
        # self.psi = psi
        self.psi = 0
        self.recursion = 0


    def run(self, initial_association, boundary_points):
        # track is vector [gamma_i, delta_i, phi_i, gamma_dot_i, delta_dot_i, phi_dot_i]
        #initial association as 1D vector. Indices of 
        # vector correspond to indices of lidar scan datapoints, 
        # and values in vector correspond to indices of track datapoints
        #unassociated datapoints are replaced with NaN maybe?

        # megacluster = self.combine_clusters(clusters) #
        assert initial_association.shape[0] == 2
        assert boundary_points.shape[1] == 2
        assert self.scan_data.shape[1] == 2
        # print("Boundary points shape {}".format(boundary_points.shape))
        individual_compatibilities = self.compute_compatibility(boundary_points)
        # np.save("P.npy", self.P)
        # np.save("xs.npy", self.xs)
        # np.save("scan_data.npy", self.scan_data)
        # np.save("boundary_points.npy", boundary_points)
        # np.save("psi.npy", self.psi)
        # np.save("initial_association.npy", initial_association)
        # np.save("track.npy", self.track)

        # print("Percent indiv compat: {}".format(np.count_nonzero(individual_compatibilities)/np.multiply(*individual_compatibilities.shape)))
        # if np.count_nonzero(individual_compatibilities)/np.multiply(*individual_compatibilities.shape) > 0.8:
        #     print("COMPAT ALERT!")
        #     plt.figure()
        #     # np.save("P.npy", self.P)
        #     # np.save("xs.npy", self.xs)
        #     # np.save("scan_data.npy", self.scan_data)
        #     # np.save("boundary_points.npy", boundary_points)
        #     # np.save("psi.npy", self.psi)
        #     # np.save("initial_association.npy", initial_association)
        #     # np.save("track.npy", self.track)

        #     scan_x, scan_y = convert_scan_polar_cartesian(self.scan_data)
        #     plt.scatter(scan_x, scan_y, c="b", marker="o", alpha = 0.5, label="Scan Data")
        #     plt.scatter(boundary_points[:,0]+self.track[0], boundary_points[:,1]+self.track[1], c="orange", marker="o", alpha = 0.5, label="Boundary Points")
        #     plt.legend()
        #     plt.show()


        pruned_associations = self.prune_associations(initial_association, individual_compatibilities)
        JNIS = self.calc_JNIS(pruned_associations, boundary_points)

        JNIS_delta = 0
        dof = np.count_nonzero(~np.isnan(pruned_associations[1]))*2
        chi2 = stats.chi2.ppf(self.alpha, df=dof)
        max_iter = 10
        i = 0

        while i < max_iter:
            curr_association = np.copy(pruned_associations)
            for index in np.arange(pruned_associations.shape[1])[~np.isnan(pruned_associations[1])]:
                curr_pairing = pruned_associations[1, index]
                pruned_associations[1, index] = np.nan
                JNIS_new = self.calc_JNIS(pruned_associations, boundary_points)
                JNIS_new_delta = JNIS-JNIS_new

                if JNIS_new_delta > JNIS_delta:
                    JNIS_delta = JNIS_new_delta

                    curr_association = np.copy(pruned_associations)
                pruned_associations[1, index] = curr_pairing

            dof = np.count_nonzero(~np.isnan(curr_association[1]))*2
            chi2 = stats.chi2.ppf(self.alpha, df=dof)

            if JNIS-JNIS_delta <= chi2 or dof ==0:
                minimal_association = np.copy(curr_association)
                JNIS = JNIS-JNIS_delta
                # print("MIN ASSO {}".format(minimal_association))
                break
            else:
                pruned_associations = np.copy(curr_association)
                i+=1
        unassociated_measurements = minimal_association[0, np.isnan(minimal_association[1])]
        compat_boundaries = {}
        for measurement in unassociated_measurements:
            boundary_idxs = np.where(individual_compatibilities[int(measurement),:] == 1)[0]

            selected_boundaries = set(boundary_idxs)
            min_asso_vals = np.unique(minimal_association[1])
            min_asso_vals = min_asso_vals[~np.isnan(min_asso_vals)]

            selected_boundaries.add(np.nan)
            # print("Len selected bound {}".format(len(list(selected_boundaries))))

            selected_boundaries = np.setdiff1d(np.array(list(selected_boundaries)), min_asso_vals)

            compat_boundaries[measurement] = list(selected_boundaries)
        assigned_associations = self.branch_and_bound(unassociated_measurements, minimal_association, compat_boundaries, boundary_points)

        return assigned_associations
    
    def branch_and_bound(self, unassociated_measurements, minimal_association, compat_boundaries, boundary_points):
        self.best_JNIS = np.inf
        self.best_num_associated = np.count_nonzero(~np.isnan(minimal_association[1]))
        self.best_association = np.copy(minimal_association)
        boundaries_taken = set()
        self.unassociated_measurements = unassociated_measurements
        try:
            self.DFS(0, minimal_association, compat_boundaries, boundary_points, boundaries_taken)
        except RecursionStop:
            pass

        jnis = self.calc_JNIS(self.best_association, boundary_points)
        joint_compat = self.check_compat(jnis, DOF =np.count_nonzero(~np.isnan(self.best_association[1]))*2)
        if joint_compat:
            return self.best_association
        else:
            return np.zeros((self.best_association.shape))

    def DFS(self, level, association, compat_boundaries, boundary_points, boundaries_taken):
        self.recursion += 1
        if self.recursion >= 200:
            raise RecursionStop

        boundaries_taken = boundaries_taken.copy()
        avail_boundaries = compat_boundaries[self.unassociated_measurements[level]]
        # print(avail_boundaries)
        for next_boundary in avail_boundaries:
            # print("Next boundary {}".format(next_boundary))
            isValidBoundary = next_boundary not in boundaries_taken or np.isnan(next_boundary)
            if isValidBoundary and level < len(self.unassociated_measurements):
                test_association = np.copy(association)
                test_association[1,int(self.unassociated_measurements[level])] = next_boundary
                JNIS = self.calc_JNIS(test_association, boundary_points)
                joint_compat = self.check_compat(JNIS, DOF =np.count_nonzero(~np.isnan(test_association[1]))*2)
                num_associated = np.count_nonzero(~np.isnan(test_association[1]))

                update = False
                if joint_compat and num_associated >= self.best_num_associated:
                    if num_associated == self.best_num_associated:
                        if JNIS <= self.best_JNIS:
                            update = True
                    else:
                        update = True
                if update:
                    self.best_JNIS = JNIS
                    self.best_num_associated = num_associated
                    self.best_association = np.copy(test_association)
                if joint_compat and level+1 < len(self.unassociated_measurements):
                    boundaries_taken.add(next_boundary)
                    try:
                        self.DFS(level+1, np.copy(test_association), compat_boundaries, boundary_points, boundaries_taken)
                    except RecursionStop:
                        raise RecursionStop
    def check_compat(self, JNIS, DOF):
        if DOF == 0:
            return True
        else:
            chi2 = stats.chi2.ppf(self.alpha, df=DOF)
            return JNIS <= chi2
 
    def compute_compatibility(self, boundary_points):
        #returns MxN matrix of compatibility boolean

        individual_compatabilities = np.zeros((self.scan_data.shape[0], boundary_points.shape[0]))
        chi2_val = stats.chi2.ppf(self.alpha, df=2)
        for i in range(self.scan_data.shape[0]):
            #Code optimization: can I do all this in one swoop with a 4d matrix??
            association = np.zeros((2, boundary_points.shape[0]))
            association[0] = i
            association[1] = np.arange(boundary_points.shape[0])
            JNIS = self.calc_JNIS(association, boundary_points, indiv=True, i = i)
            individual_compatabilities[i, np.where(JNIS<=chi2_val)] = 1

        return individual_compatabilities


    def prune_associations(self, associations, individual_compatibilies):
        #check if row in individual compatibilities contains any True's
        null_rows = np.where(~individual_compatibilies.any(axis=1))[0]
        associations[1, null_rows] = np.nan

        null_cols = np.where(~individual_compatibilies.any(axis=0))[0]
        for col in null_cols:
            associations[1][np.where(associations[1]==col)] = np.nan
        u, c = np.unique(associations[1], return_counts=True)
        dup = u[c > 1]
        for item in dup:
            rm_idxs = np.where(associations[1] == item)[0]
            if len(rm_idxs) >= 2:
                rm_idxs = np.random.choice(rm_idxs, size = len(rm_idxs)-1, replace = False)
            associations[1][rm_idxs] = np.nan
        return associations

        
    def calc_JNIS(self, association, boundary_points, indiv= False, i = 0):
        #want JNIS to output vector of JNIS's if individual
        #want single JNIS if joint.
        bndry_points_idx = association[1][~np.isnan(association[1])].astype(int)
        z_hat_idx = association[0][~np.isnan(association[1])].astype(int)

        associated_points = boundary_points[bndry_points_idx]

        if len(associated_points) == 0:
            return 0

        g, G = self.calc_g_and_G(associated_points, indiv)
        h = self.calc_h(g)

        R = self.calc_R(associated_points, indiv)
        H = self.calc_Jacobian_H(g, G, associated_points, indiv)
        S = self.calc_S(H,R, indiv)
        if indiv:
            z_hat = self.scan_data[z_hat_idx]

            a = (z_hat-h)
            # print("z_hat {},h {}".format(z_hat, h))
            b = np.linalg.inv(S)
            JNIS = np.einsum('ki,kij,kj->k', a, b, a)
        else:
            z_hat = self.scan_data[z_hat_idx].flatten()
            h = h.flatten()
            JNIS = (z_hat-h).T@np.linalg.inv(S)@(z_hat-h)
        return JNIS

    def calc_R(self, associated_points, indiv):
        #https://dspace.mit.edu/handle/1721.1/32438#files-area
        R_indiv = np.array([[0.1, 0], [0,0.1]])
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
            S = H@self.P@H.T + R
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

        alpha = self.xs[0]
        beta = self.xs[1]
        alpha_beta_arr = np.array([alpha, beta])
        if not self.static:
            phi = self.track[2]
            gamma = self.track[0]
            delta = self.track[1]
            R_phi = Helper.compute_rot_matrix(phi)


        # R_pi_by_2 = Helper.compute_rot_matrix(np.pi/2)
        R_psi = Helper.compute_rot_matrix(self.psi)
        #naive way-- with for loops. need to think how to get rid of.
        for index, point in enumerate(associated_points):
            x = point[0]
            y = point[1]

            if self.static:
                g[index] = R_psi.T @ np.array(np.array([x, y])- alpha_beta_arr).T
            else:
                g[index] = R_psi.T@(R_phi@np.array([x, y])+np.array([gamma, delta])-alpha_beta_arr)
            if indiv:
                # G[index] = -R_psi.T-R_pi_by_2@g[index]
                if self.static:
                    G[index] = R_psi.T
                else:
                    G[index] = R_psi.T@R_phi
            else:
                # G[index*2:index*2+2] = -R_psi.T-R_pi_by_2@g[index]
                if self.static:
                     G[index*2:index*2+2] = R_psi.T
                else:
                    G[index*2:index*2+2] = R_psi.T@R_phi
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
        return U


def convert_scan_polar_cartesian(scan):
    return np.cos(scan[:,1])*scan[:,0], np.sin(scan[:,1])*scan[:,0]

def convert_cartesian_to_polar(data):
    r = np.sqrt(data[:,0]**2+data[:,1]**2)
    phi = np.arctan2(data[:,1], data[:,0])
    return r, phi


def plot_association(asso, polar):
    pairings = asso[:,~np.isnan(asso[1])]
    selected_bndr_pts = boundary_points[pairings[1].astype(int)]
    selected_scan_pts = scan_data[pairings[0].astype(int)]

    if not polar:
        selected_scan_x, selected_scan_y = convert_scan_polar_cartesian(selected_scan_pts)
        scan_x, scan_y = convert_scan_polar_cartesian(scan_data)

        #scan data points plot
        plt.scatter(scan_x, scan_y, c="b", marker="o", alpha = 0.5, label="Scan Data")
        # for i in range(scan_x.shape[0]):
        #     plt.text(scan_x[i], scan_y[i], str(i), size = "xx-small")

        #boundary points plot
        plt.scatter(boundary_points[:,0]+track[0], boundary_points[:,1]+track[1], c="orange", marker="o", alpha = 0.5, label="Boundary Points")
        # for i in range(boundary_points.shape[0]):
        #     plt.text(boundary_points[i,0]+track[0], boundary_points[i,1]+track[1], str(i), size = "xx-small")


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
    else:
        selected_bndr_pts[:,0]+= track[0]
        selected_bndr_pts[:,1]+= track[1]
        boundary_points[:,0] += track[0]
        boundary_points[:,1] += track[1]

        selected_boundary_r, selected_boundary_phi = convert_cartesian_to_polar(selected_bndr_pts)
        boundary_r, boundary_phi = convert_cartesian_to_polar(boundary_points)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        c = ax.scatter(boundary_phi, boundary_r, c="orange",marker = "o", alpha=0.5, label="Boundary Points")
        c = ax.scatter(scan_data[:,1], scan_data[:,0], c="b", marker="o", alpha=0.5,label="Scan Data")

        c = ax.scatter(selected_boundary_phi, selected_boundary_r, c="black",marker = "v", alpha=0.5)
        for i in range(selected_boundary_phi.shape[0]):
            plt.text(selected_boundary_phi[i], selected_boundary_r[i], str(i))

        c = ax.scatter(selected_scan_pts[:,1], selected_scan_pts[:,0], c="red", marker = "v",label="Paired Scan Points")
        for i in range(selected_scan_pts[:,1].shape[0]):
            plt.text(selected_scan_pts[:,1][i], selected_scan_pts[:,0][i], str(i))

        ax.set_xlim(0.18*np.pi, 0.35*np.pi)
        ax.set_ylim(27, 38)

        plt.title("Runtime: {}".format(runtime))
        plt.show()




if __name__ == "__main__":
    import numpy as np
    import scipy as sp
    from scipy.linalg import block_diag
    from scipy import stats
    from helper import Helper
    import random
    import sys
    import time
    import matplotlib.pyplot as plt

    jc = JCBB()
    # for i in range(100):
    # np.random.seed(2003)
    # xs = [0,0]
    # n = 500
    # initial_association = np.zeros((2, n))
    # initial_association[0] = np.arange(n)
    # initial_association[1] = np.random.randint(0, 10, n)
    # scan_data = np.zeros((n,2))
    # scan_data[:,0] = np.random.uniform(28, 38, n) #1st row is ranges
    # scan_data[:,1] = np.random.uniform(0.6, 1.1, n) #2nd row is angles (radians)


    # # scan_data[:,0], scan_data[:,1] = convert_scan_polar_cartesian(scan_data)
    # # scan_data[:,0] = np.random.normal(33,sigma, n) #1st row is ranges
    # # scan_data[:,1] =  np.random.normal(0.85,sigma, n) #1st row is ranges

    # n_boundary = 10
    # boundary_points= np.zeros((n_boundary,2))
    # # boundary_points[:,0] = np.random.uniform(-3, 3, n_boundary) #x coord, relative to track coordinate
    # # boundary_points[:,1] = np.random.uniform(-3, 3, n_boundary) #y coord, relative to track coordinate
    # boundary_points[:,0] = np.array([0,0,0,0,0,1,2,3,4,5]) #x coord, relative to track coordinate
    # boundary_points[:,1] = np.array([0,1,2,3,4,0,0,0,0,0]) #y coord, relative to track coordinate


    # track = [20, 25, 0]
    # P = np.eye(2)*0
    # static = False
    # psi = 0 #sensor angle. Will work if adjusted for JCBB running purposes, but 
    #         #don't change-- need to refactor a bit to make the plot look nice too.
    xs = np.load("xs.npy")
    scan_data = np.load("scan_data.npy")
    # scan_x, scan_y = Helper.convert_scan_polar_cartesian_joint(scan_data)
    # plt.scatter(scan_x, scan_y, c="b", marker="o", alpha = 0.5, label="Scan Data")
    # plt.xlim(-10,3)
    # plt.ylim(-1, 9)

    P = np.load("P.npy")
    track = np.load("track.npy")
    static = False
    psi = np.load("psi.npy")
    initial_association = np.load("initial_association.npy")
    boundary_points = np.load("boundary_points.npy")
    # plt.scatter(boundary_points[:,0]+track[0], boundary_points[:,1]+track[1], c="orange", marker="o", alpha = 0.5, label="Scan Data")
    # plt.xlim(-10,3)
    # plt.ylim(-1, 9)
    # plt.show()

    jc.assign_values(xs, scan_data, track, P, static, psi)


    starttime = time.time()
    asso = jc.run(initial_association, boundary_points)
    endtime = time.time()
    runtime = endtime-starttime

    if np.any(asso):
        plot_association(asso, polar=False)
    else:
        print("No associations found.")
        # plot_association(asso)