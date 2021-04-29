import numpy as np
import scipy as sp
from scipy.linalg import block_diag
from scipy.linalg import solve_triangular
from scipy import stats
# from perception.helper import Helper
import random
import sys
import time
import matplotlib.pyplot as plt
import logging
import datetime as dt
import os
import logging
import torch






#Sample code:
#https://github.com/Scarabrine/EECS568Project_Team2_iSAM/blob/master/JCBB_R.m
class RecursionStop(Exception):
    pass

class JCBB:
    def __init__(self):
        self.alpha = 1-0.95
        #Prefire DFS

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

        self.firstrun = True
        logging.info("Boundary points shape {}".format(boundary_points.shape))
        logging.info("Scan data shape {}".format(self.scan_data.shape))
        logging.info("Calculating indiv compat...")
        start = time.time()
        individual_compatibilities = self.compute_compatibility(boundary_points)
        end = time.time()
        if end-start < 1:
            logging.info("Indiv compat time taken {}".format(end-start))
        else:
            logging.warn("Indiv compat excessive time. Time: {}".format(end-start))

        pruned_associations = self.prune_associations(initial_association, individual_compatibilities)
        JNIS = self.calc_JNIS(pruned_associations, boundary_points)

        JNIS_delta = 0
        dof = np.count_nonzero(~np.isnan(pruned_associations[1]))*2
        chi2 = stats.chi2.ppf(self.alpha, df=dof)
        max_iter = 5
        i = 0
        minimal_association = np.zeros((pruned_associations.shape))
        minimal_association[0] = np.arange(len(self.scan_data))
        minimal_association[1] = np.nan
        #print("While loop begin.")
        while i < max_iter:
            curr_association = np.copy(pruned_associations)
            start = time.time()
            rm_idx = []
            testable_idxs = np.arange(pruned_associations.shape[1])[~np.isnan(pruned_associations[1])]
            for index in testable_idxs:
                rm_idx.append(index)

                if len(rm_idx) > 10 or len(testable_idxs) <=10:
                    curr_pairing = pruned_associations[1, rm_idx]
                    pruned_associations[1, rm_idx] = np.nan
                    JNIS_new = self.calc_JNIS(pruned_associations, boundary_points)
                    JNIS_new_delta = JNIS-JNIS_new
                    if JNIS_new_delta > JNIS_delta:
                        JNIS_delta = JNIS_new_delta
                        curr_association = np.copy(pruned_associations)
                    pruned_associations[1, rm_idx] = curr_pairing
                    rm_idx = []
            end = time.time()
            if end-start > 1:
                logging.warn("Long JCBB loop time: {}".format(end-start))

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
        #print("While loop complete.")

        unassociated_measurements = minimal_association[0, np.isnan(minimal_association[1])]
        compat_boundaries = {}
        for measurement in unassociated_measurements:
            boundary_idxs = np.where(individual_compatibilities[int(measurement),:] == 1)[0]

            selected_boundaries = set(boundary_idxs)
            min_asso_vals = np.unique(minimal_association[1])
            min_asso_vals = min_asso_vals[~np.isnan(min_asso_vals)]

            selected_boundaries.add(np.nan)

            selected_boundaries = np.setdiff1d(np.array(list(selected_boundaries)), min_asso_vals)

            compat_boundaries[measurement] = list(selected_boundaries)
        st = time.time()
        assigned_associations = self.branch_and_bound(unassociated_measurements, minimal_association, compat_boundaries, boundary_points)
        et = time.time()
        return assigned_associations


    def branch_and_bound(self, unassociated_measurements, minimal_association, compat_boundaries, boundary_points):
        self.best_JNIS = np.inf
        self.best_num_associated = np.count_nonzero(~np.isnan(minimal_association[1]))
        self.best_association = np.copy(minimal_association)
        boundaries_taken = set()
        self.unassociated_measurements = unassociated_measurements
        st = time.time()
        try:
            print("DFS begin.")
            self.DFS(0, minimal_association, compat_boundaries, boundary_points, boundaries_taken)
        except RecursionStop:
            #print("DFS complete!")
            pass
        et = time.time()
        print("DFS time {}".format(et-st))
        # jnis = self.calc_JNIS(self.best_association, boundary_points)
        joint_compat = self.check_compat(self.best_JNIS, DOF =np.count_nonzero(~np.isnan(self.best_association[1]))*2)
        if joint_compat:
            #print("Best JNIS {}".format(self.best_JNIS))
            return self.best_association
        else:
            return np.zeros((self.best_association.shape))


    def DFS(self, level, association, compat_boundaries, boundary_points, boundaries_taken):
        self.recursion += 1
        if self.recursion >= 200:
            print("RECURSIONSTOP")
            raise RecursionStop
        boundaries_taken = boundaries_taken.copy()
        avail_boundaries = compat_boundaries[self.unassociated_measurements[level]]
        # print(avail_boundaries)
        for next_boundary in avail_boundaries:
            # print("Next boundary {}".format(next_boundary))
            isValidBoundary = next_boundary not in boundaries_taken or np.isnan(next_boundary)
            if isValidBoundary and level < len(self.unassociated_measurements):
                # st = time.time()
                test_association = np.copy(association)
                test_association[1,int(self.unassociated_measurements[level])] = next_boundary #2xn
                JNIS = self.calc_JNIS(test_association, boundary_points, DFS = True)
                joint_compat = self.check_compat(JNIS, DOF =np.count_nonzero(~np.isnan(test_association[1]))*2)
                num_associated = np.count_nonzero(~np.isnan(test_association[1]))
                update = False
                # et = time.time()
                # print("{},".format(et-st))

                if joint_compat and num_associated >= self.best_num_associated:
                    if num_associated == self.best_num_associated:
                        if JNIS < self.best_JNIS:
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
            # print("JNIS {}, DOF {}, chi2 {}".format(JNIS, DOF, chi2))
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


    def calc_JNIS(self, association, boundary_points, indiv= False, i = 0, DFS = False):
        #want JNIS to output vector of JNIS's if individual
        #want single JNIS if joint.

        bndry_points_idx = association[1][~np.isnan(association[1])].astype(int)
        z_hat_idx = association[0][~np.isnan(association[1])].astype(int)

        associated_points = boundary_points[bndry_points_idx]

        if len(associated_points) == 0:
            return 0
        if indiv:
            if self.firstrun:
                g, G = self.calc_g_and_G(associated_points, indiv)
                self.h = self.calc_h(g)
                self.g = g

                R = self.calc_R(associated_points, indiv)

                G2 =  G.reshape((G.shape[0]*2, 2))
                H, Hs = self.calc_Jacobian_H(g, G, G2, indiv)
                self.S = self.calc_S(H,R, indiv)
                S = self.S
                h = self.h

                Rs = block_diag(*R)
                # breakpoint()
                S2= self.calc_S(Hs,Rs, indiv=False)
                self.L = np.linalg.cholesky(S2)
                self.firstrun = False
            else:
                S = self.S
                h = self.h
        else:
            idxs = np.zeros((bndry_points_idx.shape[0]*2), dtype=int)
            idxs[::2] = bndry_points_idx*2
            idxs[1::2]=bndry_points_idx*2+1
            L = self.L[idxs[:,None],idxs]

        if indiv:
            z_hat = self.scan_data[z_hat_idx]
            a = (z_hat-h)
            b = np.linalg.inv(S)
            JNIS = np.einsum('ki,kij,kj->k', a, b, a)*0.04
        else:
            h = self.h[bndry_points_idx]
            z_hat = self.scan_data[z_hat_idx].flatten()
            h = h.flatten()
            a = (z_hat-h)
            L_t = torch.from_numpy(L)
            a_t = torch.from_numpy(a).unsqueeze(1)
            y, LU = torch.solve(a_t, L_t) #or solve_triangular, with lower = True??
            JNIS = (np.linalg.norm(y.detach().numpy())**2) * 0.04
        return JNIS

    def calc_R(self, associated_points, indiv):
        #https://dspace.mit.edu/handle/1721.1/32438#files-area
        R_indiv = np.array([[0.001, 0], [0,0.001]])
        R_stacked = np.zeros((len(associated_points), 2,2))
        R_stacked[:] = R_indiv
        return R_stacked


    # def calc_u(self, f):
    #     r = np.sqrt(f[:,0]**2+f[:,1]**2)
    #     theta = np.arctan2(f[:,1], f[:,0])
    #     return np.array([r,theta]).T

    def calc_h(self, g):
        h = g
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
            G[:] = np.eye(2).T
        else:
            G = np.tile(np.eye(2).T, (associated_points.shape[0], 1))

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
                g[index] = np.array(np.array([x, y])- alpha_beta_arr).T
            else:
                g[index] = np.array([x, y])+np.array([gamma, delta])-alpha_beta_arr

        return g, G



    def calc_Jacobian_H(self, g, G, G2, indiv):
        # U, Us = self.calc_U(g, indiv)
        H = G
        Hs = G2
        return H, Hs

    # def calc_U(self, g, indiv):
    #     r = np.sqrt(g[:,0]**2+g[:,1]**2)
    #     U = (np.array([[r*g[:,0], r*g[:,1]],[-g[:,1], g[:,0]]]))/r**2

    #     U_matrices = tuple([U[:,:,i] for i in range(U.shape[2])])
    #     Us =  block_diag(*U_matrices)
    #     return U, Us


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
        selected_scan_x = selected_scan_pts[:,0]
        selected_scan_y = selected_scan_pts[:,1]
        scan_x = scan_data[:,0]
        scan_y = scan_data[:,1]


        #scan data points plot
        plt.scatter(scan_x+xs[0], scan_y+xs[1], c="b", marker="o", alpha = 0.5, label="Scan Data")
        # for i in range(scan_x.shape[0]):
        #     plt.text(scan_x[i], scan_y[i], str(i), size = "xx-small")

        #boundary points plot
        plt.scatter(boundary_points[:,0]+track[0], boundary_points[:,1]+track[1], c="orange", marker="o", alpha = 0.5, label="Boundary Points")
        # for i in range(boundary_points.shape[0]):
        #     plt.text(boundary_points[i,0]+track[0], boundary_points[i,1]+track[1], str(i), size = "xx-small")


        #SELECTED/PAIRED POINTS
        plt.scatter(selected_scan_x+xs[0], selected_scan_y+xs[1], c="red", marker="v", label="Paired Scan Points")
        for i in range(selected_scan_x.shape[0]):
            plt.text(selected_scan_x[i]+xs[0], selected_scan_y[i]+xs[1], str(i))

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
        ax.scatter(boundary_phi, boundary_r, c="orange",marker = "o", alpha=0.5, label="Boundary Points")
        ax.scatter(scan_data[:,1], scan_data[:,0], c="b", marker="o", alpha=0.5,label="Scan Data")

        ax.scatter(selected_boundary_phi, selected_boundary_r, c="black",marker = "v", alpha=0.5)
        for i in range(selected_boundary_phi.shape[0]):
            plt.text(selected_boundary_phi[i], selected_boundary_r[i], str(i))

        ax.scatter(selected_scan_pts[:,1], selected_scan_pts[:,0], c="red", marker = "v",label="Paired Scan Points")
        for i in range(selected_scan_pts[:,1].shape[0]):
            plt.text(selected_scan_pts[:,1][i], selected_scan_pts[:,0][i], str(i))

        ax.set_xlim(0.7*np.pi, 0.9*np.pi)
        # ax.set_ylim(27, 38)

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


    initial_association= np.load("tests/npy_files/initial_association.npy")
    P = np.load("tests/npy_files/P.npy")

    psi = np.load("tests/npy_files/psi.npy")
    scan_data = np.load("tests/npy_files/scan_data.npy")
    xs = np.load("tests/npy_files/xs.npy")
    track = np.load("tests/npy_files/track.npy")
    boundary_points = np.load("tests/npy_files/boundary_points.npy")
    jc.assign_values(xs = xs, scan_data = scan_data, track = track, P = P, static=False, psi=psi)

    # for i in range(100):
    # np.random.seed(2003)
    # xs = [0,0]
    # n = 1000
    # initial_association = np.zeros((2, n))
    # initial_association[0] = np.arange(n)
    # initial_association[1] = np.random.randint(0, 10, n)
    # scan_data = np.zeros((n,2))
    # scan_data[:,0] = np.random.uniform(28, 38, n) #1st row is ranges
    # scan_data[:,1] = np.random.uniform(0.6, 1.1, n) #2nd row is angles (radians)


    # # scan_data[:,0], scan_data[:,1] = convert_scan_polar_cartesian(scan_data)
    # # scan_data[:,0] = np.random.normal(33,sigma, n) #1st row is ranges
    # # scan_data[:,1] =  np.random.normal(0.85,sigma, n) #1st row is ranges

    # n_boundary = 500
    # boundary_points= np.zeros((n_boundary,2))
    # boundary_points[:,0] = np.random.uniform(-3, 3, n_boundary) #x coord, relative to track coordinate
    # boundary_points[:,1] = np.random.uniform(-3, 3, n_boundary) #y coord, relative to track coordinate
    # # boundary_points[:,0] = np.array([0,0,0,0,0,1,2,3,4,5]) #x coord, relative to track coordinate
    # # boundary_points[:,1] = np.array([0,1,2,3,4,0,0,0,0,0]) #y coord, relative to track coordinate


    # track = [20, 25, 0]
    # P = np.eye(2)*0.2
    # static = False
    # psi = 0 #sensor angle. Will work if adjusted for JCBB running purposes, but
    #         #don't change-- need to refactor a bit to make the plot look nice too.
    # # xs = np.load("xs.npy")
    # # scan_data = np.load("scan_data.npy")
    # scan_x, scan_y = Helper.convert_scan_polar_cartesian_joint(scan_data)
    # # plt.scatter(scan_x, scan_y, c="b", marker="o", alpha = 0.5, label="Scan Data")
    # # plt.xlim(-10,3)
    # # plt.ylim(-1, 9)

    # # P = np.load("P.npy")
    # # track = np.load("track.npy")
    # # static = False
    # # psi = np.load("psi.npy")
    # # initial_association = np.load("initial_association.npy")
    # # boundary_points = np.load("boundary_points.npy")
    # # plt.scatter(boundary_points[:,0]+track[0], boundary_points[:,1]+track[1], c="orange", marker="o", alpha = 0.5, label="Scan Data")
    # # plt.xlim(-10,3)
    # # plt.ylim(-1, 9)
    # # plt.show()

    # jc.assign_values(xs, scan_data, track, P, static, psi)


    starttime = time.time()
    asso = jc.run(initial_association, boundary_points)
    endtime = time.time()
    runtime = endtime-starttime
    print(runtime)
    if np.any(asso):
        plot_association(asso, polar=False)
    else:
        print("No associations found.")
        # plot_association(asso)
