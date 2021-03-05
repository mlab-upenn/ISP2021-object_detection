import numpy as np
from scipy import stats
from helper import Helper


class InitAndMerge:
    def __init__(self):
        self.alpha = 1-0.95

    def run(self, tentative, state):
        self.tentative = tentative  #=== new tracks #xt is a 1D list where each element corresponds with the id of
                    #new tentative track T. 
        self.state = state
        self.static_check()
        self.dynamic_check()


    def static_check(self):
        static_check = np.zeros((self.tentative.shape[0]))
        chi2 = stats.chi2.ppf(self.alpha, df=3)

        h = self.tentative[:, 3:6] #fictious measurement model
        h = np.reshape(h, (h.shape[0],1,-1)) #reshape into 3D matrix, where vectors of 3 are stacked on top of each other
        #optimization: move creation of H and z_hat into init to avoid calling every time.
        H= np.zeros((h.shape[0], 3, 6))
        H[:] = np.hstack((np.eye(3), np.ones((3,3))))
        S = self.state.static_background.kf.P #do I need to reshape P to make S into a 3D matrix?
        z_hat = np.zeros((h.shape[0], 3, 1))
        z_hat[:] = np.zeros((3,1))

        a = z_hat - h
        b = np.linalg.inv(S)
        val = np.einsum('ki,kij,kj->k', a, b, a)

        static_check[np.where(val <= chi2)] = 1 #1 indicates that the track has been flagged by
                                                #the static check, and should be merged into the static backgorund.

        for idx in np.where(static_check):
            self.state.merge_tracks(self.tentative[idx], None, kind="static")
            self.tentative.pop(idx)
    

    def dynamic_check(self):
        chi2 = stats.chi2.ppf(self.alpha, df=3)

        h2 = np.zeros((self.tentative.shape[0], 3, 1))
        R_phi_e = np.zeros((self.tentative.shape[0], 2, 2))
        R_phi_rot_e = np.zeros((self.tentative.shape[0], 2, 2))
        vel_e = np.zeros((self.tentative.shape[0], 2, 1))
        pos_e = np.zeros((self.tentative.shape[0], 2, 1))

        pos_t = self.tentative[0:2]
        pos_t = pos_t.reshape(-1, 1, pos_t.shape[1])
        vel_t = self.tentative[3:5]
        vel_t = vel_t.reshape(-1, 1, vel_t.shape[1])

        HT = np.zeros((self.tentative.shape[0], 3, 6))

        HE = np.zeros((self.tentative.shape[0], 3, 6))
        D = np.array([[0,1,0],[-1,0,0]])
        
        z_hat = np.zeros((self.tentative.shape[0], 3, 1))
        z_hat[:] = np.zeros((3,1))

        joint_check = np.zeros((self.state.num_dynamic_tracks(), self.tentative.shape[0]))

        for idx, track in self.state.dynamic_tracks.items():
            #generate a bunch of stacked matrices to run all of this 
            # for all the tentative tracks at once.
            vel_e[:] = np.array([[track.kf.x[3], track.kf.x[4]]]).T
            pos_e[:] = np.array([[track.kf.x[0], track.kf.x[1]]]).T

            R_phi_e[:] = Helper.compute_rot_matrix(-track.kf.x[2])
            R_phi_rot_e[:] = Helper.compute_rot_matrix(np.pi/2- track.kf.x[2])

            h2[:, 0:2] = R_phi_e @ (vel_t-vel_e)-track.kf.x[5]*R_phi_rot_e @ (pos_t-pos_e)#need to adjust this with einsum later.
            h2[:, 2] = 0

            HT[:,0:2, 0:2] = -track.kf.x[5]*R_phi_rot_e
            HT[:,0:2, 3:5] = R_phi_e
            HT[:,2,5] = 1

            HE[:,0:2, 0:2] = track.kf.x[5]*R_phi_rot_e
            HE[:,0:2, 2] = D@h2
            HE[:,0:2, 3:5] = -R_phi_e
            HE[:,0:2, 5] = R_phi_rot_e @ (pos_t - pos_e)
            HE[:,2, 5] = 1

            H = HT-HE

            S = H.T@track.kf.P@H #R = 0


            a = z_hat - h2
            b = np.linalg.inv(S)
            val = np.einsum('ki,kij,kj->k', a, b, a)

            joint_check[idx][np.where(val <= chi2)] = 1 #1 indicates that the track has been flagged by joint check.

        idxs = zip(*np.where(joint_check))
        for track_id, target_id in idxs:
            self.state.merge_tracks(self.tentative[track_id], target_id, kind="dynamic")
            self.tentative.pop(track_id)
