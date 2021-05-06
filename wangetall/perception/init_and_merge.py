import numpy as np
from scipy import stats
from perception.helper import Helper
import sys
import logging
class InitAndMerge:
    def __init__(self):
        self.alpha = 1-0.90

    def run(self, tentative, state):
        self.tentative = tentative #xt is a 1D list where each element corresponds with the id of
                    #new tentative track T. 
        self.state = state
        self.static_check()
        if len(self.tentative) > 0:
            self.dynamic_check()


    def static_check(self):
        static_check = np.zeros(len(self.tentative))
        chi2 = stats.chi2.ppf(self.alpha, df=3)

        h = np.zeros((len(self.tentative), 3))
        for idx, track_id in enumerate(self.tentative):
            h[idx] = [self.state.dynamic_tracks[track_id].kf.x[3], self.state.dynamic_tracks[track_id].kf.x[4], self.state.dynamic_tracks[track_id].kf.x[5]]        
        
        #fictious measurement model
        #optimization: move creation of H and z_hat into init to avoid calling every time.
        # H= np.zeros((h.shape[0], 3, 6))
        # H[:] = np.hstack((np.eye(3), np.ones((3,3))))
        S = np.zeros((len(self.tentative), 3, 3))
        for idx, track_id in enumerate(self.tentative):
            S[idx] = self.state.dynamic_tracks[track_id].kf.P[3:6,3:6]   

        
        z_hat = np.zeros((h.shape[0], 3))
        z_hat[:] = np.zeros((3))

        a = z_hat - h
        b = np.linalg.inv(S)

        val = np.einsum('ki,kij,kj->k', a, b, a)

        static_check[np.where(val <= chi2)] = 1 #1 indicates that the track has been flagged by
                                                #the static check, and should be merged into the static backgorund.

        for idx in np.where(static_check)[0]:
            logging.info("Merging {} into static.".format(self.tentative[idx]))
            self.state.merge_tracks(self.tentative[idx], None, kind="static")
        
        self.tentative = [x for i, x in enumerate(self.tentative) if i not in np.where(static_check)[0]]
    

    def dynamic_check(self):
        chi2 = stats.chi2.ppf(self.alpha, df=3)

        h2 = np.zeros((len(self.tentative), 4, 1))
        # R_phi_e = np.zeros((len(self.tentative), 2, 2))
        # R_phi_rot_e = np.zeros((len(self.tentative), 2, 2))
        vel_e = np.zeros((len(self.tentative), 2))
        pos_e = np.zeros((len(self.tentative), 2))

        pos_t = np.zeros((len(self.tentative), 2))
        for idx, track_id in enumerate(self.tentative):
            pos_t[idx] = self.state.dynamic_tracks[track_id].kf.x[0:2] 
        

        vel_t = np.zeros((len(self.tentative), 2))
        for idx, track_id in enumerate(self.tentative):
            vel_t[idx] = self.state.dynamic_tracks[track_id].kf.x[3:5] 

        
        #need to change all of these...
        # pos_t = pos_t.reshape(-1, 1, pos_t.shape[1])
        # vel_t = vel_t.reshape(-1, 1, vel_t.shape[1])

        # HT = np.zeros((len(self.tentative), 3, 6))

        # HE = np.zeros((len(self.tentative), 3, 6))
        # D = np.zeros((len(self.tentative), 2, 3))
        # D[:] = np.array([[0,1,0],[-1,0,0]])
        
        z_hat = np.zeros((len(self.tentative), 4, 1))
        z_hat[:] = np.zeros((4,1))

        joint_check = np.zeros((self.state.num_dynamic_tracks(), len(self.tentative)))
        track_arr = np.array(list(self.state.dynamic_tracks.keys()))
        for idx, track_id in enumerate(track_arr):
            track = self.state.dynamic_tracks[track_id]
            #generate a bunch of stacked matrices to run all of this 
            # for all the tentative tracks at once.
            vel_e[:] = np.array([track.kf.x[3], track.kf.x[4]])
            pos_e[:] = np.array([track.kf.x[0], track.kf.x[1]])

            # R_phi_e[:] = Helper.compute_rot_matrix(-track.kf.x[2])
            # R_phi_rot_e[:] = Helper.compute_rot_matrix(np.pi/2- track.kf.x[2])
            dvel= vel_t-vel_e
            dpos = pos_t-pos_e
            dvel = dvel.reshape(-1, dvel.shape[1], 1)
            dpos = dpos.reshape(-1, dpos.shape[1], 1)

            
            # h2[:, 0:2] = R_phi_e @ dvel-track.kf.x[5]*R_phi_rot_e @ dpos

            # h2[:, 0:2] = np.einsum("ijk, ijl -> ijl", R_phi_e, dvel) - track.kf.x[5]*np.einsum("ijk, ijl -> ijl", R_phi_rot_e, dpos)
            h2[:, 0:2] = dpos


            h2[:, 2:4] = dvel

            # HT[:,0:2, 0:2] = -track.kf.x[5]*R_phi_rot_e
            # HT[:,0:2, 3:5] = R_phi_e
            # HT[:,2,5] = 1

            # HE[:,0:2, 0:2] = track.kf.x[5]*R_phi_rot_e

            # HE[:,0:2, 2] =(D@h2).transpose(0, 2, 1)
            # HE[:,0:2, 3:5] = -R_phi_e
            # HE[:,0:2, 5] = np.einsum("ijk, ijl -> ijl", R_phi_rot_e, dpos).transpose(0,2,1)
            # HE[:,2, 5] = 1

            # H = HT-HE
            P = np.zeros((len(self.tentative), 4, 4))
            # P[:] = track.kf.P[3:6,3:6]

            P[:,0:2,0:2] = track.kf.P[0:2,0:2]
            P[:,2:4,2:4] = track.kf.P[3:5,3:5]

            S = P
            # S = H.transpose(0,2, 1)@P@H

            a = z_hat - h2
            b = np.linalg.inv(S)

            val = np.einsum('kil,kij,kji->k', a, b, a)
            joint_check[idx, np.where(val <= chi2)] = 1 #1 indicates that the track has been flagged by joint check.
            # breakpoint()
        idxs = zip(*np.where(joint_check))
        rmed_list = []
        merged_list = []
        for i, j in idxs:
            track_id = track_arr[i]
            target_id = self.tentative[j]
            if track_id != target_id and j not in rmed_list and track_id not in merged_list:
                logging.info("Merging dynamic Track {} with dynamic Track {}".format(target_id, track_id))
                try:
                    self.state.merge_tracks(target_id, track_id, kind="dynamic")
                except:
                    breakpoint()
                rmed_list.append(j)
                merged_list.append(target_id)
        self.tentative = [x for i, x in enumerate(self.tentative) if i not in rmed_list]
