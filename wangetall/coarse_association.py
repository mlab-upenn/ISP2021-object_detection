import random
import numpy as np
import matplotlib.pyplot as plt
import perception.icp
from perception.cluster import Cluster
import copy

from pdb import set_trace as bp
import time
class Coarse_Association():
    def __init__(self):
        self.cl = Cluster()

    def most_frequent(self,List):
        return max(set(List), key = List.count)

    def run(self, lidar, state): #Q_d, dynamic_tracks_dict):
        #3.: (x, P, A) <- ASSOCIATEANDUPDATEWITHSTATIC(x, P, C)
        self.state = state
        self.C, self.C_roots_arr = self.cl.cluster(lidar)

        if self.state.static_background.xb.size != 0:
            static_associations, static_point_pairs = self.associateAndUpdateWithStatic(lidar, self.state.static_background.xb)
            #4. C <- C/A
        #     for key in A.keys():
        #         del C[key]
        else:
            static_associations = {}
            static_point_pairs = []

        #5. for i = 1,2,.,Nt do
        if len(self.state.dynamic_tracks) != 0:
            dynamic_point_pairs = {}
            dynamic_associations = {}
            for key, track in self.state.dynamic_tracks.items():
                dynanmic_P = track.xp+track.kf.x[0:2]
                #6. (x, P, A) <- ASSOCIATEANDUPDATEWITHDYNAMIC(x, P, C, i)

                A_d, point_pairs = self.associateAndUpdateWithDynamic(lidar, dynanmic_P)
                dynamic_associations[key] = A_d
                dynamic_point_pairs[key] = point_pairs
                #7. C <- C/A
                for key in A_d.keys():
                    del self.C[key]
        else:
            dynamic_associations = {}
            dynamic_point_pairs = {}

        #9. for all C do
        new_tracks = {}
        for key in self.C.keys():
            #10. (x, P) INITIALISENEWTRACK(x, P, C)
            new_tracks[key] = self.C[key]
        # print("new tracks:",len(new_tracks),"dynamic tracks:",len(dynamic_associations),"static background:",len(static_associations))
        # bp()
        return static_associations, static_point_pairs, dynamic_associations, dynamic_point_pairs, new_tracks #A_d, new_tracks

    def associateAndUpdateWithStatic(self, lidar, static_background):
        pass
    def associateAndUpdateWithDynamic(self, lidar, lidar_dynamic_prev):
        C_dynamic_prev, C_dynamic_prev_roots_arr = self.cl.cluster(lidar_dynamic_prev)
        dynamic_C = {}
        self.C_roots = list(dict.fromkeys(self.C_roots_arr))
        try:
            self.C_roots.remove(-1)
        except ValueError:
            pass
        icp_obj = perception.icp.ICP()
        indices, point_pairs = icp_obj.run(lidar_dynamic_prev, lidar) #outputs indicies corresponding to points in lidar associated with nn in lidar_prev

        C_dynamic_prev_roots = list(dict.fromkeys(C_dynamic_prev_roots_arr))


        # print("C_dynamic_prev_roots", C_dynamic_prev_roots)
        matched_b_indices = []
        # print("dynamic roots:",C_dynamic_prev_roots)
        # print("C roots:",self.C_roots)

        for i, a_root in enumerate(self.C_roots):  # this root represent a cluster
            # for all the element in the cluster, find those that their roots are this root
            b_roots = []  # collect matched point's root
            a_indices = []
            b_indices = []
            for j, i_root in enumerate(self.C_roots_arr):
                if i_root == a_root and indices[j] >= 0:
                    # print("indicies:", len(indicies))
                    b_index = indices[j]  # matched b index
                    # print("b_index:", b_index)
                    # print("C_dynamic_prev_roots_arr:", C_dynamic_prev_roots_arr)
                    b_root = C_dynamic_prev_roots_arr[b_index]
                    b_roots.append(b_root)
                    a_indices.append(j)
                    b_indices.append(b_index)
            #breakpoint()
            # there are matching points for this cluster
            if b_roots != []:
                # print("b_roots:",b_roots)
                best_b_root = self.most_frequent(b_roots)
                # print("best_b_root:",best_b_root)
                # record this matchiing
                try:  # if matched_dst_root is a cluster smaller than Cluster.min_size, it can not be found in the Cluster.roots
                    matched_b_index = C_dynamic_prev_roots.index(best_b_root)  # index of the cluster (clusters.roots has the same ordring as clusters.clusters)
                except:
                    matched_b_indices.append(-1)
                else:
                    # we only allow one to one matching
                    if matched_b_index in matched_b_indices:  # if someone else is matched
                        exist_i = matched_b_indices.index(matched_b_index)
                        # if i am bigger
                        #breakpoint()
                        #print("C_roots_arr == a_root:",np.count_nonzero(self.C_roots_arr[np.where(indicies != -1)] == a_root),"C_roots_arr == self.C_roots[exist_i]", np.count_nonzero(self.C_roots_arr[np.where(indicies != -1)] == self.C_roots[exist_i]))
                        if (np.count_nonzero(np.array(self.C_roots_arr)[np.where(indices != -1)] == a_root) - np.count_nonzero(np.array(self.C_roots_arr)[np.where(indices == -1)] == a_root)) > (np.count_nonzero(np.array(self.C_roots_arr)[np.where(indices != -1)] == self.C_roots[exist_i]) - np.count_nonzero(np.array(self.C_roots_arr)[np.where(indices == -1)] == self.C_roots[exist_i])):
                            matched_b_indices[exist_i] = -1  # someone will be loser
                            matched_b_indices.append(matched_b_index)  # i won
                        else:
                            matched_b_indices.append(-1)  # i lose
                    else:
                        matched_b_indices.append(matched_b_index)

            # there is no matching point for this cluster
                # print("mathcing in loop:",matched_b_indices)
            else:
                matched_b_indices.append(-1)

        matched = np.array(matched_b_indices)
        #breakpoint()
        # print("matched:",matched)
        C_dynamic = copy.deepcopy(self.C)
        not_dynamic_C_idx = [i for i, x in enumerate(matched) if x == -1]
        if (len(not_dynamic_C_idx) > 0):
            keys_list = np.array(list(self.C))
            key_to_keep = keys_list[not_dynamic_C_idx]
            for item in key_to_keep:
                C_dynamic = {k: v for k, v in C_dynamic.items() if k != item}
            self.C_roots_arr = [x if x in key_to_keep else -1 for x in self.C_roots_arr]

        #print(C_dynamic)
        # plt.scatter(lidar_dynamic_prev[:,0],lidar_dynamic_prev[:,1], label="reference points")
        # for key in C_dynamic.keys():
        #     Z = lidar[C_dynamic[key]]
        #     plt.scatter(Z[:,0],Z[:,1], label="assigned dynamic object" )
        # plt.scatter(lidar[:,0],lidar[:,1], label="incoming points", s = 4)
        # plt.legend()
        # plt.show()
        # breakpoint()
        return C_dynamic, point_pairs
