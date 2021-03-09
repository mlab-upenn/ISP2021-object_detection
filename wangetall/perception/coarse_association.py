import random
import numpy as np
import matplotlib.pyplot as plt
from icp import ICP
from cluster import Cluster

class Coarse_Association():
    def __init__(self, icp):
        self.icp = icp

    def most_frequent(self,List):
        return max(set(List), key = List.count)

    def associateAndUpdateWithStatic(self, lidar, lidar_static_prev, C, C_static_prev, C_roots_arr, C_static_prev_roots_arr):
        static_C = {}
        indicies = self.icp.run(lidar_static_prev, lidar) #outputs indicies corresponding to points in lidar associated with nn in lidar_prev
        C_static_prev_roots = list(dict.fromkeys(C_static_prev_roots_arr))
        C_roots = list(dict.fromkeys(C_roots_arr))

        matched_b_indices = []
        for i, a_root in enumerate(C_roots):  # this root represent a cluster
            # for all the element in the cluster, find those that their roots are this root
            b_roots = []  # collect matched point's root
            a_indices = []
            b_indices = []
            for j, i_root in enumerate(C_roots_arr):
                if i_root == a_root and indicies[i] >= 0:
                    b_index = indicies[j]  # matched b index
                    b_root = C_static_prev_roots_arr[b_index]
                    b_roots.append(b_root)
                    a_indices.append(j)
                    b_indices.append(b_index)

            # there are matching points for this cluster
            if b_roots != []:
                best_b_root = self.most_frequent(b_roots)
                # record this matchiing
                try:  # if matched_dst_root is a cluster smaller than Cluster.min_size, it can not be found in the Cluster.roots
                    matched_b_index = C_static_prev_roots.index(best_b_root)  # index of the cluster (clusters.roots has the same ordring as clusters.clusters)
                except:
                    matched_b_indices.append(-1)
                else:
                    # we only allow one to one matching
                    if matched_b_index in matched_b_indices:  # if someone else is matched
                        exist_i = matched_b_indices.index(matched_b_index)
                        # if i am bigger
                        if np.count_nonzero(C_roots_arr == a_root) > np.count_nonzero(C_roots_arr == C_roots[exist_i]):
                            matched_b_indices[exist_i] = -1  # someone will be loser
                            matched_b_indices.append(matched_b_index)  # i won
                        else:
                            matched_b_indices.append(-1)  # i lose
                    else:
                        matched_b_indices.append(matched_b_index)


            # there is no matching point for this cluster
            else:
                matched_b_indices.append(-1)

        matched = np.array(matched_b_indices)

        not_static_C_idx = [i for i, x in enumerate(matched_b_indices) if x == -1]
        return not_static_C_idx

    def run(self, lidar, lidar_static_prev, lidar_dynamic_prev, C, C_static_prev, C_dynamic_prev, C_roots_arr, C_static_prev_roots_arr, C_dynamic_prev_roots_arr):
        print("clusters:", len(C))
        #3.: (x, P, A) <- ASSOCIATEANDUPDATEWITHSTATIC(x, P, C)
        non_static_indicies = self.associateAndUpdateWithStatic(lidar, lidar_static_prev, C, C_static_prev, C_roots_arr, C_static_prev_roots_arr)

        #4. C <- C/A
        #print("non-matched cluster:",non_static_indicies)
        keys_list = np.array(list(C))
        key_to_del = keys_list[non_static_indicies]
        C_static = C
        for item in key_to_del:
            C_static = {k: v for k, v in C_static.items() if k != item}
        print("static clusters:", len(C_static))
        for key in C_static.keys():
            del C[key]
        print("total clusters - static clusters:", len(C))

        #5. for i = 1,2,.,Nt do
        #for key in dynamic_tracks_dict.keys():
            #dynanmic_P = lidar_prev[dynamic_tracks_dict[key]]

#             plt.scatter(dynanmic_P[:,0],dynanmic_P[:,1])
#             plt.show()
            #6. (x, P, A) <- ASSOCIATEANDUPDATEWITHDYNAMIC(x, P, C, i)
        #A_d = self.associateAndUpdateWithDynamic(Z, dynanmic_P)
        #print("dynamic clusters:", len(A_d))

#             #7. C <- C/A
#             for key in A_d.keys():
#                 del self.C[key]

#             print("cluster-static-dynamic clusters:", len(self.C))
#             # for key in A_d.keys():
#             #     P = Z[A_d[key]]
#             #     plt.scatter(P[:,0], P[:,1], marker = 'x')
#             #     plt.scatter(Z[:,0], Z[:,1])
#             # plt.show()
#         #9. for all C do
#         new_tracks = {}
#         for key in self.C.keys():
#             #10. (x, P) INITIALISENEWTRACK(x, P, C)
#             P = Z[self.C[key]]
#             new_tracks[key] = self.C[key]
#         # for key in new_tracks.keys():
#         #     P = Z[new_tracks[key]]
#         #     plt.scatter(P[:,0], P[:,1])
#         # plt.show()
        return C_static#, A_d, new_tracks


    def associateAndUpdateWithDynamic(self, Z, Q):
        icp_obj = ICP()
        dynamic_C = {}
        for key in self.C.keys():
            P = Z[self.C[key]]
            dynamic = icp_obj.run(Q, P)
            if dynamic:
                dynamic_C[key] = self.C[key]
        return dynamic_C


if __name__ == "__main__":
    # TESTS
    dir = "gym_testing/testing_data/"
    with open(dir + 'lidar0.npy', 'rb') as f:
        lidar_prev = np.load(f)
    with open(dir + 'static_background0.npy', 'rb') as f:
        static_background_prev = np.load(f)
    with open(dir + 'lidar1.npy', 'rb') as f:
        lidar = np.load(f)
    with open(dir + 'static_background1.npy', 'rb') as f:
        static_background = np.load(f)
    with open(dir + 'dynamic_tracks0.npy', 'rb') as f:
        dynamic_tracks_prev = np.load(f)
    with open(dir + 'dynamic_tracks1.npy', 'rb') as f:
        dynamic = np.load(f)

    cl = Cluster()
    C, C_roots_arr = cl.cluster(lidar)
    C_prev, C_prev_roots_arr  = cl.cluster(lidar_prev)
    C_static_prev, C_static_prev_roots_arr = cl.cluster(static_background_prev)
    C_dynamic_prev, C_dynamic_prev_roots_arr = cl.cluster(dynamic_tracks_prev)

    icp = ICP()
    ca = Coarse_Association(icp)

    C_static = ca.run(lidar, static_background_prev, dynamic_tracks_prev, C, C_static_prev, C_dynamic_prev, C_roots_arr, C_static_prev_roots_arr, C_dynamic_prev_roots_arr)
