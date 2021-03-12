import random
import numpy as np
import matplotlib.pyplot as plt
from icp import ICP
from cluster import Cluster

class Coarse_Association():
    def __init__(self, icp):#, state):
        self.icp = icp
        #self.state = state

    def most_frequent(self,List):
        return max(set(List), key = List.count)

    def associateAndUpdateWithDynamic(self, lidar, lidar_static_prev, C, C_static_prev, C_roots_arr, C_static_prev_roots_arr):
        static_C = {}
        print(len(C))
        indicies = self.icp.run(lidar_static_prev, lidar) #outputs indicies corresponding to points in lidar associated with nn in lidar_prev
        C_static_prev_roots = list(dict.fromkeys(C_static_prev_roots_arr))
        C_roots = list(dict.fromkeys(C_roots_arr))
        matched_b_indices = []
        print("dynamic roots:",C_static_prev_roots)
        print("C roots:",C_roots)

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
        not_static_C_idx = [i for i, x in enumerate(matched) if x == -1]
        return not_static_C_idx

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
        #if self.state.static_background.xb.size != 0:
        #3.: (x, P, A) <- ASSOCIATEANDUPDATEWITHSTATIC(x, P, C)
        not_static_C_idx = self.associateAndUpdateWithStatic(lidar, lidar_static_prev, C, C_static_prev, C_roots_arr, C_static_prev_roots_arr)
        print(not_static_C_idx)
        if(len(not_static_C_idx) > 0):
            #4. C <- C/A
            #print("non-matched cluster:",C_associations_static)
            keys_list = np.array(list(C))
            key_to_keep = keys_list[not_static_C_idx]
            C_static = C.copy()
            for item in key_to_keep:
                C_static = {k: v for k, v in C_static.items() if k != item}
            C_roots_arr = [x for x in C_roots_arr if x in key_to_keep]

            for key in C_static.keys():
                del C[key]

            print("total clusters - static clusters:", len(C))
        else:
            C_static = C.copy()

            for key in C_static.copy():
                del C[key]

        #5. for i = 1,2,.,Nt do
        #if len(self.state.dynamic_tracks) != 0:
        dynamic_point_pairs = []
        dynamic_associations = {}
        #for key, track in self.state.dynamic_tracks.items():
        #dynanmic_P = track.xp
        #6. (x, P, A) <- ASSOCIATEANDUPDATEWITHDYNAMIC(x, P, C, i)
        not_dynamic_C_idx = self.associateAndUpdateWithDynamic(lidar, lidar_dynamic_prev, C, C_dynamic_prev, C_roots_arr, C_dynamic_prev_roots_arr)
        if (len(not_dynamic_C_idx) > 0):
            keys_list = np.array(list(C))
            key_to_keep = keys_list[not_dynamic_C_idx]
            C_dynamic = C.copy()
            for item in key_to_keep:
                C_dynamic = {k: v for k, v in C_dynamic.items() if k != item}
            C_roots_arr = [x for x in C_roots_arr if x in key_to_keep]
            #7. C <- C/A
            for key in C_dynamic.keys():
                del C[key]

        else:
            C_dynamic = C.copy()

            for key in C_dynamic.keys():
                del C[key]

        print("total clusters - static - dynamic clusters:", len(C))

        #9. for all C do
        new_tracks = {}
        if(len(C) > 0):
            for key in C.keys():
                #10. (x, P) INITIALISENEWTRACK(x, P, C)
                P = lidar[C[key]]
                new_tracks[key] = C[key]

        print("static:",C_static,"dynamic:", C_dynamic,"new:", new_tracks)
        return C_static, C_dynamic, new_tracks


## ------------------ TEST ----------------------------- ###
def forward(dt):
    """Propagates forward all tracks
    based on current transition model"""
    F = self.calc_F(dt)
    Q = self.calc_Q(dt)

    for id, track in self.state.dynamic_tracks.items():
        track.kf.F = F
        track.kf.Q = Q
        track.kf.predict()

def update(dt, data, state):
        self.state = state
        self.forward(dt)
        y, x = data[:,0], data[:,1]

        for key, points in new_tracks.items():
            idx = self.state.create_new_track(self.laserpoints, points)

        tracks_to_init_and_merge = []
        print("to init: {}".format(tracks_to_init_and_merge))
        for track_id, track in self.state.dynamic_tracks.items():
            print("Track id {}, num_viewings {}".format(track_id, track.num_viewings))
            if track.num_viewings == track.mature_threshold:
                tracks_to_init_and_merge.append(track_id)
        if len(tracks_to_init_and_merge) > 0:
            self.InitAndMerge.run(tracks_to_init_and_merge, self.state)

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


    # lidarUpdater = lidarUpdater()
    # time  = time.time()
    # prev_Lidar_callback_time = time
    # dt = time - self.prev_Lidar_callback_time
    # lidarUpdater.update(dt, data, self.state)
    # static_background_state = tracker.state.static_background

    cl = Cluster()
    C, C_roots_arr = cl.cluster(lidar)
    C_prev, C_prev_roots_arr  = cl.cluster(lidar_prev)
    C_static_prev, C_static_prev_roots_arr = cl.cluster(static_background_prev)
    C_dynamic_prev, C_dynamic_prev_roots_arr = cl.cluster(dynamic_tracks_prev)

    icp = ICP()
    ca = Coarse_Association(icp)
    plt.scatter(lidar[:,0], lidar[:,1])
    plt.title("current lidar scan")
    plt.show()
    for key in C_prev.keys():
        P = lidar_prev[C_prev[key]]
        plt.scatter(P[:,0], P[:,1])
    plt.title("Previous scan")
    plt.show()
    for key in C.keys():
        P = lidar[C[key]]
        plt.scatter(P[:,0], P[:,1])
    plt.title("CLusters")
    plt.show()
    C_static, C_dynamic, C_new = ca.run(lidar, static_background_prev, dynamic_tracks_prev, C, C_static_prev, C_dynamic_prev, C_roots_arr, C_static_prev_roots_arr, C_dynamic_prev_roots_arr)
    for key in C_static.keys():
        P = lidar[C_static[key]]
        plt.scatter(P[:,0], P[:,1])
    plt.title("seperated static background")
    plt.show()
    for key in C_dynamic.keys():
        P = lidar[C_dynamic[key]]
        plt.scatter(P[:,0], P[:,1])
    plt.title("separated dynamic tracks")
    plt.show()
    for key in C_new.keys():
        P = lidar[C_dynamic[key]]
        plt.scatter(P[:,0], P[:,1])
    plt.title("new tracks")
    plt.show()
