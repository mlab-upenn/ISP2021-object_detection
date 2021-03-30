import random
import numpy as np
import matplotlib.pyplot as plt
import perception.icp

import perception.cluster


class Coarse_Association():
    def __init__(self, C):
        """
        An implementation of the Coarse Association to differentiate between static background and dynamic track.
        :param C: The cluster dictonary of key value pair as num. of cluster and indexes of the points
                                                    belonging to that cluster
        :param Z: All points from LiDAR scan (todo dims)
        :param Q_s: Reference static object  (todo dims)
        :param Q_d: reference dynamic objects (todo dims)
        :param dynamic_tracks_dict: Dictonary of key value pair as num. of objects and indexes of the points
                                                    belonging to that object
        :return: A: Dictonary of key value pair as num. of clusters for static background and indexes of the points
                                                  belonging to that clusters
                 A_d: Dictonary of key value pair as num. of clusters for dynamic track and indexes of the points
                                                           belonging to that tracks
                 new_tracks: Dictonary of key value pair as num. of clusters for new tentative dynamic track and indexes
                                                    of the points belonging to that tracks
        """
        self.C = C
        self.ICP = perception.icp.ICP()

    def run(self, Z, state): #Q_d, dynamic_tracks_dict):
        #3.: (x, P, A) <- ASSOCIATEANDUPDATEWITHSTATIC(x, P, C)

        self.state=state
        if self.state.static_background.xb.size != 0:
            A, static_point_pairs = self.associateAndUpdateWithStatic(Z)
            #4. C <- C/A
            for key in A.keys():
                del self.C[key]
            # breakpoint()
        else:
            A = {}
            static_point_pairs = []

        #5. for i = 1,2,.,Nt do
        used_clusters = set()
        if len(self.state.dynamic_tracks) != 0:
            dynamic_point_pairs = {}
            dynamic_associations = {}
            for key, track in self.state.dynamic_tracks.items():

                dynamic_P = track.xp+track.kf.x[0:2]
                A_d, point_pairs = self.associateAndUpdateWithDynamic(Z, dynamic_P, track.id)
                # plt.scatter(dynamic_P[:,0],dynamic_P[:,1], label="dynamic_p")
                # for key in A_d.keys():
                #     plt.scatter(Z[self.C[key]][:,0],Z[self.C[key]][:,1], label="association")
                # plt.legend()
                # plt.show()
                dynamic_associations[key] = A_d
                dynamic_point_pairs[key] = point_pairs
                #7. C <- C/A
                for key in A_d.keys():
                    used_clusters.add(key)
                # for key in A_d.keys():
                #     del self.C[key]
        else:
            dynamic_associations = {}
            dynamic_point_pairs = {}
        #9. for all C do
        new_tracks = {}
        for key in self.C.keys():
            #10. (x, P) INITIALISENEWTRACK(x, P, C)

            # new_tracks[key] = self.C[key]
            if key not in used_clusters:
                new_tracks[key] = self.C[key]
        cl_w_asso = []
        for key in dynamic_associations.keys():
            if len(dynamic_associations[key]) > 0:
                cl_w_asso.append(key)
        print("Clusters with coarse associations: {}".format(cl_w_asso))
        # breakpoint()
        # plt.figure()
        # plt.scatter(Z[self.C[575],0], Z[self.C[575],1], c = "blue")
        # track = self.state.dynamic_tracks[5]
        # plt.scatter(track.xp[:,0]+track.kf.x[0], track.xp[:,1]+track.kf.x[1], c="orange", marker="o", alpha = 0.1, label="Boundary Points")
        # plt.show()
        return A, static_point_pairs, dynamic_associations, dynamic_point_pairs, new_tracks #A_d, new_tracks

    def associateAndUpdateWithStatic(self, Z):
        #print(Q)
        static_C = {}
        point_pairs_list = []
        if self.state.static_background.xb.size != 0:
            for key in self.C.keys():
                P = Z[self.C[key]]+self.state.xs[0:2]
                static, point_pairs = self.ICP.run(self.state.static_background.xb, P)
                # print("static? {}".format(static))
                if static:
                    static_C[key] = self.C[key]
                    point_pairs_list = point_pairs_list+point_pairs
        return static_C, point_pairs_list

    def associateAndUpdateWithDynamic(self, Z, points, trackid):
        dynamic_C = {}
        point_idx_pairs = []
        for key in self.C.keys():
            P = Z[self.C[key]]+self.state.xs[0:2]
            dynamic, point_pairs = self.ICP.run(points, P, key, trackid)
            if dynamic:
                dynamic_C[key] = self.C[key]
                point_idx_pairs = point_idx_pairs+point_pairs
        return dynamic_C, point_idx_pairs
