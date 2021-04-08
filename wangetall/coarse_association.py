import random
import numpy as np
import matplotlib.pyplot as plt
import perception.icp

import perception.cluster


class Coarse_Association():
    def __init__(self, C):
        """
        An implementation of the Coarse Association to differentiate between static background and dynamic track.
        """
        self.C = C
        self.ICP = perception.icp.ICP()

    def run(self, Z, state):
        #3.: (x, P, A) <- ASSOCIATEANDUPDATEWITHSTATIC(x, P, C)

        self.state=state
        if self.state.static_background.xb.size != 0:
            A, static_point_pairs = self.associateAndUpdateWithStatic(Z)
            #4. C <- C/A
            for key in A.keys():
                del self.C[key]
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
                dynamic_associations[key] = A_d
                dynamic_point_pairs[key] = point_pairs
                #7. C <- C/A
                for key in A_d.keys():
                    print(key)
                    used_clusters.add(key)

        else:
            dynamic_associations = {}
            dynamic_point_pairs = {}
        #9. for all C do
        new_tracks = {}
        for key in self.C.keys():
            #10. (x, P) INITIALISENEWTRACK(x, P, C)
            if key not in used_clusters:
                new_tracks[key] = self.C[key]
        cl_w_asso = []
        for key in dynamic_associations.keys():
            if len(dynamic_associations[key]) > 0:
                cl_w_asso.append(key)

        return A, static_point_pairs, dynamic_associations, dynamic_point_pairs, new_tracks #A_d, new_tracks

    def associateAndUpdateWithStatic(self, Z):
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
            # if trackid == 7:
            #     # plt.figure()
            #     # plt.scatter(P[:,0], P[:,1], c="blue")
            #     # track = self.state.dynamic_tracks[trackid]
            #     # plt.scatter(track.xp[:,0]+track.kf.x[0], track.xp[:,1]+track.kf.x[1], c="orange", marker="o", alpha = 0.7, label="Boundary Points")
            #     breakpoint()

            if dynamic:
                dynamic_C[key] = self.C[key]
                point_idx_pairs = point_idx_pairs+point_pairs
        return dynamic_C, point_idx_pairs
