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

    def run(self, Z, state): #Q_d, dynamic_tracks_dict):
        #3.: (x, P, A) <- ASSOCIATEANDUPDATEWITHSTATIC(x, P, C)
        self.state=state
        if self.state.static_background.xb.size != 0:
            A, static_point_pairs = self.associateAndUpdateWithStatic(Z)
            #4. C <- C/A
            for key in A.keys():
                del self.C[key]
        else:
            A = {}
            static_point_pairs = np.zeros((2,0))



        #5. for i = 1,2,.,Nt do
        if len(self.state.dynamic_tracks) != 0:
            dynamic_point_pairs = []
            dynamic_associations = {}
            for key, track in self.state.dynamic_tracks.items():
                dynanmic_P = track.xp
                #6. (x, P, A) <- ASSOCIATEANDUPDATEWITHDYNAMIC(x, P, C, i)
                A_d, dynamic_point_pairs = self.associateAndUpdateWithDynamic(Z, dynanmic_P)
                dynamic_associations[key] = A_d

                #7. C <- C/A
                for key in A_d.keys():
                    del self.C[key]
        else:
            dynamic_associations = {}
            dynamic_point_pairs = np.zeros((2,0))
            # print("cluster-static-dynamic clusters:", len(self.C))
        #9. for all C do
        new_tracks = {}
        for key in self.C.keys():
            #10. (x, P) INITIALISENEWTRACK(x, P, C)
            # P = Z[self.C[key]]
            new_tracks[key] = self.C[key]
        # for key in A.keys():
        #     P = Z[A[key]]
        #     plt.scatter(P[:,0], P[:,1])
        # plt.show()
        # print("Ad {}".format(A_d))

        return A, static_point_pairs, dynamic_associations, dynamic_point_pairs, new_tracks #A_d, new_tracks

    def associateAndUpdateWithStatic(self, Z):
        #print(Q)
        icp_obj = perception.icp.ICP()
        static_C = {}
        point_pairs = []
        if self.state.static_background.xb.size != 0:
            for key in self.C.keys():
                P = Z[self.C[key]]
                static, point_pairs = icp_obj.run(self.state.static_background.xb, P)
                if static:
                    static_C[key] = self.C[key]
        return static_C, point_pairs

    def associateAndUpdateWithDynamic(self, Z, points):
        icp_obj = perception.icp.ICP()
        dynamic_C = {}
        point_pairs = []
        for key in self.C.keys():
            P = Z[self.C[key]]
            dynamic, point_pairs = icp_obj.run(points, P)
            if dynamic:
                dynamic_C[key] = self.C[key]
        return dynamic_C, point_pairs
