import random
import numpy as np
import matplotlib.pyplot as plt
import icp
import cluster

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
        print("clusters:", len(self.C))

    def run(self, Z, Q_s, Q_d, dynamic_tracks_dict):
        #3.: (x, P, A) <- ASSOCIATEANDUPDATEWITHSTATIC(x, P, C)
        A = self.associateAndUpdateWithStatic(Z, Q_s)
        print("static clusters:", len(A))

        #4. C <- C/A
        for key in A.keys():
            del self.C[key]

        print("cluster-static clusters:", len(self.C))

        #5. for i = 1,2,.,Nt do
        for key in dynamic_tracks_dict.keys():
            dynanmic_P = Q_d[dynamic_tracks_dict[key]]
            #6. (x, P, A) <- ASSOCIATEANDUPDATEWITHDYNAMIC(x, P, C, i)
            A_d = self.associateAndUpdateWithDynamic(Z, dynanmic_P)
            print("dynamic clusters:", len(A_d))

            #7. C <- C/A
            for key in A_d.keys():
                del self.C[key]

            print("cluster-static-dynamic clusters:", len(self.C))
        #9. for all C do
        new_tracks = {}
        for key in self.C.keys():
            #10. (x, P) INITIALISENEWTRACK(x, P, C)
            P = Z[self.C[key]]
            new_tracks[key] = self.C[key]

        return A, A_d, new_tracks

    def associateAndUpdateWithStatic(self, Z, Q):
        icp_obj = icp.ICP()
        static_C = {}
        for key in self.C.keys():
            P = Z[self.C[key]]
            static = icp_obj.run(Q, P)
            if static:
                static_C[key] = self.C[key]
        return static_C

    def associateAndUpdateWithDynamic(self, Z, Q):
        icp_obj = icp.ICP()
        dynamic_C = {}
        for key in self.C.keys():
            P = Z[self.C[key]]
            dynamic = icp_obj.run(Q, P)
            if dynamic:
                dynamic_C[key] = self.C[key]
        return dynamic_C
