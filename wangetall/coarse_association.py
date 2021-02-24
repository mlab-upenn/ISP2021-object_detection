import random
import numpy as np
import matplotlib.pyplot as plt
import icp
import cluster

class Coarse_Association():
    def __init__(self, C):
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







    #testing
#     for key in C.keys():
#         P = Z[C[key]]
#         plt.scatter(P[:,0], P[:,1])
#     plt.show()



    #testing
#     data = plt.scatter(Z[:,0],Z[:,1])

#     for key in A.keys():
#         static_P = Z[A[key]]
#         static_cloud = plt.scatter(static_P[:,0],static_P[:,1], marker='D')
#         border = plt.scatter(Q[:,0],Q[:,1], marker='x')
#         plt.legend((data, static_cloud, border),
#                     ("laser scan data", "static cloud", "boundary points"))


#     plt.show()


    #testing
#     for key in C.keys():
#         P = Z[C[key]]
#         plt.scatter(P[:,0], P[:,1])
#     plt.show()


    #retrun as well clusters for each dynamic track


#         for key in C.keys():
#             P = Z[C[key]]
#             data = plt.scatter(P[:,0],P[:,1])

        #testing
#         for key in A_d.keys():
#             dynanmic_P = Z[A_d[key]]
#             dynamic_cloud = plt.scatter(dynanmic_P[:,0],dynanmic_P[:,1], marker='D')
#             border = plt.scatter(Q[:,0],Q[:,1], marker='x')
#             plt.legend((data, dynamic_cloud, border),
#                         ("laser scan data", "dynamic cloud", "boundary points"))

#         plt.show()



        #testing
#         for key in C.keys():
#             P = Z[C[key]]
#             plt.scatter(P[:,0], P[:,1])
#         plt.show()
