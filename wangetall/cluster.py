import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from itertools import permutations
from collections import defaultdict
import random

class Cluster:
    def __init__(self):
        pass

    @classmethod
    def cluster(self, points):
        tree = self.EMST(points)
        clusters = self.EGBIS(tree)
        return clusters

    def EMST(self, points):
        #https://en.wikipedia.org/wiki/Euclidean_minimum_spanning_tree
        simplices = self.compute_delauney(points)
        graph = self.label_edge(simplices, points)
        tree = self.min_spanning_tree(graph, points)
        return tree


    def compute_delauney(self, points):
        return Delaunay(points).simplices

    def label_edge(self, simplices, points):
        # point_dict = {}
        # for index in range(len(points)):
        #     point_dict[index] = set()
        graph = []
        all_edges = set([tuple(edge) for item in simplices for edge in permutations(item,2)])

        for edge in all_edges:
            dist = np.linalg.norm(points[edge[0]]-points[edge[1]])
            edgelist = list(edge)
            edgelist.append(dist)
            graph.append(edgelist)


            # point_dict[int(edge[0])].add(edge)
        
        return graph

    def find_set(self, parent, point):
        #https://en.wikipedia.org/wiki/Disjoint-set_data_structure
        #could improve with path splitting
        if parent[point] != point:
            return self.find_set(parent, parent[point])
        else:
            return point

    def union(self, parent, rank, parent_origin, parent_dest):
        #from pseudocode listed here https://en.wikipedia.org/wiki/Disjoint-set_data_structure
        x = self.find_set(parent, parent_origin)
        y = self.find_set(parent, parent_dest)

        if x == y:
            return
        if rank[x] < rank[y]:
            (x,y) = (y,x)
        parent[y] = x
        if rank[x] == rank[y]:
            rank[x] += 1
        

    def min_spanning_tree(self, graph, points):
        #Generate via Kruskal's algorithm.
        #https://en.wikipedia.org/wiki/Kruskal%27s_algorithm

        tree = defaultdict(lambda:[])
        sorted_graph = sorted(graph, key=lambda i:i[2])#need to work on this one.
        edge_count = 0
        i = 0

        parent = np.arange(len(points))
        rank = np.zeros((parent.shape))

        while edge_count < len(points)-1:
            # print("i {}".format(i))
            # print("edge count {}".format(edge_count))
            origin, dest, weight = sorted_graph[i]
            i+=1
            parent_origin = self.find_set(parent, origin)
            parent_dest = self.find_set(parent, dest)


            if parent_origin != parent_dest:
                edge_count += 1
                #append to tree
                tree[origin].append([(dest, weight)])
                self.union(parent, rank, parent_origin, parent_dest)
        
        return tree


            
    def EGBIS(self):
        #bad implementation vv -- not sure if it should be trusted
        #https://github.com/devforfu/egbis/blob/master/egbis/segmentation.py
       #the paper vv
       #http://people.cs.uchicago.edu/~pff/papers/seg-ijcv.pdf
       
        return clusters


if __name__ == "__main__":
    points= np.array(random.sample(range(200), 200)).reshape((100,2))
    # print(len(np.unique(points, axis = 0))== len(points))
    # print(len(np.unique(points, axis = 0)))
    cl = Cluster()
    tri = cl.compute_delauney(points)
    graph = cl.label_edge(tri, points)
    tree = cl.min_spanning_tree(graph, points)
    print(tree)
    # plt.triplot(points[:,0], points[:,1], tri)
    # plt.plot(points[:,0], points[:,1], 'o')
    # plt.show()

