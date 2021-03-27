import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from itertools import permutations
from collections import defaultdict
import random
import time
# from numba import jit

#Numba help:
#https://github.com/f1tenth/f1tenth_gym/blob/exp_py/gym/f110_gym/envs/laser_models.py

class Cluster:
    """Takes in nx2 list of Cartesian coordinates of LiDAR impingements"""
    def __init__(self):
        pass

    def cluster(self, points):
        tree = self.EMST(points)
        clusters = self.EGBIS(tree, points)
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

        all_edges = set([tuple(edge) for group in simplices for edge in permutations(group,2)])
        edges_array = np.array(list(all_edges))
        points_1 = points[edges_array[:,0]]
        points_2 = points[edges_array[:,1]]

        dist = np.linalg.norm(points_1- points_2, axis = 1)
        dist = dist[..., np.newaxis]
        graph = np.hstack((edges_array, dist))

        return graph

    def find_set(self, parent, point):
        #https://en.wikipedia.org/wiki/Disjoint-set_data_structure

        while parent[point] != point:
            point, parent[point] = parent[point], parent[parent[point]]
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

        tree = []
        sorted_graph = sorted(graph, key=lambda i:i[2])
        edge_count = 0
        i = 0

        parent = np.arange(len(points))
        rank = np.zeros((parent.shape))
        while edge_count < len(points)-1:
            origin, dest, weight = sorted_graph[i]
            origin = int(origin)
            dest = int(dest)
            i+=1
            parent_origin = self.find_set(parent, origin)
            parent_dest = self.find_set(parent, dest)
            if parent_origin != parent_dest:
                edge_count += 1
                tree.append([origin, dest, weight])
                self.union(parent, rank, parent_origin, parent_dest)

        return tree



    def EGBIS(self, tree, points):
        """Python implementation based on author C++ implementation
        found here: http://cs.brown.edu/people/pfelzens/segment/"""

        sorted_tree = sorted(tree, key= lambda i:i[2])

        segmentation = Universe(len(points))
        thresholds = np.ones((len(points)))*self.get_tau(1)

        for i in range(len(points)-1):
            vi, vj, w = sorted_tree[i]

            component_i = segmentation.find(vi)
            component_j = segmentation.find(vj)


            if component_i != component_j:
                if w <= thresholds[component_i] and w <= thresholds[component_j]:
                    segmentation.join(component_i, component_j)
                    component_i = segmentation.find(component_i)
                    thresholds[component_i] = w + self.get_tau(segmentation.size(component_i))

        components = segmentation.get_components()

        return components

    def get_tau(self, size):
        k = 50
        return k/size

class Universe:
    """Python implementation based on author C++ implementation
    found here: http://cs.brown.edu/people/pfelzens/segment/"""
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.elts = np.zeros((num_vertices, 3), dtype = int)
        self.elts[:,1] =  1
        self.elts[:,2] = np.arange(num_vertices)

    def find(self, x):
        y = int(x)
        while (y != self.elts[y, 2]):
            y = int(self.elts[y,2])
        self.elts[x,2] = y
        return y

    def join(self, x, y):
        if self.elts[x, 0] > self.elts[y,0]:
            self.elts[y,2] = x
            self.elts[x,1] += self.elts[y,1]
        else:
            self.elts[x, 2] = y
            self.elts[y,1] += self.elts[x,1]
            if self.elts[x, 0] == self.elts[y,0]:
                self.elts[y,0] += 1

    def size(self, x):
        return self.elts[x,1]

    def get_components(self):
        components_dict = defaultdict(lambda:[])
        for i in range(self.num_vertices):
            parent = self.find(i)
            components_dict[parent].append(i)
        return components_dict


    # def get_components(self):
    #     out_arr = np.zeros((self.num_vertices))
    #     for i in range(self.num_vertices):
    #         parent = self.find(i)
    #         out_arr[i] = parent
    #     return out_arr


# def get_cmap(n, name='hsv'):
#     '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
#     RGB color; the keyword argument name must be a standard mpl colormap name.'''
#     return plt.cm.get_cmap(name, n)

if __name__ == "__main__":
    # points= np.array(random.sample(range(2000), 2000)).reshape((1000,2))
    points = np.array((0 + np.random.random((1000,2)) * (100 - 0)))
    cl = Cluster()

    clusters = cl.cluster(points)
    # print(clusters.keys())
    # print(clusters)



    # cmap = get_cmap(len(points))
    plt.figure()
    for key in clusters.keys():
        selected_points = points[clusters[key]]
        plt.scatter(selected_points[:,0], selected_points[:,1])
    plt.show()

    # print(clusters)

    # plt.triplot(points[:,0], points[:,1], tri)
    plt.figure()
    plt.plot(points[:,0], points[:,1], 'o')
    plt.show()
