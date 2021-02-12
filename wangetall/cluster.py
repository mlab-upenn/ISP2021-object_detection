import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from itertools import permutations
from collections import defaultdict
import random

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
        graph = []
        all_edges = set([tuple(edge) for item in simplices for edge in permutations(item,2)])

        for edge in all_edges:
            dist = np.linalg.norm(points[edge[0]]-points[edge[1]])
            edgelist = list(edge)
            edgelist.append(dist)
            graph.append(edgelist)


        
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

        tree = []
        sorted_graph = sorted(graph, key=lambda i:i[2])
        edge_count = 0
        i = 0

        parent = np.arange(len(points))
        rank = np.zeros((parent.shape))

        while edge_count < len(points)-1:
            origin, dest, weight = sorted_graph[i]
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
        k = 500
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
            components_dict[self.elts[i,2]].append(i)
        
        return components_dict




if __name__ == "__main__":
    points= np.array(random.sample(range(200), 200)).reshape((100,2))
    cl = Cluster()
    clusters = cl.cluster(points)
    print(clusters)
    # cl = Cluster()
    # tri = cl.compute_delauney(points)
    # graph = cl.label_edge(tri, points)
    # tree = cl.min_spanning_tree(graph, points)

    # clusters = cl.EGBIS(tree, points)
    # print(clusters)

    # plt.triplot(points[:,0], points[:,1], tri)
    # plt.plot(points[:,0], points[:,1], 'o')
    # plt.show()

