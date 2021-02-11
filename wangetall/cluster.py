import numpy as np



class Cluster:
    def __init__(self):
        pass

    @classmethod
    def cluster(self, data):
        graph = self.EMST(data)
        clusters = self.EGBIS(graph)
        return clusters

    def EMST(self):
        #https://en.wikipedia.org/wiki/Euclidean_minimum_spanning_tree
        self.compute_delauney()
        self.label_edge()
        self.min_spanning_tree()

        return graph


    def compute_delauney(self):
        pass

    def label_edge(self):
        pass

    def min_spanning_tree(self):
        pass

    def EGBIS(self):
        #shitty implementation vv -- not sure if it should be trusted
        #https://github.com/devforfu/egbis/blob/master/egbis/segmentation.py
       #the paper vv
       #http://people.cs.uchicago.edu/~pff/papers/seg-ijcv.pdf
       
        return clusters

