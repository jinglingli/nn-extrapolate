import networkx as nx
import numpy as np
import random
import torch

class S2VGraph(object):
    def __init__(self, label, node_features, neighbors, g=None):
        '''
            label: graph label
            neighbors: list of neighbors (without self-loop)
            node_features: a torch float tensor
            g: original networkX graph
        '''
        self.label = label
        self.neighbors = neighbors
        self.node_features = node_features
        self.g = g