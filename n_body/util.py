import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import networkx as nx
import random

class S2VGraph(object):
    def __init__(self, node_features, neighbors, g=None):
        '''
            neighbors: list of neighbors (without self-loop)
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            g: original networkX graph
        '''
        self.neighbors = neighbors
        self.node_features = node_features
        self.g = g

def mape(pred, label):
    diff = torch.abs(pred - label)
    e = diff.norm(dim=2) / label.norm(dim=2)
    return 100.0 * e.mean() 

