import torch
import torch.nn as nn
from .MLPs import *
from .util import *
from .base import BasicModel
from torch_scatter import scatter_max, scatter_min, scatter_mean, scatter_add

pool_ops = {'sum': scatter_add, 'max': scatter_max, 'mean': scatter_mean, 'min': scatter_min}

''' General GNN for any graph
'''
class GNN_E(BasicModel):
    def __init__(self, args):
        super(GNN_E, self).__init__(args, 'GNN_E')
        
        self.device = args.device
        
        self.n_iter = args.n_iter
        self.mlp_layer = args.mlp_layer
        self.hidden_dim = args.hidden_dim
        self.fc_output_layer = args.fc_output_layer
        self.graph_pooling_type = args.graph_pooling_type
        self.neighbor_pooling_type = args.neighbor_pooling_type
        
        self.node_feature_size = args.node_feature_size
        self.answer_size = calc_output_size(args)
        self.edge_feature_size = args.edge_feature_size
        self.add_self_loop = args.add_self_loop
        self.weight = args.weight
        
        self.MLP0 = torch.nn.ModuleList()
     
        for layer in range(self.n_iter):
            if layer == 0:
                self.MLP0.append(MLP(self.mlp_layer, self.node_feature_size*2 + self.edge_feature_size, self.hidden_dim, self.hidden_dim))
            else:
                self.MLP0.append(MLP(self.mlp_layer, self.hidden_dim*2 + self.edge_feature_size, self.hidden_dim, self.hidden_dim))
        
        self.MLP0.append(MLP(self.mlp_layer, self.hidden_dim, self.hidden_dim, self.hidden_dim))
        
        if args.loss_fn == 'cls':
            self.fcout = FCOutputModel(self.fc_output_layer, self.hidden_dim, self.hidden_dim, self.answer_size)
        else: 
            self.fcout = RegFCOutputModel(self.fc_output_layer, self.hidden_dim, self.hidden_dim, self.answer_size)
        
        if args.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.decay)
        else:
            self.optimizer = optim.SGD(self.parameters(), lr=args.lr, weight_decay=args.decay)
        
    def preprocess_neighbors_list(self, batch_graph):
        padded_neighbor_list = []
        padded_self_list = []
        graph_level_list = []
        edges_weights = []
        start_idx = [0]

        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.neighbors))
            graph_level_list.extend([i]*(start_idx[i+1]-start_idx[i]))
            
            padded_neighbors = []
            padded_self = []
            for j in range(len(graph.neighbors)):
                #add off-set values to the neighbor indices
                pad = [n + start_idx[i] for n in graph.neighbors[j]]
                
                #add self-loop
                if self.add_self_loop:
                    pad.append(j + start_idx[i])
                    padded_self.extend([j + start_idx[i]]*(len(graph.neighbors[j])+1))
                else:
                    padded_self.extend([j + start_idx[i]]*len(graph.neighbors[j]))
                    
                padded_neighbors.extend(pad)
                
                #add edge weights for edges from j to j's neighbors
                if self.weight == 'weight':
                    edges = [graph.g[j][n][self.weight] for n in graph.neighbors[j]]
                else:
                    edges = [0 for n in graph.neighbors[j]]
                    
                if self.add_self_loop:
                    edges.append(0)
                    
                edges_weights.extend(edges)
            padded_neighbor_list.extend(padded_neighbors)
            padded_self_list.extend(padded_self)
        
        return torch.LongTensor(padded_neighbor_list).to(self.device), torch.LongTensor(padded_self_list).to(self.device), torch.LongTensor(graph_level_list).to(self.device), torch.FloatTensor(edges_weights).to(self.device)
    
    ''' One iteration/layer of GNNs, call n_layer times in forward
    '''
    def reason_step(self, h, layer, padded_neighbor_list, padded_self_list, edges_weights):
        #print("reasoning layer", layer, padded_neighbor_list.shape)
        x_i = h[padded_neighbor_list]
        x_j = h[padded_self_list]
        
        # x_pair denotes the edges from v_j to v_i
        x_pair = torch.cat((x_i, x_j, edges_weights), dim=-1)

        relations = self.MLP0[layer](x_pair)

        # when we aggregate on vertex u, we aggregate all edges whose ending point is u (incoming edges to u)
        if self.neighbor_pooling_type in ['max', 'min']:
            x, _ = pool_ops[self.neighbor_pooling_type](relations, padded_neighbor_list, dim=0)
        else:
            x = pool_ops[self.neighbor_pooling_type](relations, padded_neighbor_list, dim=0)
        return x
    
    def forward(self, batch_graph):
        x = torch.cat([graph.node_features for graph in batch_graph], 0).to(self.device)
        padded_neighbor_list, padded_self_list, graph_level_list, edges_weights = self.preprocess_neighbors_list(batch_graph)
        edges_weights = edges_weights.unsqueeze(1)
        for layer in range(self.n_iter):
            x = self.reason_step(x, layer, padded_neighbor_list, padded_self_list, edges_weights)
      
        if self.graph_pooling_type in ['max', 'min']:
            x, _ = pool_ops[self.graph_pooling_type](x, graph_level_list, dim=0)
        else:
            x = pool_ops[self.graph_pooling_type](x, graph_level_list, dim=0)
        x = self.fcout(x)
        return x
