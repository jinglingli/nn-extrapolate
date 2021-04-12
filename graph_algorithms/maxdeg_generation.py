import networkx as nx
import numpy as np
import numpy.random as nprnd
from random import sample
import os
import math
import pickle
import argparse
import torch
from util import *
import torch.nn.functional as F
from pathlib import Path


VAL_RATIO, TEST_RATIO = 0.2, 0.5

def add_self_loops(graph):
    for n in graph.nodes:
        graph.add_edge(n, n, weight=0)

def max_node_degree(graph):
    deg_list = [deg for i, deg in graph.degree]
    return max(deg_list)

def add_edge_weights(graph, low, high):
    edge_weights = nprnd.randint(low, high+1, len(graph.edges))
    edge_dict = {}
    count = 0
    for edge in graph.edges:
        edge_dict[edge] = edge_weights[count]
        count += 1
    nx.set_edge_attributes(graph, edge_dict, 'weight')   
    
def generate_random_trees(n):
    return nx.random_tree(n)
    
def generate_random_graphs(n, p):    
    while True:  
        graph = nx.random_graphs.erdos_renyi_graph(n, p)
        if nx.is_connected(graph):
            break 
    return graph

def generate_complete_graphs(n):
    return nx.complete_graph(n)

def sparse_connected_graph(args, n, p, num_components):
    graphs = []
    for i in range(num_components):
        graphs.append(generate_random_graphs(n, p))

    current_graph = graphs[0]
    for i in range(1, num_components):
        current_graph = nx.disjoint_union(current_graph, graphs[i])
        node1 = nprnd.randint((i-1)*n, i*n)
        node2 = nprnd.randint(i*n, (i+1)*n)
        current_graph.add_edge(node1, node2)   

    return current_graph


def load_data(data):
    s2vs = []
    for g, ans in data:
        neighbors = []
        node_features = []
        for i in sorted(list(g.nodes())):
            neighbors.append(list(g.neighbors(i)))
            node_features.append(g.nodes[i]['node_features'])
        node_features = np.array(node_features)
        node_features = torch.from_numpy(node_features).float()
        s2vg = S2VGraph(ans, node_features, neighbors, g)
        s2vs.append((s2vg, ans))
    return s2vs


def generate_graphs_various_nodes(args):    
    if args.min_n == args.max_n:
        n = args.min_n
    else:
        n = nprnd.randint(args.min_n, args.max_n)
    if args.graph_type == 'random_graph':
        graph = generate_random_graphs(n, args.p)
    elif args.graph_type == 'tree':
        graph = generate_random_trees(n)
    elif args.graph_type == 'complete':
        graph = generate_complete_graphs(n)
    elif args.graph_type == 'path':
        graph = nx.path_graph(n)
    elif args.graph_type == 'ladder':
        graph = nx.ladder_graph(n)
    elif args.graph_type == 'tree':
        graph = nx.random_tree(n)
    elif args.graph_type == 'cycle':
        graph = nx.cycle_graph(n)
    elif args.graph_type == 'star':
        graph = nx.star_graph(n)
    elif args.graph_type == '4regular':
        graph = nx.random_regular_graph(4, n)
    else:
        print("Invalid graph type.")

    return graph     

'''
Generate graphs based on parameters.
'''
def make_graph(args, num_graphs, min_n, max_n, graph_type):
    graphs = []

    if graph_type == 'general':
        num_each = int(num_graphs/9)
        args.min_n, args.max_n, args.graph_type = min_n, max_n, 'random_graph'
        for p in np.linspace(0.1, 0.9, 9):
            args.p = p
            for i in range(num_each):
                graph = generate_graphs_various_nodes(args)
                ans = max_node_degree(graph)
                graphs.append((graph, ans))
    elif graph_type == 'expander':
        args.min_n, args.max_n, args.graph_type = min_n, max_n, 'random_graph'
        args.p = 0.8
        for i in range(num_graphs):
            graph = generate_graphs_various_nodes(args)
            ans = max_node_degree(graph)
            graphs.append((graph, ans))
    elif graph_type == 'complete' or graph_type == 'path' or graph_type == 'ladder' or graph_type == 'tree':
        args.min_n, args.max_n, args.graph_type = min_n, max_n, graph_type
        for i in range(num_graphs):
            graph = generate_graphs_various_nodes(args)
            ans = max_node_degree(graph)
            graphs.append((graph, ans))
    elif graph_type == 'cycle' or graph_type == 'star' or graph_type == '4regular':
        args.min_n, args.max_n, args.graph_type = min_n, max_n, graph_type
        for i in range(num_graphs):
            graph = generate_graphs_various_nodes(args)
            ans = max_node_degree(graph)
            graphs.append((graph, ans))

    else:
        print("Invalid graph type!")
        exit()

    return graphs

'''
Generate task (G, y)
'''
def generate_max_degree(args, num_graphs, min_n, max_n, num_colors, graph_type):
    graphs = make_graph(args, num_graphs, min_n, max_n, graph_type)
    
    for graph, _ in graphs:
        graph_nodes = len(graph.nodes)
        # node * feature
        if args.sampling == 'identical':
            colors_ind = nprnd.randint(1, num_colors+1, (graph_nodes, args.node_dim))
        else:
            colors_ind = nprnd.uniform(-num_colors, num_colors, (graph_nodes, args.node_dim))
        node_dict = {}
        ind = 0
        for node in graph.nodes:
            node_dict[node] = colors_ind[ind]
            ind += 1
        nx.set_node_attributes(graph, node_dict, 'node_features')
    return graphs


def main():
    # parameters for graph_generation
    parser = argparse.ArgumentParser(description='Graph generation')
    parser.add_argument('--graph_type', type=str, default='random_graph', help='select which graph type to generate')
    parser.add_argument('--train_min_n', default=20, type=int, help='min number of nodes in the graph')
    parser.add_argument('--train_max_n', default=30, type=int, help='max number of nodes in the graph')
    parser.add_argument('--test_min_n', default=50, type=int, help='min number of nodes in the graph')
    parser.add_argument('--test_max_n', default=100, type=int, help='min number of nodes in the graph')
    parser.add_argument('--train_p', default=0.9, type=float, help='probability of edges in erdos_renyi_graph')
    parser.add_argument('--test_p', default=0.1, type=float, help='probability of edges in erdos_renyi_graph')
    parser.add_argument('--train_color', default=5, type=int, help='number of colors')
    parser.add_argument('--test_color', default=10, type=int, help='number of colors')
    parser.add_argument('--node_dim', default=3, type=int, help='number of node features')
    parser.add_argument('--train_graph', default='4regular', type=str, help='train graph type')
    parser.add_argument('--test_graph', default='general', type=str, help='test graph type')
    parser.add_argument('--folder', default='data', type=str, help='folder to store the generated data')
    parser.add_argument('--sampling', default='uniform', choices=['uniform', 'identical'], type=str, help='sampling rules for the node features')

    parser.add_argument('--min_edge_weight', default=1, type=int, help='min edge weight in the graph')
    parser.add_argument('--max_edge_weight', default=30, type=int, help='max edge weight in the graph')
    parser.add_argument('--max_hop', default=14, type=int, help='max number of hops expected in the shortest path')
    parser.add_argument('--num_levels', default=10, type=int, help='num of possible levels for each node')
    parser.add_argument('--num_graphs', default=5000, type=int, help='num of graphs we want in the train dataset')
    parser.add_argument('--data', type=str, help='data filename')
    parser.add_argument('--random', default=0, type=int, help='random seed')
    args = parser.parse_args()

    random_seed = args.random
    np.random.seed(random_seed)
    random.seed(random_seed)

    Path(args.folder).mkdir(parents=True, exist_ok=True)
    output = './%s/%s.pickle' %(args.folder, args.data)
    
    if not os.path.exists(output):
        train = generate_max_degree(args, args.num_graphs, args.train_min_n, args.train_max_n, args.train_color, args.train_graph)
        train = load_data(train)
        val = generate_max_degree(args, max(int(args.num_graphs*VAL_RATIO),1), args.train_min_n, args.train_max_n, args.train_color, args.train_graph)
        val = load_data(val)
        test = generate_max_degree(args, max(int(args.num_graphs*TEST_RATIO),1),  args.test_min_n, args.test_max_n, args.test_color, args.test_graph)
        test = load_data(test)

        with open(output, 'wb') as f:
            pickle.dump((train, val, test), f)

        print("data file saved to %s" % output)

if __name__ == '__main__':
    main()
