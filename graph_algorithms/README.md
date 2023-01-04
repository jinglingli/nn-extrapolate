# Experiments on architectures that help extrapolation
We validate that **linear algorithmic alignment** helps extrapolation on two Dynamic Programming (DP) tasks: 
- Max Degree 
- Shortest path 


## Reproducing results with one command
We include a script file [`reproduce.sh`](./reproduce.sh) to repoduce the results on extrapolation settings for the graph algorithm experiments in the paper. To be more specific,
```
bash ./reproduce.sh
```
reproduces the results (pink bars) in Figure 6(a). Continue reading below for more details.

# Maximum Degree

### Data Generation

Run the following script to generate datasets with identical node features:

```
python sample_scripts/sample_maxdeg_identical.py
```

Run the following script to generate datasets with randomly sampled node features:

```
python sample_scripts/sample_maxdeg_uniform.py
```

The names of the data files are of the form:
```
maxdeg_{node_feature_type}_{graph_type}_Ndim{node_dimension}_Train_V{min_num_of_vertices_in_train}_{max_num_of_vertices_in_train}_C{train_node_feature_range}_Test_V{min_num_of_vertices_in_test}_{max_num_of_vertices_in_test}_C{test_node_feature_range}.pickle`
```

For example, the data file
```
maxdeg_uniform_general_Ndim3_Train_V20_30_C5_Test_V50_100_C10.pickle 
```
denotes the dataset with general graphs (Erdős–Rényi random graphs with various edge probability) as the training data, where the number of vertices is sampled uniformly from [20, 30], and each node in the training graph has a three-dimensional real vector as its node feature where each dimension is sampled from [−5.0, 5.0]. Accordingly, the test data consists random graphs where the number of vertices is sampled uniformly from [50, 100], and each node in the training graph has a three-dimensional real vector as its node feature where each dimension is sampled uniformly from [−10.0, 10.0] (a larger range than the ones in the training data). 


### Training
A sample command to train a 2-layer GNN with `max` graph pooling and `max` neighbor pooling:

```
CUDA_VISBILE_DEVICES=0 python main.py --model=GNN --n_iter=2 --weight=None --lr=0.01 --fc_output_layer=1 --mlp_layer=2 --hidden_dim=256 --batch_size=64  --graph_pooling_type=max --neighbor_pooling_type=sum --epochs=250 --data=maxdeg_uniform_general_Ndim3_Train_V20_30_C5_Test_V50_100_C10.pickle
``` 

Note that:
- In max degree, there is no edge features, so the model should be set to `GNN`, and the  flag `--weight` should be turn off (i.e., `--weight=None`).
- Run 300 epochs for graphs with uniform node features, and 100 epochs for graphs with identical node features.

## Shortest Path

### Data Generation
Run the following script to generate datasets.

```
python sample_scripts/sample_shortest_uniform.py
```

The names of the data files are of the form:
```
shortestpath_uniform_{graph_type}_Ndim{node_dimension}_maxhop{max_length_of_the_shortest_path}_Train_V{min_num_of_vertices_in_train}_{max_num_of_vertices_in_train}_C{train_node_feature_range}_E{max_edge_weight_in_train}_Test_V{min_num_of_vertices_in_test}_{max_num_of_vertices_in_test}_C{test_node_feature_range}_E{max_edge_weight_in_test}
```

For example, the file
```
shortestpath_uniform_general_Ndim1_maxhop3_Train_V20_40_C5_E5_Test_V50_70_C5_E10.pickle
```
denotes the dataset with general graphs (Erdős–Rényi random graphs with various edge probability) as the training data, where the number of vertices is sampled uniformly from [20, 40], and the edge weights are uniformly drawn from [1.0, 5.0]. Accordingly, the test data consists random graphs where the number of vertices is sampled uniformly from [50, 70], and the edge weights are uniformly drawn from [1.0, 10.0].
For all graphs in the dataset, each node feature contains a scalar sampled uniformaly from [−5.0, 5.0] along with two binary indicators, which correspondingly denotes whether the node is a starting node or not, and whether the node is a terminal node or not.

### Training
A sample command to train a 3-layer GNN with `min` graph pooling and `min` neighbor pooling:
```
CUDA_VISBILE_DEVICES=0 python main.py --model=GNN_E --n_iter=3 --weight=weight --lr=0.005 --fc_output_layer=1 --mlp_layer=2 --hidden_dim=256 --batch_size=32 --graph_pooling_type=min --neighbor_pooling_type=min --epochs=250 --data=shortestpath_uniform_general_Ndim1_maxhop3_Train_V20_40_C5_E5_Test_V50_70_C5_E10.pickle
``` 

Note that:
- There are edge features,  so the model should be set to `GNN_E`, and the flag `--weight` should be turned on (i.e., `--weight=weight`).

