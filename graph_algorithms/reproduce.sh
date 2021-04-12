#!/bin/bash 

# max degree interpolate, training on general graph
python main.py --model=GNN --n_iter=2 --weight=None --lr=0.01 --fc_output_layer=1 --mlp_layer=2 --hidden_dim=256 --batch_size=64  --graph_pooling_type=max --neighbor_pooling_type=sum --epochs=300 --data=maxdeg_uniform_general_Ndim3_Train_V20_30_C5_Test_V50_100_C5.pickle

# max degree extrapolate, training on general graph
python main.py --model=GNN --n_iter=2 --weight=None --lr=0.01 --fc_output_layer=1 --mlp_layer=2 --hidden_dim=256 --batch_size=64  --graph_pooling_type=max --neighbor_pooling_type=sum --epochs=300 --data=maxdeg_uniform_general_Ndim3_Train_V20_30_C5_Test_V50_100_C10.pickle

# shortest path interpolate, training on general graph
python main.py --model=GNN_E --n_iter=3 --weight=weight --lr=0.005 --fc_output_layer=1 --mlp_layer=2 --hidden_dim=256 --batch_size=32 --graph_pooling_type=min --neighbor_pooling_type=min --epochs=300 --data=shortestpath_uniform_general_Ndim1_maxhop3_Train_V20_40_C5_E5_Test_V50_70_C5_E5.pickle

# shortest path extrapolate, training on general graph
python main.py --model=GNN_E --n_iter=3 --weight=weight --lr=0.005 --fc_output_layer=1 --mlp_layer=2 --hidden_dim=256 --batch_size=32 --graph_pooling_type=min --neighbor_pooling_type=min --epochs=300 --data=shortestpath_uniform_general_Ndim1_maxhop3_Train_V20_40_C5_E5_Test_V50_70_C5_E10.pickle