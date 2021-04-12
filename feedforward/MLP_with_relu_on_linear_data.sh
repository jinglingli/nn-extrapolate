#!/bin/bash 

# linear function in dimension 32 with training, valdiation, and test data uniformly sampled from a hyper-cube
python main.py --activation=relu --loss_fn=reg --decay=1e-5 --lr=0.001 --activation=relu --mlp_layer=2 --hidden_dim=256 --batch_size=64 --epochs=500  --optimizer=Adam --data=./data/systematic_linear/linear_xdim32_item1_trainscube_testscube_signno_fix1_testr100.0_trainr5.0_valr5.0_ntrain10000_nval1000_ntest2000_Ar20.0_br0.0.pickle

# linear function in dimension 16 with training and valdiation uniformly sampled from a sphere and test data uniformly sampled from a hyper-ball
python main.py --activation=relu --loss_fn=reg --decay=1e-5 --lr=0.001 --activation=relu --mlp_layer=2 --hidden_dim=256 --batch_size=64 --epochs=500  --optimizer=Adam --data=./data/systematic_linear/linear_xdim16_item1_trainssphere_testsball_signno_fix1_testr200.0_trainr5.0_valr5.0_ntrain10000_nval1000_ntest2000_Ar20.0_br0.0.pickle

# linear function in dimension 32 where the training data is only restricted to positive in all dimensions
python main.py --activation=relu --loss_fn=reg --decay=1e-5 --lr=0.001 --activation=relu --mlp_layer=2 --hidden_dim=256 --batch_size=64 --epochs=500  --optimizer=Adam --data=./data/linear_miss_direction/linear_xdim32_item1_trainscube_testscube_signp_fix32_testr20.0_trainr5.0_valr5.0_ntrain10000_nval1000_ntest2000_Ar10.0_br0.0.pickle

# linear function in dimension 32 where the training data is only restricted to positive in the first 16 dimensions
python main.py --activation=relu --loss_fn=reg --decay=1e-5 --lr=0.001 --activation=relu --mlp_layer=2 --hidden_dim=256 --batch_size=64 --epochs=500  --optimizer=Adam --data=./data/linear_miss_direction/linear_xdim32_item1_trainscube_testscube_signp_fix16_testr20.0_trainr5.0_valr5.0_ntrain10000_nval1000_ntest2000_Ar10.0_br0.0.pickle
