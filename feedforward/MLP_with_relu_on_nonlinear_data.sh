#!/bin/bash 

# quadratic function
python main.py --activation=relu --loss_fn=reg --decay=1e-5 --lr=0.0001 --activation=relu --mlp_layer=2 --hidden_dim=128 --batch_size=64 --epochs=500  --optimizer=Adam --data=./data/non-linear/quadratic/quadratic_xdim8_item1_trainssphere_testsball_signno_fix1_testr5.0_trainr1.0_valr1.0_ntrain20000_nval1000_ntest20000_Ar1.0_br0.0.pickle


# cosine function
python main.py --activation=relu --loss_fn=reg --decay=1e-5 --lr=0.0001 --activation=relu --mlp_layer=2 --hidden_dim=128 --batch_size=64 --epochs=500  --optimizer=Adam --data=./data/non-linear/cos/cos_xdim2_item1_trainscube_testscube_signno_fix1_testr2.0_trainr1.0_valr1.0_ntrain20000_nval1000_ntest80000_Ar1.0_br0.0.pickle

# square root function
python main.py --activation=relu --loss_fn=reg --decay=1e-5 --lr=0.0001 --activation=relu --mlp_layer=2 --hidden_dim=128 --batch_size=64 --epochs=500  --optimizer=Adam --data=./data/non-linear/sqrt/sqrt_xdim8_item1_trainscube_testscube_signno_fix1_testr10.0_trainr2.0_valr2.0_ntrain2000_nval1000_ntest20000_Ar1.0_br0.0.pickle

# l1 norm function, sensitive to hyper-parameters
## The following hyper-parameters lead to a small MAPE on test data
python main.py --activation=relu --loss_fn=reg --decay=1e-5 --lr=0.001 --activation=relu --mlp_layer=2 --hidden_dim=256 --batch_size=64 --epochs=500  --optimizer=Adam --data=./data/non-linear/l1/l1_xdim8_item1_trainssphere_testsball_signno_fix1_testr10.0_trainr0.5_valr0.5_ntrain20000_nval1000_ntest20000_Ar1.0_br0.0.pickle

## The following hyper-parameters lead to a large MAPE on test data
python main.py --activation=relu --loss_fn=reg --decay=1e-5 --lr=0.0001 --activation=relu --mlp_layer=2 --hidden_dim=128 --batch_size=64 --epochs=500  --optimizer=Adam --data=./data/non-linear/l1/l1_xdim8_item1_trainssphere_testsball_signno_fix1_testr10.0_trainr0.5_valr0.5_ntrain20000_nval1000_ntest20000_Ar1.0_br0.0.pickle