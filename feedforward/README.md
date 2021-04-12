# How Feedforward Neural Networks Extrapolate

We provide instructions to reproduce the following categories of experiments on feedforward networks:
- Learning simple nonlinear tasks
- Computation of R^2 of the learned functions (Theorem 1).
- Learning linear tasks with different data geometry (Theorem 2)
- Exact computation with neural tangent kernel (Lemma 1).


## Data generation
This section shows general rules for generating the data files via [`data_generation.py`](./data_generation.py). For each task, we will still provide specific instructions to generate the required datasets.

The folder [`sample_scripts`](./sample_scripts) contains scripts to generate the datasets. Generated data files will be under the `data` folder by default. Data files are named with the following rule:

```
{target function type}_xdim{input dimension}_item1_trains{shape of training data}_tests{shape of test data}_signno_fix1_testr{test data range}_trainr{training data range}_valr{validation data range}_ntrain{# of training data}_nval{# of validation data}_ntest{# of test data}_Ar1.0_br0.0.pickle
```
For example, the data file

```
quadratic_xdim8_item1_trainssphere_testsball_signno_fix1_testr5.0_trainr1.0_valr1.0_ntrain20000_nval1000_ntest20000_Ar1.0_br0.0.pickle
```
denotes the dataset with quadratic target function and input dimension = 8. There are 20000 samples in the training set, each uniformaly sampled from an eight-dimensional hyper-sphere with radius 1.0. Similar intepretations for val and test sets.


##  Learning simple nonlinear tasks
This section shows training feedforward networks on simple tasks and computing the extrapolation error in MAPE.

First type the following commands to generate datasets. All generated data files will be under the `data/non-linear` folder by default. 

```
python sample_scripts/sample_quadratic
python sample_scripts/sample_cos.py
python sample_scripts/sample_sqrt.py
python sample_scripts/sample_l1.py
```

The following command provides an example to train a ReLU MLP on a generated task:

```
CUDA_VISBILE_DEVICES=0 python main.py --activation=relu --loss_fn=reg --decay=1e-5 --lr=0.0001 --activation=relu --mlp_layer=2 --hidden_dim=128 --batch_size=64 --epochs=500  --optimizer=Adam --data=./data/non-linear/quadratic/quadratic_xdim8_item1_trainssphere_testsball_signno_fix1_testr5.0_trainr1.0_valr1.0_ntrain20000_nval1000_ntest20000_Ar1.0_br0.0.pickle
```

The training logs will be under the `results` folder by default. Script [`MLP_with_relu_on_nonlinear_data.sh`](./MLP_with_relu_on_nonlinear_data.sh) shows more examples.


## Computation of R^2
This instruction shows how to reproduce the 0.99 R^2 for learned functions in the OOD domain, which supports the non-asymptotic linear extrapolation (Theorem 1). 


First type the following commands to generate datasets.

```
python sample_scripts/sample_quadratic.py
python sample_scripts/sample_cos.py
python sample_scripts/sample_sqrt.py
python sample_scripts/sample_l1.py
```

To compute the R^2 of NN’s learned functions along randomly sampled directions in out-of-distribution domain, we first need a "saved model" and supply the "path to the saved model" along with its training log to [`sphere_rsquare.py`](./sphere_rsquare.py). The following command provides an example of first training a MLP and save the best model (based on validation error) with `--save_model` flag:

```
CUDA_VISBILE_DEVICES=0 python main.py --activation=relu --loss_fn=reg --decay=1e-5 --lr=0.0001 --activation=relu --mlp_layer=2 --hidden_dim=128 --batch_size=64 --epochs=500  --optimizer=Adam --data=./data/non-linear/quadratic/quadratic_xdim8_item1_trainssphere_testsball_signno_fix1_testr5.0_trainr1.0_valr1.0_ntrain20000_nval1000_ntest20000_Ar1.0_br0.0.pickle --save_model
```

The following command computes the R^2 of the saved model in OOD domain:

```
python sphere_rsquare.py --path=./models_dir/linear_xdim16_item1_trainssphere_testsball_signno_fix1_testr200.0_trainr5.0_valr5.0_ntrain10000_nval1000_ntest2000_Ar20.0_br0.0.pickle/FeedForward_lr0.001_actrelu_mlp2_hdim256_idim16_odim1_bs64_optionNone_epoch500_seed2.log/model_best.pth.tar --log=./results/linear_xdim16_item1_trainssphere_testsball_signno_fix1_testr200.0_trainr5.0_valr5.0_ntrain10000_nval1000_ntest2000_Ar20.0_br0.0.pickle/FeedForward_lr0.001_actrelu_mlp2_hdim256_idim16_odim1_bs64_optionNone_epoch500_seed2.log
```


## Learning linear tasks with different data geometry
This section shows how the learned function interacts with data geometry (Theorem 2).

First type the following commands to generate datasets. All generated data files will be under the `data/systematic_linear` (training data covers all directions) or `data/linear_miss_direction` (training data is restricted in some direction) folder by default.

```
python sample_scripts/sample_linear.py
python sample_scripts/sample_direction.py
```
The following command provides an example for training.

```
CUDA_VISBILE_DEVICES=0 python main.py --activation=relu --loss_fn=reg --decay=1e-5 --lr=0.001 --activation=relu --mlp_layer=2 --hidden_dim=256 --batch_size=64 --epochs=500  --optimizer=Adam --data=./data/systematic_linear/linear_xdim16_item1_trainssphere_testsball_signno_fix1_testr200.0_trainr5.0_valr5.0_ntrain10000_nval1000_ntest2000_Ar20.0_br0.0.pickle
```

The training logs will be put under the `results` folder. Refer to script [`MLP_with_relu_on_linear_data.sh`](./MLP_with_relu_on_linear_data.sh) for more options.



## Exact computation with neural tangent kernel
This section validates Lemma 1: provable extrapolation of linear functions with 2d data. We provide implementation for the exact computation of infinitely wide neural networks, i.e.,  neural tangent kernel. In this set of experiments, the training data contains an orthogonal basis and their opposite vectors. A two-layer neural tangent kernel (NTK) achieve zero test error up to machine precision.

First type the following command to generate the training data.

```
python data_generation.py --folder='data/ntk_linear' --data='linear' --x_dim=32 --train_shape='basis' --test_shape='ball' --sign=no  --fix=1 --test_r=20.0 --train_r=5.0 --val_r=5.0 --n_train=1000 --n_val=1000 --n_test=2000  --A_r=20.0 --b_r=0.0
```

Compute the NTK's performance on the generated dataset with the following command. The code for NTK is adapted from Arora et al 2020.

```
python NTK_main.py --data=data/ntk_linear/linear_xdim32_item1_trainsbasis_testsball_signno_fix1_testr20.0_trainr5.0_valr5.0_ntrain1000_nval1000_ntest2000_Ar20.0_br0.0.pickle
```
