# How Neural Networks Extrapolate 

This repository is the PyTorch implementation of the experiments in the following paper: 

Keyulu Xu, Mozhi Zhang, Jingling Li, Simon S. Du, Ken-ichi Kawarabayashi, Stefanie Jegelka. How Neural Networks Extrapolate: From Feedforward to Graph Neural Networks. ICLR 2021. 

[arXiv](https://arxiv.org/abs/2009.11848) [OpenReview](https://openreview.net/forum?id=UH-cmocLJC) 

If you make use of the relevant code/experiment/idea in your work, please cite our paper (Bibtex below).
```
@inproceedings{
xu2021how,
title={How Neural Networks Extrapolate: From Feedforward to Graph Neural Networks},
author={Keyulu Xu and Mozhi Zhang and Jingling Li and Simon Shaolei Du and Ken-Ichi Kawarabayashi and Stefanie Jegelka},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=UH-cmocLJC}
}
```


## Requirements
- This codebase has been tested for `python3.7` and `pytorch 1.4.0` (with `CUDA VERSION 10.0`).
- To install necessary python packages, run `pip install -r requirements.txt` (This installs pytorch).
- The packages [networkx](https://networkx.org/documentation/stable/install.html) and [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) need to be installed separately. networkx and geometric versions can be decided based on pytorch and CUDA version.

## Instructions
Refer to each folder for instructions to reproduce the experiments. All experiments can be easily reproduced by typing the commands provided.
- Experiments related to feedforward networks may be found in the [`feedforward`](./feedforward) folder.
- Experiments on **architectures** that help extrapolation may be found in the [`graph_algorithms`](./graph_algorithms) folder.
- Experiments on **representations** that help extrapolation may be found in the [`n_body`](./n_body) folder.
