import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import numpy.random as nprnd
from random import sample 
import math
import pickle
import argparse
import torch
import torch.nn.functional as F
import numpy.linalg as LA
import random

def square(x):
    return x ** 2

actfun = {'sin': np.sin, 'square': square, 'tanh': np.tanh, 'exp': torch.exp, 'log':torch.log, 'relu': F.relu, 'gelu': F.gelu, 'sigmoid': F.sigmoid}

def sample_cube(lower, upper, x_dim):
    return np.random.uniform(lower, upper, x_dim)

def sample_ball(radius, x_dim):
    u = np.random.normal(0, 1, x_dim)
    u = u / LA.norm(u)
    r = np.random.uniform(0.0, radius)
    x = u * r
    return x

def sample_sphere(radius, x_dim):
    u = np.random.normal(0, 1, x_dim)
    u = u / LA.norm(u)
    x = u * radius
    return x

def sample_A(args):
    if args.data=='linear' or args.data == 'mix':
        A = np.random.uniform(-args.A_r, args.A_r, (args.y_dim, args.x_dim))
    else:
        A = np.random.uniform(-args.A_r, args.A_r, (args.x_dim, args.x_dim))
    return A

def sample_WA(args):
    a = np.random.uniform(-args.A_r, args.A_r, (args.y_dim, args.x_dim))
    W = np.random.uniform(-args.A_r, args.A_r, (args.x_dim, args.x_dim))
    return W, a

def sample_AB(args):
    AB = np.random.uniform(-args.A_r, args.A_r, (4, args.n_item))
    return AB

def process_data(args, A, AB, b, W, a, data_range, n_data, noise, shape, sign, fix):
    data = []
    target = []
    data_cnt = 0
    idx = 0
    turn = 1
    prev_x = 0
    if args.data in ['sin', 'cos', 'zigzag'] or 'periodic' in args.data:
        data_range = data_range * np.pi
    if args.data == 'sqrt':
        sign = 'p'
        fix = args.x_dim

    while data_cnt < n_data:
        if shape == 'cube':
            if sign == 'z':
                x0 = [0.1]
                x1 = sample_cube(-data_range, data_range, args.x_dim-1)
                x = np.concatenate((x0, x1))
            elif sign == 'p':
                x0 = sample_cube(0.0, data_range, fix)
                x1 = sample_cube(-data_range, data_range, args.x_dim-fix)
                x = np.concatenate((x0, x1))
            elif sign == 'n':
                x0 = sample_cube(-data_range, 0.0, fix)
                x1 = sample_cube(-data_range, data_range, args.x_dim-fix)
                x = np.concatenate((x0, x1))
            else:
                x = sample_cube(-data_range, data_range, args.x_dim)
        elif shape == 'sphere':
            if sign == 'z':
                x0 = [0.1]
                x1 = sample_sphere(data_range, args.x_dim-1)
                x = np.concatenate((x0, x1))
            elif sign == 'p':
                x0 = sample_sphere(0.0, data_range, fix)
                x1 = sample_sphere(-data_range, data_range, args.x_dim-fix)
                x = np.concatenate((x0, x1))
            elif sign == 'n':
                x0 = sample_sphere(-data_range, 0.0, fix)
                x1 = sample_sphere(-data_range, data_range, args.x_dim-fix)
                x = np.concatenate((x0, x1))
            else:
                x = sample_sphere(data_range, args.x_dim)
        elif shape == 'basis':
            if turn == 1:
                x = np.zeros(args.x_dim)
                x[idx] = data_range
                prev_x = x
                x = np.matmul(args.Q, x)
            else:
                x = prev_x
                x = -x
                x = np.matmul(args.Q, x)
        elif shape == 'rd':
            if data_cnt > 2 * args.x_dim:
                x = sample_sphere(data_range, args.x_dim)
            elif turn == 1:
                x = np.zeros(args.x_dim)
                x[idx] = data_range
                prev_x = x
                x = np.matmul(args.Q, x)
            else:
                x = prev_x
                x = -x
                x = np.matmul(args.Q, x)
        else:
            if sign == 'z':
                x0 = [0.1]
                x1 = sample_ball(data_range, args.x_dim-1)
                x = np.concatenate((x0, x1))
            else:
                x = sample_ball(data_range, args.x_dim)

        data_cnt += 1
        turn = -turn
        if turn == 1:
            idx += 1
        if idx == args.x_dim:
            idx = 0

        if 'act' in args.data:
            x = np.matmul(W, x)
            
        if args.data=='linear':
            y = np.matmul(A, x)
            y = y+ b
        elif args.data in ['linear_plain']:
            y = [x.sum()]
        elif args.data=='quadratic':
            a = np.matmul(x.T, A)
            y = np.matmul(a, x)
            y = [y]
        elif args.data=='mix':
            mid = int(args.x_dim / 2)
            x_linear = x[:mid]
            x_nonlinear = x[mid:]
            x_nonlinear = x_nonlinear ** 2
            x_feature = np.concatenate((x_linear, x_nonlinear))
            y = np.matmul(A, x_feature)
        elif args.data=='cos':
            y = [np.cos(x * np.pi).sum()]
        elif args.data=='sin':
            y = [np.sin(x * np.pi).sum()]
        elif args.data=='l1':
            y = [LA.norm(x, ord=1)]
        elif args.data=='sqrt':
            y = [np.sqrt(x).sum()]
        elif args.data=='square':
            y = [np.sum(np.square(x))]
        elif args.data=='constant':
            y = [1]
        elif args.data=='tanh':
            y = [np.tanh(x).sum()]
        elif args.data=='sum':
            y = [x.sum()]
        elif args.data=='exp':
            y = [np.exp(x).sum()]
        elif args.data=='log':
            y = [np.log(x).sum()]
        elif args.data=='gelu':
            y = [F.gelu(torch.from_numpy(x)).sum().item()]
        elif args.data=='sigmoid':
            y = [torch.sigmoid(torch.from_numpy(x)).sum().item()]
        elif args.data=='zigzag':
            y = [np.arccos(np.cos(x)).sum()]
        elif 'periodic' in args.data:
            a1, a2, b1, b2 = AB[0,:], AB[1,:], AB[2,:], AB[3,:]
            total = 0
            for i in range(len(a1)):
                if 'sin' in args.data:
                    total = total + a1[i] * np.sin(a2[i] * x * np.pi) 
                elif 'cos' in args.data:
                    total = total + b1[i] * np.cos(b2[i] * x * np.pi)
                else:
                    total = total + a1[i] * np.sin(a2[i] * x * np.pi) + b1[i] * np.cos(b2[i] * x * np.pi)
            y = [total.sum()]
        elif 'act' in args.data:
            active = args.data[3:]
            y = [np.matmul(a, actfun[active](x))]
        if noise > 0:
            y += np.random.normal(0, args.noise, args.y_dim)
        data.append(x)
        target.append(y)
    return (data, target)

def sample_data(args):
    A = sample_A(args)
    AB = sample_AB(args)
    W, a = sample_WA(args)
    b = np.random.uniform(-args.b_r, args.b_r, args.y_dim)
    P = np.random.uniform(-1.0, 1.0, (args.x_dim, args.x_dim))
    Q, R = LA.qr(P)
    args.Q = Q
       
    train = process_data(args, A, AB, b, W, a, args.train_r, args.n_train, args.noise, args.train_shape, args.sign, args.fix)
    val =  process_data(args, A, AB, b, W, a, args.val_r, args.n_val, args.noise, args.train_shape, args.sign, args.fix)
    test = process_data(args, A, AB, b, W, a, args.test_r, args.n_test, args.noise, args.test_shape, 'no', 0)
    output = '%s/%s_xdim%s_item%s_trains%s_tests%s_sign%s_fix%s_testr%s_trainr%s_valr%s_ntrain%s_nval%s_ntest%s_Ar%s_br%s.pickle' %(args.folder, args.data, args.x_dim, args.n_item, args.train_shape, args.test_shape, args.sign, args.fix, args.test_r, args.train_r, args.val_r, args.n_train, args.n_val, args.n_test, args.A_r, args.b_r)
    return (train, val, test), output

def main():
    parser = argparse.ArgumentParser(description='Data Generation')
    parser.add_argument('--data', type=str, default="linear", help='data function')
    parser.add_argument('--x_dim', type=int, default=10, help='input x dim')
    parser.add_argument('--y_dim', type=int, default=1, help='output y dim')
    parser.add_argument('--n_item', type=int, default=1, help='items in periodic')
    parser.add_argument('--train_shape', type=str, default='cube', help='cube, sphere, ball')
    parser.add_argument('--test_shape', type=str, default='cube', help='cube, sphere, ball')

    parser.add_argument('--A_r', type=float, default=0.1, help='A linear func entry uniform in -A_r..A_r')
    parser.add_argument('--b_r', type=float, default=0.5, help='bias uniform in -b_r..b_r')
    parser.add_argument('--train_r', type=float, default=0.1, help='train sample range')
    parser.add_argument('--val_r', type=float, default=0.1, help='validation sample range')
    parser.add_argument('--test_r', type=float, default=2.0, help='test sample range')
    parser.add_argument('--n_train', type=int, default=5000, help='# training data samples')
    parser.add_argument('--n_val', type=int, default=1000, help='# validation data samples')
    parser.add_argument('--n_test', type=int, default=3000, help='# test data samples')
    parser.add_argument('--sign', type=str, default='no', help='[no, p, n, z] miss direction(sign) ')
    parser.add_argument('--fix', type=int, default=0, help='fix first dimensions sign')

    parser.add_argument('--folder', type=str, default='data', help='data/linear/...')

    parser.add_argument('--ex_dim', type=int, default=0, help='extrapolate dimension')
    parser.add_argument('--noise', type=float, default=0, help='Gaussian noise sigma')
    parser.add_argument('--random_seed', type=int, default=2, help='random seed')
    args = parser.parse_args()

    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # create folder if not exists
    Path(args.folder).mkdir(parents=True, exist_ok=True)

    # extrapolage all dimension by default
    if args.ex_dim == 0:
        args.ex_dim = args.x_dim

    # select a function to generate data 
    data, output = sample_data(args)

    with open(output, 'wb') as f:
        pickle.dump(data, f)

    print("data saved to %s" % output)
                              
if __name__ == '__main__':
    main() 
