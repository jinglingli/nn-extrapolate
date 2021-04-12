import argparse
import os
import math
import numpy as np
import NTK
from sklearn.kernel_ridge import KernelRidge
import torch
import pickle
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

np.set_printoptions(threshold=20)

parser = argparse.ArgumentParser()
parser.add_argument('--data', default = "data", type = str, help = "data file")
parser.add_argument('--files_dir', type=str, default='results', help='the directory to store trained models logs')
parser.add_argument('--filename', type=str, default=None, help='the file which store trained model logs')
parser.add_argument('--loss_fn', type=str, choices=['mse', 'mape'], default='mse', help='various regression loss fucntions')
parser.add_argument('--alpha', type=float, default=0, help='alpha')

args = parser.parse_args()

def load_data(data_file):
    with open("%s" %data_file, 'rb') as f:
        train, val, test = pickle.load(f)

    X, y = [], []
    for data in [train, val, test]:
        X.extend(data[0])
        y.extend(data[1])

    train_fold = list(range(0, len(train[0])))
    val_fold = list(range(len(train[0]), len(train[0]) + len(val[0])))
    test_fold = list(range(len(val[0]) + len(train[0]), len(train[0]) + len(val[0]) + len(test[0])))
    X = np.asarray(X)
    y = np.asarray(y)
    
    return X, y, train_fold, val_fold, test_fold
    
def ridge_regression(K1, K2, y1, y2, alpha):
    n_val, n_train = K2.shape
    clf = KernelRidge(kernel = "precomputed", alpha = 0.0)
    clf.fit(K1, y1)
    z = clf.predict(K2)
    loss = (np.square(z - y2)).mean(axis=ax)
    return loss

# kernel ridge regression
def process(args, K, X, y, train_fold, val_fold):
    K1 = K[train_fold][:, train_fold]
    K2 = K[val_fold][:, train_fold] 
    y1 = y[train_fold]
    y2 = y[val_fold]
    
    n_val, n_train = K2.shape
    clf = KernelRidge(kernel = "precomputed", alpha = args.alpha)
    clf.fit(K1, y1)
    z = clf.predict(K2)
    loss = (np.square(z - y2)).mean()

    return loss, z

def mape(y_pred, y):
    e = torch.abs(y.view_as(y_pred) - y_pred) / torch.abs(y.view_as(y_pred))
    return 100.0 * torch.median(e)

mse_criterion = torch.nn.MSELoss()
lossfun = {'mse': mse_criterion, 'mape': mape}

if not args.filename:
    args.filename = args.data.split('/')[-1]
    args.filename = args.filename + "_alpha" + str(args.alpha)
    print(args.filename)

if not os.path.exists(args.files_dir):
    os.makedirs(args.files_dir)    
outf = open('%s/%s.log' %(args.files_dir, args.filename), "w")
print(args, file = outf)

X, y, train_fold, val_fold, test_fold = load_data(args.data)
print('calculating kernel...')
Ks = NTK.kernel_value_batch(X, d_max=4)
K = NTK.kernel_paper(X)

print('done calculating kernel')

train_loss, y_pred_train = process(args, K, X, y, train_fold, train_fold) 
val_loss, y_pred_val = process(args, K, X, y, train_fold, val_fold) 
test_loss, y_pred_test = process(args, K, X, y, train_fold, test_fold)    

print('train loss: %f, val loss: %f, test_loss: %f' %(train_loss, val_loss, test_loss))    
print('train loss: %f, val loss: %f, test_loss: %f' %(train_loss, val_loss, test_loss), file = outf)  
outf.close()
