"""Plot model's prediction for a 1-d model"""

from argparse import ArgumentParser

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import torch
import random

from MLPs import RegFCOutputModel
from main import load_data, cvt_data_axis, tensor_data
import numpy as np
from sklearn.linear_model import LinearRegression

check = {'val': 1, 'test': 2}
random.seed(1)
n_sample = 100
n_trials = 5000
extend_ratio, adjust = 10, 1.1 

random.seed(1)
def get_range(dataset_name):
    line = dataset_name.split('/')[-1]
    ks = {}
    for item in line.split('_'):        
        if 'trainr' in item:
            r = float(item.replace('trainr', ''))
            return r

def get_data_shape(dataset_name):
    line = dataset_name.split('/')[-1]
    
    for item in line.split('_'):
        if 'cube' in item:
            return 'cube'
    return 'sphere'

def get_data_settings(dataset_name):
    line = dataset_name.split("/")[-1]
    ks = {}
    for item in line.split('_'):
        if '.pickle' in item:
            item = item.strip('.pickle')
        if 'xdim' in item:
            ks['xdim'] = int(item.replace('xdim', ''))
        elif 'trainr' in item:
            ks['train_range'] = float(item.replace('trainr', ''))    
        elif 'testr' in item:
            ks['test_range'] = float(item.replace('testr', ''))
        elif 'valr' in item:
            ks['val_range'] = float(item.replace('valr', ''))
        elif 'ntrain' in item:
            ks['ntrain'] = int(item.replace('ntrain', ''))
        elif 'nval' in item:
            ks['nval'] = int(item.replace('nval', ''))
        elif 'fix' in item:
            ks['fix'] = int(item.replace('fix', ''))
        elif 'ntest' in item:
            ks['ntest'] = int(item.replace('ntest', ''))
        elif 'Ar' in item:
            ks['Ar'] = float(item.replace('Ar', ''))
        elif 'br' in item:
            ks['br'] = float(item.replace('br', ''))
        elif 'sign' in item:
            ks['sign'] = item.replace('sign', '')
    return ks  
            

def find_k(w, r, shape, adjust=adjust):
    if shape == 'cube':
        k = adjust * (r / w).abs().min() 
    elif shape == 'sphere':
        k = adjust * r 
    return k
    
def main():
    parser = ArgumentParser()
    parser.add_argument('--path', help='path to model')
    parser.add_argument('--log', help='path to the training log')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    args = parser.parse_args()

    with open(args.log, 'r') as f:
        args_dict = f.readlines()[0]
    params = eval(args_dict)

    # Parse path to get model hyper-parameters
    args.mlp_layer = params['mlp_layer']
    args.hidden_dim = params['hidden_dim']
    args.input_dim = params['input_dim']
    args.output_dim = params['output_dim']
    args.activation = 'relu'
    args.option = params['option']

    args.loss_fn = params['loss_fn']
    args.optimizer = params['optimizer']
    args.lr = 0
    args.decay = 0
    args.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    # finding datasets
    args.path.replace("data", "models_dir")
    dataset_name = params['data']

    # Load data
    datasets = {
            'train': load_data(dataset_name, 0),
            'dev': load_data(dataset_name, 1),
            'test': load_data(dataset_name, 2)
    }
    dim = datasets['train'][0][0].shape[0]
    
    # Load model
    model = RegFCOutputModel(args).to(args.device)
    checkpoint = torch.load(args.path, map_location=args.device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    dataset = datasets['test']
    args.batch_size = len(dataset)
    dataset = cvt_data_axis(dataset)
    input_nodes, label = tensor_data(dataset, 0, args)
    data = input_nodes.data.view((-1, dim))

    r = get_range(dataset_name)
    shape = get_data_shape(dataset_name)
    data_settings = get_data_settings(dataset_name)
    data_settings['shape'] = shape
    mean = np.zeros(dim)
    cov = np.identity(dim)

    scores, xps = [], np.zeros((n_trials, dim))
    bad_count = 0
    for i in range(n_trials):
        w = np.random.multivariate_normal(mean, cov) # sample w from a hypersphere
        w = torch.Tensor(w) / torch.Tensor(w).norm() # normalize w 
        k = find_k(w, r, shape) 
        
        sign_x = w.sign()
        w_abs = w.abs()

        ks = torch.Tensor(np.linspace(0, extend_ratio*data.max().item(), num=n_sample)).reshape(n_sample) + k
        ks = ks.T
        ws = w_abs.repeat(n_sample, 1)
        ws = ws * ks[:, None]
        wks = sign_x * ws
        
        wks = wks.to(args.device)
        with torch.no_grad():
            y0 = model(wks[0]).item()
            ys = model(wks).data.cpu().numpy()
        
        
        X = (wks - wks[0]).cpu()
        y = ys - y0
        reg = LinearRegression().fit(X, y)
        score = reg.score(X, y)
        if score < 0.9:
            #print(wks[0], score)
            bad_count += 1
        scores.append(score)

    X = scores
    
    print(f"The mean and median R-square for the input model is: {np.mean(X):.2f} and {np.median(X):.2f}")
        
if __name__ == '__main__':
    main()
