import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import LSTM
from torch.autograd import Variable
import itertools
from math import *

DEFAULT_MODE = -1
DEFAULT_PAIR = (-1,-1)
DEFAULT_IND = -1

def square(x):
    return x ** 2

def mape(y_pred, y):
    e = torch.abs(y.view_as(y_pred) - y_pred) / torch.abs(y.view_as(y_pred))
    return 100.0 * torch.median(e)

cls_criterion = torch.nn.CrossEntropyLoss()
mse_criterion = torch.nn.MSELoss()
lossfun = {'cls': cls_criterion, 'reg': mse_criterion, 'mape': mape}
actfun = {'sin': torch.sin, 'square': square, 'tanh': F.tanh, 'exp': torch.exp, 'log':torch.log, 'relu': F.relu, 'gelu': F.gelu, 'sigmoid': F.sigmoid}

class BasicModel(nn.Module):
    def __init__(self, args, name):
        super(BasicModel, self).__init__()
        self.name=name
        self.loss_fn = args.loss_fn
        self.activation = args.activation
        self.actfunc = actfun[self.activation]
        
    def train_(self, input_nodes, label):
        self.optimizer.zero_grad()
        output = self(input_nodes)
        
        if self.loss_fn != 'cls':
            loss = lossfun[self.loss_fn](output.view(label.shape), label)
            
            loss.backward()
            self.optimizer.step()
            
            mape_loss = lossfun['mape'](output.view(label.shape), label)
            return 0, loss.cpu(), mape_loss.cpu()
        else:
            loss = lossfun[self.loss_fn](output, label)
            
            loss.backward()
            self.optimizer.step()
            
            pred = output.data.max(1)[1]
            correct = pred.eq(label.data).cpu().sum()
            accuracy = correct.to(dtype=torch.float) * 100. / len(label)
            
        return accuracy, loss.cpu(), 0
        
    def test_(self, input_nodes, label, print_info=False):
        with torch.no_grad():
            output = self(input_nodes)
            if print_info:
                print(output.view(-1), label)
                
            if self.loss_fn != 'cls':
                loss = lossfun[self.loss_fn](output.view(label.shape), label)
                mape_loss = lossfun['mape'](output.view(label.shape), label)
                return 0, loss.cpu(), mape_loss.cpu()
            else:
                loss = lossfun[self.loss_fn](output, label)
                pred = output.data.max(1)[1]
                correct_ind = pred.eq(label.data).cpu()
                correct = pred.eq(label.data).cpu().sum()
                accuracy = correct.to(dtype=torch.float) * 100. / len(label)
                
                return accuracy, loss.cpu(), 0

    def pred_(self, input_nodes):
        with torch.no_grad():
            output = self(input_nodes)
            pred = output.data.max(1)[1]
            return pred

    def save_model(self, epoch):
        torch.save(self.state_dict(), 'model/epoch_{}_{:02d}.pth'.format(self.name, epoch))
        
class FeedForward(BasicModel):
    def __init__(self, args):
        super(FeedForward, self).__init__(args, 'FeedForward')
        self.n_layer = args.mlp_layer 
        self.input_dim, self.hidden_dim, self.output_dim = args.input_dim, args.hidden_dim, args.output_dim
        self.option = args.option
        
        if self.n_layer == 1:
            self.linears = nn.ModuleList([nn.Linear(self.input_dim, self.output_dim)])
        else:
            self.linears = nn.ModuleList([nn.Linear(self.input_dim, self.hidden_dim)])
            for layer in range(self.n_layer-2):
                self.linears.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.linears.append(nn.Linear(self.hidden_dim, int(self.output_dim)))
        
        if self.option == 'A':
            for m in self.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif self.option == 'B':
            for m in self.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.normal_(m.weight, mean=0.0, std=sqrt(2))

        if args.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.decay)
        else:
            self.optimizer = optim.SGD(self.parameters(), lr=args.lr, weight_decay=args.decay)
        
    def forward(self, x):
        if self.n_layer == 1:
            layer = self.linears[self.n_layer-1]
            x = layer(x)
            if self.option == 'B':
                x = x / sqrt(layer.out_features * 1.0)
            return x
        
        for i in range(self.n_layer-1):
            layer = self.linears[i]
            x = layer(x)
            if not self.activation == 'linear':
                x = self.actfunc(x)
            if self.option == 'B':
                x = x / sqrt(layer.out_features * 1.0)

        x = self.linears[self.n_layer-1](x)
            
        return x