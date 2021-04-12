import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import LSTM
from torch.autograd import Variable
import itertools
from .util import median_absolute_percentage_error_compute_fn as mape
from .MLPs import *
from .util import *

DEFAULT_MODE = -1
DEFAULT_PAIR = (-1,-1)
DEFAULT_IND = -1

cls_criterion = torch.nn.CrossEntropyLoss()
mse_criterion = torch.nn.MSELoss()
lossfun = {'cls': cls_criterion, 'reg': mse_criterion, 'mape': mape}
actfun = {'relu': F.relu, 'tanh': F.tanh, 'sigmoid': F.sigmoid} 

class BasicModel(nn.Module):
    def __init__(self, args, name):
        super(BasicModel, self).__init__()
        self.name=name
        self.loss_fn = args.loss_fn
        self.activation = args.activation
        self.actfunc = actfun[self.activation]

    def train_(self, input_nodes, label):
        self.optimizer.zero_grad()

        #print(input_nodes[0].node_features)

        output = self(input_nodes)
        pred = output.data.max(1)[1]
        loss = lossfun[self.loss_fn](output.view(label.shape), label)
        mape_loss = lossfun['mape'](output.view(label.shape), label)
        loss.backward()
        self.optimizer.step()
        
        if self.loss_fn != 'cls':
            return 0, loss.cpu(), mape_loss
        
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct.to(dtype=torch.float) * 100. / len(label)
        return accuracy, loss
        
    def test_(self, input_nodes, label, print_info=False):
        with torch.no_grad():
            output = self(input_nodes)
            loss = lossfun[self.loss_fn](output.view(label.shape), label)
            mape_loss = lossfun['mape'](output.view(label.shape), label)
            if print_info:
                print(output.view(-1), label)
            if self.loss_fn != 'cls':
                return 0, loss.cpu(), mape_loss
        
            pred = output.data.max(1)[1]
            correct_ind = pred.eq(label.data).cpu()
            correct = pred.eq(label.data).cpu().sum()
            accuracy = correct.to(dtype=torch.float) * 100. / len(label)
            
            return accuracy, loss

    def pred_(self, input_nodes):
        with torch.no_grad():
            output = self(input_nodes)
            pred = output.data.max(1)[1]
            return pred

    def save_model(self, epoch):
        torch.save(self.state_dict(), 'model/epoch_{}_{:02d}.pth'.format(self.name, epoch))

