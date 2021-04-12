import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from util import *

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
        if self.name == 'InteractionNetwork':
            bs = label.shape[0]
            output = self(input_nodes, bs)
        else:
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
            if self.name == 'InteractionNetwork':
                bs = label.shape[0]
                output = self(input_nodes, bs)
            else:
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

class MLP(nn.Module):
    def __init__(self, n_layer, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()

        self.n_layer = n_layer 
        self.linears = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
        for layer in range(n_layer-2):
            self.linears.append(nn.Linear(hidden_dim, hidden_dim))
        self.linears.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for layer in range(self.n_layer):
            x = self.linears[layer](x)
            x = F.relu(x)

        return x

class FCOutputModel(nn.Module):
    def __init__(self, n_layer, input_dim, hidden_dim, output_dim):
        super(FCOutputModel, self).__init__()
        self.n_layer = n_layer 
        if self.n_layer == 1:
            self.linears = nn.ModuleList([nn.Linear(input_dim, output_dim)])
        else:
            self.linears = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
            for layer in range(n_layer-2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, int(output_dim)))

    def forward(self, x):
        if self.n_layer == 1:
            x = self.linears[self.n_layer-1](x)
            return x
        
        for layer in range(self.n_layer-1):
            #if layer == self.n_layer - 2:
                #x = F.dropout(x)
            x = self.linears[layer](x)
            x = F.relu(x)

        #x = F.dropout(x)
        x = self.linears[self.n_layer-1](x)

        return F.log_softmax(x)
    
class RegFCOutputModel(nn.Module):
    def __init__(self, n_layer, input_dim, hidden_dim, output_dim):
        super(RegFCOutputModel, self).__init__()
        self.n_layer = n_layer 
        if self.n_layer == 1:
            self.linears = nn.ModuleList([nn.Linear(input_dim, output_dim)])
        else:
            self.linears = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
            for layer in range(n_layer-2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, int(output_dim)))

    def forward(self, x):
        if self.n_layer == 1:
            x = self.linears[self.n_layer-1](x)
            return x
        
        for layer in range(self.n_layer-1):
            #if layer == self.n_layer - 2:
                #x = F.dropout(x)
            x = self.linears[layer](x)
            x = F.relu(x)

        #x = F.dropout(x)
        x = self.linears[self.n_layer-1](x)

        return x

class Bottleneck(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Bottleneck, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(input_dim, hidden_dim), nn.Linear(hidden_dim, output_dim)])
        
    def forward(self, x):
        x_input = x
        x = self.linears[0](x)
        x = self.linears[1](x)
        x = x_input + x
        
        return x

class FCOutputModel_SkipConnection(nn.Module):
    def __init__(self, n_layer, input_dim, hidden_dim, output_dim, block=Bottleneck):
        super(FCOutputModel_SkipConnection, self).__init__()
        self.n_layer = n_layer 
        self.blocks = torch.nn.ModuleList()
        self.blocks.append(nn.Linear(input_dim, hidden_dim))
        for layer in range(0, self.n_layer-2, 2):
            self.blocks.append(block(hidden_dim, hidden_dim, hidden_dim))
        self.blocks.append(nn.Linear(hidden_dim, output_dim)) 

    def forward(self, x):
        index = 0
        x = self.blocks[index](x)
        index = index + 1
        
        for layer in range(0, self.n_layer-2, 2):
            #if layer == (self.n_layer - 4):
                #x = F.dropout(x)
            x = self.blocks[index](x)
            x = F.relu(x)
            index = index + 1

        #x = F.dropout(x)
        x = self.blocks[index](x)

        return F.log_softmax(x)
