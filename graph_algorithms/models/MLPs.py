import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


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
            x = self.linears[layer](x)
            x = F.relu(x)

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
            x = self.linears[layer](x)
            x = F.relu(x)

        x = self.linears[self.n_layer-1](x)

        return x
