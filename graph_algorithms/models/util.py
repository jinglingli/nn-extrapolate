import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


def calc_output_size(args):
    ans = 1  #regression type of problem
    return 1

def median_absolute_percentage_error_compute_fn(y_pred, y):
    e = torch.abs(y.view_as(y_pred) - y_pred) / torch.abs(y.view_as(y_pred))
    return 100.0 * torch.mean(e)

