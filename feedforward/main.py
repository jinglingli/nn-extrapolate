import os
import argparse
import pickle
import random
import numpy as np
import shutil
import torch
from torch.autograd import Variable

from MLPs import *
import logging
import math

random_seed = 2
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

best_prec, best_loss, best_model_test_acc, best_model_test_loss, best_model_mape_loss = 0.0, 1e+8*1.0, 0.0, 1e+8*1.0, 1e+8*1.0
is_best = False
best_epoch = 0

def save_checkpoint(state, is_best, epoch, args):
    if not is_best:
        return
    """Saves checkpoint to disk"""
    data_path = args.data.split("/")[-1]
    directory = "models_dir/%s/%s/"%(data_path, args.filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + 'model_best.pth.tar' 
    torch.save(state, filename)

def cvt_data_axis(dataset):
    data = []
    label = []
    for d, ans in dataset:
        data.append(d)
        label.append(ans)    
    return (data, label)

def tensor_data(data, i, args):
    nodes = torch.FloatTensor(data[0][args.batch_size*i:args.batch_size*(i+1)]).to(args.device)
    if args.loss_fn == 'cls':
        ans = torch.LongTensor(data[1][args.batch_size*i:args.batch_size*(i+1)]).to(args.device)
        return nodes, ans
    else:
        ans = torch.FloatTensor(data[1][args.batch_size*i:args.batch_size*(i+1)]).to(args.device)
        return nodes.unsqueeze(dim=1), ans.unsqueeze(dim=1)

def load_data(datafile, mode):
    data = []
    label = []
    with open(datafile, 'rb') as f2:
        dataset = pickle.load(f2)[mode]
        data.extend(dataset[0])
        label.extend(dataset[1])
    dataset = []
    for i in range(len(data)):
        dataset.append( (data[i], label[i]))
    return dataset

def train(epoch, dataset, args, model):
    model.train()
    train_size = len(dataset)
    bs = args.batch_size
    random.shuffle(dataset)
    
    dataset = cvt_data_axis(dataset)
    
    running_loss, running_loss_mape = 0.0, 0.0
    accuracys = []
    losses, losses_mape = [], []
    
    batch_runs = max(1, train_size // bs)
    for batch_idx in range(batch_runs):
        input_nodes, label = tensor_data(dataset, batch_idx, args)
        accuracy, loss, mape_loss = model.train_(input_nodes, label)
        running_loss += loss
        running_loss_mape += mape_loss
        
        accuracys.append(accuracy)
        losses.append(loss)
        losses_mape.append(mape_loss)
        
        if (batch_idx + 1) % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.2f}%)] loss: {:.7f}\t'.format(epoch, batch_idx * bs, train_size, 100 * batch_idx * bs / train_size, running_loss/(1 * args.log_interval)))
            logging.info('Train Epoch: {} [{}/{} ({:.2f}%)] loss: {:.7f} \t'.format(epoch, batch_idx * bs, train_size, 100 * batch_idx * bs / train_size, running_loss/(1 * args.log_interval)))
            running_loss = 0.0
    
    avg_accuracy = sum(accuracys) *1.0 / len(accuracys)
    avg_losses = sum(losses) *1.0 / len(losses)
    avg_losses_mape = sum(losses_mape) *1.0 / len(losses_mape)
    print('\nTrain set: accuracy: {:.2f}% \t | loss: {:.7f}  \t | \t mape: {:.7f}'.format(avg_accuracy, avg_losses, avg_losses_mape))
    logging.info('\nTrain set: accuracy: {:.2f}% \t | loss: {:.7f}  \t\t | \t mape: {:.7f}'.format(avg_accuracy, avg_losses, avg_losses_mape))

def validate(epoch, dataset, args, model):
    global is_best, best_prec, best_loss
    
    model.eval()
    test_size = len(dataset)
    bs = args.batch_size
    dataset = cvt_data_axis(dataset)

    accuracys = []
    losses, mape_losses = [], []
    batch_runs = max(1, test_size // bs)
    for batch_idx in range(batch_runs):
        input_nodes, label = tensor_data(dataset, batch_idx, args)
        accuracy, loss, mape_loss = model.test_(input_nodes, label)
        accuracys.append(accuracy)
        losses.append(loss)
        mape_losses.append(mape_loss)

    avg_accuracy = sum(accuracys) *1.0 / len(accuracys)
    avg_losses = sum(losses) *1.0 / len(losses)
    avg_losses_mape = sum(mape_losses) *1.0 / len(mape_losses)
    print('Validation set: accuracy: {:.2f}% | loss: {:.7f} \t | \t mape: {:.7f}'.format(avg_accuracy, avg_losses, avg_losses_mape))
    logging.info('Validation set: accuracy: {:.2f}% | loss: {:.7f} \t | \t mape: {:.7f}'.format(avg_accuracy, avg_losses, avg_losses_mape))
    
    if args.loss_fn == 'cls':
        is_best = avg_accuracy > best_prec
    else: 
        is_best = avg_losses < best_loss
    best_prec = max(avg_accuracy, best_prec)
    best_loss = min(avg_losses, best_loss)

def test(epoch, dataset, args, model):
    global is_best, best_model_test_acc, best_model_test_loss, best_epoch, best_model_mape_loss
    
    model.eval()
    test_size = len(dataset)
    bs = args.batch_size
    dataset = cvt_data_axis(dataset)
    
    accuracys = []
    losses, mape_losses = [], []
    batch_runs = max(1, test_size // bs)
    for batch_idx in range(batch_runs):
        input_nodes, label = tensor_data(dataset, batch_idx, args)
        accuracy, loss, mape_loss = model.test_(input_nodes, label)
        accuracys.append(accuracy)
        losses.append(loss)
        mape_losses.append(mape_loss)

    avg_accuracy = sum(accuracys) *1.0 / len(accuracys)
    avg_losses = sum(losses) *1.0 / len(losses)
    avg_losses_mape = sum(mape_losses) *1.0 / len(mape_losses)
    
    print('Test set: accuracy: {:.2f}% \t | loss: {:.7f} \t | \t mape: {:.7f} \n'.format(avg_accuracy, avg_losses, avg_losses_mape))
    logging.info('Test set: accuracy: {:.2f}% \t | loss: {:.7f} \t | \t mape: {:.7f} \n'.format(avg_accuracy, avg_losses, avg_losses_mape))
    
    if is_best:
        best_model_test_acc = avg_accuracy
        best_model_test_loss = avg_losses
        best_model_mape_loss = avg_losses_mape
        best_epoch = epoch
               
    if epoch%10 == 0:
        print('************ Best model\'s test acc: {:.2f}%, test loss: {:.7f}, mape: {:.7f} (best model is from epoch {}) ************\n'.format(best_model_test_acc, best_model_test_loss, best_model_mape_loss, best_epoch))
        logging.info('************ Best model\'s test acc: {:.2f}%, test loss: {:.7f}, mape: {:.7f} (best model is from epoch {}) ************\n'.format(best_model_test_acc, best_model_test_loss, best_model_mape_loss, best_epoch))

def setup_logs(args):
    file_dir = "results"
    if not args.no_log:
        data_path = args.data.split("/")[-1]
        files_dir = '%s/%s' %(file_dir, data_path)
        args.files_dir = files_dir
        args.filename = '%s_lr%s_act%s_mlp%s_hdim%s_idim%s_odim%s_bs%s_option%s_epoch%d_seed%d.log' %(args.model, args.lr, args.activation, args.mlp_layer, args.hidden_dim, args.input_dim, args.output_dim, args.batch_size, args.option, args.epochs, random_seed)

        if not os.path.exists(files_dir):
            os.makedirs(files_dir)
        mode = 'w+'
        if args.resume:
            mode = 'a+'
        logging.basicConfig(format='%(message)s',
                            level=logging.INFO,
                            datefmt='%m-%d %H:%M',
                            filename="%s/%s" %(args.files_dir, args.filename),
                            filemode='w+')

        print(vars(args))
        logging.info(vars(args))

def resume(args, model):
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        logging.info("=> loading checkpoint '{}'".format(args.resume))

        checkpoint = torch.load(args.resume)

        args.start_epoch = checkpoint['epoch']
        best_prec = checkpoint['best_prec']
        best_model_test_acc = checkpoint['best_model_test_acc']
        best_model_test_loss = checkpoint['best_model_test_loss']
        best_model_mape_loss = checkpoint['best_model_mape_loss']
        model.load_state_dict(checkpoint['state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        logging.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        logging.info("=> no checkpoint found at '{}'".format(args.resume))
    return model


def main():
    parser = argparse.ArgumentParser()

    #Model specifications
    parser.add_argument('--model', type=str, default='FeedForward', help='choose which model')
    parser.add_argument('--activation', type=str, choices=['sin', 'square', 'tanh', 'exp', 'log', 'relu', 'gelu', 'linear','sigmoid'], default='relu', help='activation function')
    parser.add_argument('--option', type=str, choices=['A', 'B', 'None'], default='None', help='initialization options')
    parser.add_argument('--mlp_layer', type=int, default=3, help='number of layers for MLPs')
    parser.add_argument('--hidden_dim', type=int, default=256, help='feature hidden dimension of MLPs')
    parser.add_argument('--input_dim', type=int, default=0, help='input dim')
    parser.add_argument('--output_dim', type=int, default=1, help='output dim')

    # Training settings
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--resume', type=str, help='resume from model stored')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=250, help='number of epochs to train')
    parser.add_argument('--loss_fn', type=str, choices=['cls', 'reg', 'mape'], default='reg', help='classification or regression loss')
    parser.add_argument('--optimizer', type=str, choices=['Adam', 'SGD'], default='Adam', help='Adam or SGD')

    # Logging and storage settings
    parser.add_argument('--save_model', action='store_true', default=False, help='flag to store the training models')
    parser.add_argument('--no_log', action='store_true', default=False, help='flag to disable logging of results')
    parser.add_argument('--log_interval', type=int, default=100, help='how many batches to wait before logging training status')
    parser.add_argument('--filename', type=str, default='', help='file to store the training log')
    parser.add_argument('--files_dir', type=str, default='', help='the directory to store trained models logs')
    
    # Data settings
    parser.add_argument('--data', type=str, default='', help='path to the data file')

    args = parser.parse_args()
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        
    train_datasets = load_data(args.data, 0)
    validation_datasets = load_data(args.data, 1)
    test_datasets = load_data(args.data, 2)

    args.input_dim = len(train_datasets[0][0])
    if args.loss_fn == 'cls':
        args.output_dim = 2 # binary classification
    else:
        args.output_dim = len(train_datasets[0][1])
    
    setup_logs(args)
    
    model = FeedForward(args).to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, step_size=50, gamma=0.5)
    
    bs = args.batch_size
    
    model_dirs = './models_dir'
    try:
        os.makedirs(model_dirs)
    except:
        print('directory {} already exists'.format(model_dirs))

    if args.epochs == 0:
        epoch = 0
        validate(epoch, validation_datasets, args, model)
        test(epoch, test_datasets, args, model)
        args.epochs = -1

    for epoch in range(1, args.epochs + 1):
        if epoch == 1:
            is_best = True
        train(epoch, train_datasets, args, model)
        validate(epoch, validation_datasets, args, model)
        test(epoch, test_datasets, args, model)
        scheduler.step()
        if is_best and args.save_model:
            save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.model,
                    'args': args, 
                    'state_dict': model.state_dict(),
                    'best_prec': best_prec,
                    'best_model_test_acc': best_model_test_acc,
                    'best_model_test_loss': best_model_test_loss,
                    'best_model_mape_loss': best_model_mape_loss,
                    'optimizer' : model.optimizer.state_dict(),
                }, is_best, epoch, args)

    print('************ Best model\'s test acc: {:.2f}%, test loss: {:.7f} throughout training (best model is from epoch {}) ************\n'.format(best_model_test_acc, best_model_test_loss, best_epoch))
    logging.info('************ Best model\'s test acc: {:.2f}%, test loss: {:.7f} throughout training (best model is from epoch {}) ************\n'.format(best_model_test_acc, best_model_test_loss, best_epoch))

if __name__ == '__main__':
    main()
