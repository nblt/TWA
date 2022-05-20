from random import choices
import argparse
import _osx_support
import time
import os
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
import utils
from utils import get_datasets, get_model, set_seed, adjust_learning_rate, bn_update, eval_model, Logger

########################## parse arguments ##########################
parser = argparse.ArgumentParser(description='SGD in Projected Subspace')
parser.add_argument('--EXP', metavar='EXP', help='experiment name', default='P-SGD')
parser.add_argument('--arch', '-a', metavar='ARCH', default='VGG16BN',
                    help='model architecture (default: VGG16BN)')
parser.add_argument('--datasets', metavar='DATASETS', default='CIFAR10', type=str,
                    help='The training datasets')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('-acc', '--accumulate', default=1, type=int,
                    metavar='A', help='accumulate times for batch gradient (default: 1)')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--print-freq', '-p', default=200, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--randomseed', 
                    help='Randomseed for training and initialization',
                    type=int, default=1)           
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--log-dir', dest='log_dir',
                    help='The directory used to save the log',
                    default='save_temp', type=str)
parser.add_argument('--log-name', dest='log_name',
                    help='The log file name',
                    default='log', type=str)

########################## P-SGD setting ##########################
parser.add_argument('--extract', metavar='EXTRACT', help='method for extracting subspace', 
                    default='Schmidt', choices=['Schmidt', 'PCA'])
parser.add_argument('--params_start', default=0, type=int, metavar='N',
                    help='which idx start for PCA') 
parser.add_argument('--params_end', default=51, type=int, metavar='N',
                    help='which idx end for PCA') 
parser.add_argument('--train_start', default=0, type=int, metavar='N',
                    help='which idx start for training')     
# PCA
parser.add_argument('--n_components', default=40, type=int, metavar='N',
                    help='n_components for PCA')    
parser.add_argument('--opt', metavar='OPT', help='optimization method for TWA', 
                    default='SGD', choices=['SGD'])
parser.add_argument('--schedule', metavar='SCHE', help='learning rate schedule for P-SGD', 
                    default='step', choices=['step', 'constant', 'linear'])
parser.add_argument('--lr', default=1, type=float, metavar='N',
                    help='lr for PSGD')

args = parser.parse_args()
set_seed(args.randomseed)
best_prec1 = 0
P = None
train_acc, test_acc, train_loss, test_loss = [], [], [], []

def get_model_param_vec(model):
    """
    Return model parameters as a vector
    """
    vec = []
    for name,param in model.named_parameters():
        vec.append(param.detach().cpu().numpy().reshape(-1))
    return np.concatenate(vec, 0)

def get_model_param_vec_torch(model):
    """
    Return model parameters as a vector
    """
    vec = []
    for name,param in model.named_parameters():
        vec.append(param.data.detach().reshape(-1))
    return torch.cat(vec, 0)

def get_model_grad_vec(model):
    """
    Return model grad as a vector
    """
    vec = []
    for name,param in model.named_parameters():
        vec.append(param.grad.detach().reshape(-1))
    return torch.cat(vec, 0)

def update_grad(model, grad_vec):
    """
    Update model grad
    """
    idx = 0
    for name,param in model.named_parameters():
        arr_shape = param.grad.shape
        size = arr_shape.numel()
        param.grad.data = grad_vec[idx:idx+size].reshape(arr_shape).clone()
        idx += size

def update_param(model, param_vec):
    idx = 0
    for name,param in model.named_parameters():
        arr_shape = param.data.shape
        size = arr_shape.numel()
        param.data = param_vec[idx:idx+size].reshape(arr_shape).clone()
        idx += size

def main():

    global args, best_prec1, Bk, P, coeff, coeff_inv

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Check the log_dir exists or not
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    sys.stdout = Logger(os.path.join(args.log_dir, args.log_name))
    print ('twa-psgd')
    print ('save dir:', args.save_dir)
    print ('log dir:', args.log_dir)
    
    # Define model
    if args.datasets == 'ImageNet':
        model = torch.nn.DataParallel(get_model(args))
    else:
        model = get_model(args)
    model.cuda()
    cudnn.benchmark = True

    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()    

    optimizer = optim.SGD(model.parameters(), lr=args.lr, \
                            momentum=args.momentum, \
                            weight_decay=args.weight_decay)

    if args.schedule == 'step':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, \
                    milestones=[int(args.epochs*0.5), int(args.epochs*0.75+0.9)], last_epoch=args.start_epoch - 1)

    elif args.schedule == 'constant' or args.schedule == 'linear':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, \
                    milestones=[args.epochs + 1], last_epoch=args.start_epoch - 1)
    
    optimizer.zero_grad()
    
    # Prepare Dataloader
    train_loader, val_loader = get_datasets(args)    

    args.total_iters = len(train_loader) * args.epochs
    args.current_iters = 0

    ########################## extract subspaces ##########################
    # Load sampled model parameters
    print ('weight decay:', args.weight_decay)
    print ('params: from', args.params_start, 'to', args.params_end)
    W = []
    for i in range(args.params_start, args.params_end):
        model.load_state_dict(torch.load(os.path.join(args.save_dir,  str(i) +  '.pt')))
        W.append(get_model_param_vec(model))
    W = np.array(W)
    print ('W:', W.shape)

    # Evaluate swa performance
    center = torch.from_numpy(np.mean(W, axis=0)).cuda()

    update_param(model, center)
    bn_update(train_loader, model)
    print (utils.eval_model(val_loader, model, criterion))

    if args.extract == 'PCA':
        # Obtain base variables through PCA
        print ('PCA')
        pca = PCA(n_components=args.n_components, svd_solver='randomized')
        pca.fit_transform(W)
        P = np.array(pca.components_)
        print ('ratio:', list(pca.explained_variance_ratio_))
        print ('variance:', list(pca.explained_variance_))
        print ('P:', P.shape)
        P = torch.from_numpy(P).cuda()

    elif args.extract == 'Schmidt':
        P = torch.from_numpy(np.array(W)).cuda()
        n_dim = P.shape[0]
        args.n_components = n_dim
        coeff = torch.eye(n_dim).cuda()
        for i in range(n_dim):
            if i > 0:
                tmp = torch.mm(P[:i, :], P[i].reshape(-1, 1))
                P[i] -= torch.mm(P[:i, :].T, tmp).reshape(-1)
                coeff[i] -= torch.mm(coeff[:i, :].T, tmp).reshape(-1)
            tmp = torch.norm(P[i])
            P[i] /= tmp
            coeff[i] /= tmp
        coeff_inv = coeff.T.inverse()

        print (P.shape)

    # set the start point
    if args.train_start >= 0:
        model.load_state_dict(torch.load(os.path.join(args.save_dir,  str(args.train_start) +  '.pt')))
        print ('train start:', args.train_start)

    if args.half:
        model.half()
        criterion.half()

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    print ('Train:', (args.start_epoch, args.epochs))
    end = time.time()
    p0 = get_model_param_vec(model)

    for epoch in range(args.start_epoch, args.epochs):
        # Train for one epoch

        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, args, epoch, center)

        if args.schedule != 'linear':
            lr_scheduler.step()

        # Evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # Remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

    print ('Save final model')
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'PSGD.pt'))

    bn_update(train_loader, model)
    print (utils.eval_model(val_loader, model, criterion))

    print ('total time:', time.time() - end)
    print ('train loss: ', train_loss)
    print ('train acc: ', train_acc)
    print ('test loss: ', test_loss)
    print ('test acc: ', test_acc)      
    print ('best_prec1:', best_prec1)


def train(train_loader, model, criterion, optimizer, args, epoch, center):
    # Run one train epoch

    global P, W, iters, T, train_loss, train_acc, search_times, coeff
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # Switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # Measure data loading time
        data_time.update(time.time() - end)

        # Load batch data to cuda
        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()

        # Compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        gk = get_model_grad_vec(model)
        
        if args.schedule == 'linear':
            adjust_learning_rate(optimizer, (1 - args.current_iters / args.total_iters) * args.lr)
            args.current_iters += 1

        if args.opt == 'SGD':
            P_SGD(model, optimizer, gk, center)

        # Measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0 or i == len(train_loader)-1:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
        
    train_loss.append(losses.avg)
    train_acc.append(top1.avg)

def P_SGD(model, optimizer, grad, center):

    # p = get_model_param_vec_torch(model)
    gk = torch.mm(P, grad.reshape(-1,1))
    grad_proj = torch.mm(P.transpose(0, 1), gk)
    
    update_grad(model, grad_proj.reshape(-1))

    optimizer.step()

def validate(val_loader, model, criterion):
    # Run evaluation

    global test_acc, test_loss  

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # Switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if args.half:
                input_var = input_var.half()

            # Compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # Measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    # Store the test loss and test accuracy
    test_loss.append(losses.avg)
    test_acc.append(top1.avg)

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    # Save the training model

    torch.save(state, filename)

class AverageMeter(object):
    # Computes and stores the average and current value

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    # Computes the precision@k for the specified values of k

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()