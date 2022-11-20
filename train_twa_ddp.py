import argparse
import time
import os
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
from utils import get_imagenet_dataset, get_model, set_seed, adjust_learning_rate, bn_update, eval_model, Logger

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

########################## parse arguments ##########################
parser = argparse.ArgumentParser(description='TWA ddp')
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
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--print-freq', '-p', default=200, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
# env
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
# project subspace setting 
parser.add_argument('--params_start', default=0, type=int, metavar='N',
                    help='which idx start for project subspace') 
parser.add_argument('--params_end', default=51, type=int, metavar='N',
                    help='which idx end for project subspace') 
parser.add_argument('--train_start', default=0, type=int, metavar='N',
                    help='which idx start for training')     
# optimizer and scheduler
parser.add_argument('--opt', metavar='OPT', help='optimization method for TWA', 
                    default='SGD', choices=['SGD'])
parser.add_argument('--schedule', metavar='SCHE', help='learning rate schedule for P-SGD', 
                    default='step', choices=['step', 'constant', 'linear'])
parser.add_argument('--lr', default=1, type=float, metavar='N',
                    help='lr for PSGD')
# ddp
parser.add_argument("--local_rank", default=-1, type=int)

args = parser.parse_args()
set_seed(args.randomseed)

def reduce_value(value, op=dist.ReduceOp.SUM):
    world_size = dist.get_world_size()
    if world_size < 2:  # single GPU
        return value
 
    with torch.no_grad():
        dist.all_reduce(value, op)
        return value

def get_model_param_vec_torch(model):
    """
    Return model parameters as a vector
    """
    vec = []
    for _, param in model.named_parameters():
        vec.append(param.data.detach().reshape(-1))
    return torch.cat(vec, 0)

def get_model_grad_vec_torch(model):
    """
    Return model grad as a vector
    """
    vec = []
    for _, param in model.named_parameters():
        vec.append(param.grad.detach().reshape(-1))
    return torch.cat(vec, 0)

def update_grad(model, grad_vec):
    """
    Update model grad
    """
    idx = 0
    for _, param in model.named_parameters():
        arr_shape = param.grad.shape
        size = arr_shape.numel()
        param.grad.data = grad_vec[idx:idx+size].reshape(arr_shape).clone()
        idx += size

def update_param(model, param_vec):
    idx = 0
    for _, param in model.named_parameters():
        arr_shape = param.data.shape
        size = arr_shape.numel()
        param.data = param_vec[idx:idx+size].reshape(arr_shape).clone()
        idx += size

def main(args):
    # DDP initialize backend
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl')
    world_size = torch.distributed.get_world_size()
    device = torch.device("cuda", args.local_rank)
    dist.barrier() # Synchronizes all processes

    if dist.get_rank() == 0:
        # Check the save_dir exists or not
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        # Check the log_dir exists or not
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)

        sys.stdout = Logger(os.path.join(args.log_dir, args.log_name))
        print('twa-ddp')
        print('save dir:', args.save_dir)
        print('log dir:', args.log_dir)
    
    # Define model
    model = get_model(args).to(device)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    cudnn.benchmark = True

    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)    

    optimizer = optim.SGD(model.parameters(), lr=args.lr, \
                            momentum=args.momentum, \
                            weight_decay=args.weight_decay)
    optimizer.zero_grad()

    if args.schedule == 'step':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, \
                    milestones=[int(args.epochs*0.5), int(args.epochs*0.75+0.9)], last_epoch=args.start_epoch - 1)
    elif args.schedule == 'constant' or args.schedule == 'linear':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, \
                    milestones=[args.epochs + 1], last_epoch=args.start_epoch - 1)
    
    # Prepare Dataloader
    train_dataset, val_dataset = get_imagenet_dataset()
    assert args.batch_size % world_size == 0, f"Batch size {args.batch_size} cannot be divided evenly by world size {world_size}"
    batch_size_per_GPU = args.batch_size // world_size
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size_per_GPU, sampler=train_sampler,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size_per_GPU, sampler=val_sampler,
        num_workers=args.workers)

    args.total_iters = len(train_loader) * args.epochs
    args.current_iters = 0

    ########################## extract subspaces ##########################
    # Load sampled model parameters
    if dist.get_rank() == 0:
        print('weight decay:', args.weight_decay)
        print('params: from', args.params_start, 'to', args.params_end)
    W = []
    for i in range(args.params_start, args.params_end):
        if i%2==1: continue 
        model.load_state_dict(torch.load(os.path.join(args.save_dir, f'{i}.pt')))
        W.append(get_model_param_vec_torch(model))
    W = torch.stack(W, dim=0)

    # Schmidt
    P = W
    n_dim = P.shape[0]
    coeff = torch.eye(n_dim).to(device)
    for i in range(n_dim):
        if i > 0:
            tmp = torch.mm(P[:i, :], P[i].reshape(-1, 1))
            P[i] -= torch.mm(P[:i, :].T, tmp).reshape(-1)
            coeff[i] -= torch.mm(coeff[:i, :].T, tmp).reshape(-1)
        tmp = torch.norm(P[i])
        P[i] /= tmp
        coeff[i] /= tmp
    coeff_inv = coeff.T.inverse()

    # Slice P 
    slice_start = (n_dim//world_size)*dist.get_rank()
    if dist.get_rank() == world_size-1:
        slice_P = P[slice_start:,:].clone()
    else:
        slice_end = (n_dim//world_size)*(dist.get_rank()+1)
        slice_P = P[slice_start:slice_end,:].clone()
    if dist.get_rank() == 0:
        print(f'W: {W.shape} {W.device}')
        print(f'P: {P.shape} {P.device}')
        print(f'Sliced P: {slice_P.shape} {slice_P.device}')
    del P 
    torch.cuda.empty_cache()
    dist.barrier() # Synchronizes all processes

    # set the start point
    if args.train_start >= 0:
        model.load_state_dict(torch.load(os.path.join(args.save_dir, str(args.train_start) + '.pt')))
        if dist.get_rank() == 0:
            print('train start:', args.train_start)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    if dist.get_rank() == 0:
        print('Train:', (args.start_epoch, args.epochs))
    end = time.time()
    his_train_acc, his_test_acc, his_train_loss, his_test_loss = [], [], [], []
    best_prec1 = 0
    for epoch in range(args.start_epoch, args.epochs):
        # Train for one epoch
        if dist.get_rank() == 0:
            print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train_loss, train_prec1 = train(train_loader, model, criterion, optimizer, 
                                        args, epoch, slice_P, device, world_size)
        his_train_loss.append(train_loss)
        his_train_acc.append(train_prec1)

        if args.schedule != 'linear':
            lr_scheduler.step()

        # Evaluate on validation set
        test_loss, test_prec1 = validate(val_loader, model, criterion, device, world_size)
        his_test_loss.append(test_loss)
        his_test_acc.append(test_prec1)

        # Remember best prec@1 and save checkpoint
        best_prec1 = max(test_prec1, best_prec1)
        if dist.get_rank() == 0:
            print(f'Epoch: [{epoch}] * Best Prec@1 {best_prec1:.3f}')
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'ddp{epoch}.pt'))

    if dist.get_rank() == 0:
        print('total time:', time.time() - end)
        print('train loss: ', his_train_loss)
        print('train acc: ', his_train_acc)
        print('test loss: ', his_test_loss)
        print('test acc: ', his_test_acc)      
        print('best_prec1:', best_prec1)


def train(train_loader, model, criterion, optimizer, args, epoch, P, device, world_size=1):
    # Run one train epoch
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    correctes = 0
    count = 0

    # Switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        # Load batch data to cuda
        target = target.to(device)
        input = input.to(device)

        batch_size = torch.tensor(target.size(0)).to(device)
        reduce_value(batch_size)
        count += batch_size

        # Compute output
        output = model(input)
        loss = criterion(output, target)

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        
        if args.schedule == 'linear':
            adjust_learning_rate(optimizer, (1 - args.current_iters / args.total_iters) * args.lr)
            args.current_iters += 1

        project_gradient(model, P)
        optimizer.step()

        # Measure accuracy and record loss
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_1 = correct[:1].view(-1).float().sum(0)
        reduce_value(correct_1)
        correctes += correct_1

        reduce_value(loss)
        loss /= world_size
        losses.update(loss.item(), input.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if (i % args.print_freq == 0 or i == len(train_loader)-1) and dist.get_rank() == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Prec@1 {correct_1/batch_size*100:.3f} ({correctes/count*100:.3f})')
        
    return losses.avg, correctes/count*100

def project_gradient(model, P):
    grad = get_model_grad_vec_torch(model)
    gk = torch.mm(P, grad.reshape(-1, 1))
    grad_proj = torch.mm(P.transpose(0, 1), gk)
    reduce_value(grad_proj) # Sum-reduce projected gradients on different GPUs

    update_grad(model, grad_proj.reshape(-1))

def validate(val_loader, model, criterion, device, world_size=1):
    # Run evaluation 

    batch_time = AverageMeter()
    losses = AverageMeter()
    correctes = 0
    count = 0

    # Switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.to(device)
            input = input.to(device)

            batch_size = torch.tensor(target.size(0)).to(device)
            reduce_value(batch_size)
            count += batch_size

            # Compute output
            output = model(input)
            loss = criterion(output, target)

            # Measure accuracy and record loss
            _, pred = output.topk(1, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            correct_1 = correct[:1].view(-1).float().sum(0)
            reduce_value(correct_1)
            correctes += correct_1

            reduce_value(loss)
            loss /= world_size
            losses.update(loss.item(), input.size(0))


            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and dist.get_rank() == 0:
                print(f'Test: [{i}/{len(val_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Prec@1 {correct_1/batch_size*100:.3f} ({correctes/count*100:.3f})')

    print(f' * Prec@1 {correctes/count*100:.3f}')

    return losses.avg, correctes/count*100


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
    main(args)