import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import time 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Subset
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import get_model, set_seed, get_datasets

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)
    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)
    return _lr_adjuster

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-location", type=str, help="The root directory for the datasets.")
    parser.add_argument("--model-location", type=str, help="Where to download the models.")
    parser.add_argument('-a', '--arch', default='resnet18')
    parser.add_argument('--datasets', metavar='DATASETS', default='CIFAR10', type=str)
    parser.add_argument("--cosine-lr", action="store_true", default=False)
    parser.add_argument("--test-swa", action="store_true", default=False)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument('--print-freq', '-p', default=50, type=int, help='print frequency (default: 50)')
    parser.add_argument('--randomseed', help='Randomseed for training and initialization', type=int, default=1)           
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--optimizer", type=str, default='sgd')
    parser.add_argument("--save-dir", type=str, default='partition')
    parser.add_argument("--layer-size", type=int, default=0)
    parser.add_argument('--epochs', default=5, type=int, help='number of total epochs to run')
    parser.add_argument('--labelsmooth', dest='labelsmooth', action='store_true', help='use label smooth (0.1)')
    parser.add_argument('--params_start', default=0, type=int, help='which idx start for TWA') 
    parser.add_argument('--params_end', default=51, type=int, help='which idx end for TWA')
    parser.add_argument("--local-rank", default=-1, type=int)
    parser.add_argument('--split', dest='split', action='store_true', help='use split dataset')
    parser.add_argument('--val_ratio', default=10, type=int, metavar='R', help='val ratio for training')
    parser.add_argument('--randaug', dest='randaug', action='store_true', help='use randaug data augmentation')

    parser.add_argument('--img_size', type=int, default=224, help="Resolution size")
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT', help='Drop block rate (default: None)')
    parser.add_argument('--ddp', dest='ddp', action='store_true', help='ddp training')
    parser.add_argument('--finetune', dest='finetune', action='store_true', help='finetune training')
    parser.add_argument('--rho', type=float, default=0.0, metavar='RHO', help='radius for SAM')
    parser.add_argument('--model_batch', type=int, default=5, help="batch of models for training once")
    parser.add_argument('--Pbits', type=int, default=4, help="bits for the projection matrix")
    parser.add_argument('--models_epochs', type=int, default=1, help="number of training epochs for models")
                
    return parser.parse_args()

class LabelSmoothCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, label, smoothing=0.1):
        pred = F.softmax(pred, dim=1)
        one_hot_label = F.one_hot(label, pred.size(1)).float()
        smoothed_one_hot_label = (1.0 - smoothing) * one_hot_label + smoothing / pred.size(1)
        loss = (-torch.log(pred)) * smoothed_one_hot_label
        loss = loss.sum(axis=1, keepdim=False)
        loss = loss.mean()

        return loss

def reduce_value(value, op=dist.ReduceOp.SUM):
    args.world_size = dist.get_world_size()
    if args.world_size < 2:  # single GPU
        return value
 
    with torch.no_grad():
        dist.all_reduce(value, op)
        return value
                        
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
        
def bn_update(loader, model):
    model.train()
    for i, (input, target) in enumerate(loader):
        # if (i > 1000): break
        input = input.cuda()
        target = target.cuda()
        model(input)
    
def validate(test_loader, model, criterion, device):
    global train_loader
    
    # bn_update(train_loader, model)
    # Run evaluation 

    batch_time = AverageMeter()
    losses = AverageMeter()
    correctes = 0
    count = 0

    # Switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            input = input.to(device)
            target = target.to(device)

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

            loss = loss * input.size(0)
            reduce_value(loss)
            losses.update(loss.item() / batch_size.item(), batch_size.item())

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and dist.get_rank() == 0:
                print(f'Test: [{i}/{len(test_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Prec@1 {correct_1/batch_size*100:.3f} ({correctes/count*100:.3f})')

    print(f' * Prec@1 {correctes/count*100:.3f}')

    return losses.avg, correctes/count*100

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

def get_model_param_vec_torch(model):
    """
    Return model grad as a vector
    """
    vec = []
    for param in model.parameters():
        vec.append(param.data.clone().reshape(-1))
    return torch.cat(vec, 0)

def get_model_grad_vec_torch(model):
    """
    Return model grad as a vector
    """
    vec = []
    for param in model.parameters():
        if param.grad is not None:
            vec.append(param.grad.clone().reshape(-1))
        else:
            vec.append(torch.zeros_like(param).reshape(-1).to(param))
    return torch.cat(vec, 0)

def update_grad(model, grad_vec):
    """
    Update model grad
    """
    idx = 0
    for param in model.parameters():
        arr_shape = param.data.shape
        size = arr_shape.numel()
        if param.grad is not None:
            param.grad.data = grad_vec[idx:idx+size].reshape(arr_shape).clone()
        idx += size
        
def update_param(model, param_vec):
    """
    Update model grad
    """
    idx = 0
    for param in model.parameters():
        arr_shape = param.data.shape
        size = arr_shape.numel()
        param.data = param_vec[idx:idx+size].reshape(arr_shape).clone()
        idx += size

def zero_grad(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()
            
def quantize_to_int8(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    
    grids = (2 ** args.Pbits) - 1
    scale = (max_val - min_val) / grids
    zero_point = min_val

    quantized_tensor = torch.round((tensor - zero_point) / scale)
    quantized_tensor = quantized_tensor.to(torch.uint8)

    return quantized_tensor, scale, zero_point

def P_Step(model, optimizer, X, Q, layers, center):
    grad = get_model_grad_vec_torch(model)
    p = get_model_param_vec_torch(model)

    idx = 0
    for i, size in enumerate(layers):
        X.grad.data[:, i] = (Q[i][0] * (Q[i][1] * grad[idx:idx+size]).reshape(-1)).sum(dim=1) + \
            Q[i][2] * grad[idx:idx+size].sum()
        idx += size

    optimizer.step()
    param_proj = torch.zeros_like(grad, dtype=torch.float).to(grad)

    idx = 0
    for i, size in enumerate(layers):
        param_proj[idx:idx+size] = (Q[i][0].transpose(0, 1) * (Q[i][1] * X.data[:, i].reshape(-1))).sum(dim=1) + \
            Q[i][2] * X.data[:, i].sum()
        idx += size
    
    reduce_value(param_proj)
    update_param(model, param_proj + center)

def bn_update(loader, model, device):
    model.train()
    for i, (input, target) in enumerate(loader):
        # if (i > 1000): break
        input = input.to(device)
        target = target.to(device)
        model(input)
        
def train(model, optimier, center, Q, train_loader, test_loader, args, device):
    if args.labelsmooth:
        print ('label smooth: 0.1')
        criterion = LabelSmoothCELoss().to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)    

    dist.barrier()
    torch.cuda.empty_cache()

    if args.test_swa:
        # validate swa performance
        print ('SWA:')
        bn_update(train_loader, model, device)
        validate(test_loader, model, criterion, device)

    end = time.time()
    _step = 0
    for epoch in range(args.epochs):
        # Run one train epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train_loader.sampler.set_epoch(epoch)
        count = 0
        correctes = 0
        batch_time = AverageMeter()
        data_time = AverageMeter()
        pstep_time = AverageMeter()
        losses = AverageMeter()

        # Switch to train mode
        model.train()
        for i, (input, target) in enumerate(train_loader):
            input = input.to(device)
            target = target.to(device)

            if args.cosine_lr:
                lr_scheduler(_step)
            _step += 1

            # Measure data loading time
            data_time.update(time.time() - end)

            batch_size = torch.tensor(target.size(0)).to(device)
            reduce_value(batch_size)
            count += batch_size

            # Compute output
            output = model(input)
            loss = criterion(output, target)

            # Compute gradient and do SGD step
            zero_grad(model)
            loss.backward()
            
            # SAM
            if args.rho > 0:
                p0 = get_model_param_vec_torch(model)
                g0 = get_model_grad_vec_torch(model)
                g0 /= g0.norm()
                update_param(model, p0 + g0 * args.rho)
                output = model(input)
                loss_sam = criterion(output, target) * 0.1
                # zero_grad(model)
                loss_sam.backward()
                update_param(model, p0)

            if 'vit' in args.arch:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            end1 = time.time()
            P_Step(model, optimizer, X, Q, layers, center)
            pstep_time.update(time.time() - end1)
            
            # Measure accuracy and record loss
            _, pred = output.topk(1, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            correct_1 = correct[:1].view(-1).float().sum(0)
            reduce_value(correct_1)
            correctes += correct_1

            loss = loss * input.size(0)
            reduce_value(loss)
            losses.update(loss.item() / batch_size.item(), batch_size.item())

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if args.local_rank == 0 and (i == num_batches-1) and i > 0:
                print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    f'Pstep {pstep_time.val:.3f} ({pstep_time.avg:.3f})\t'
                    f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                    f'Prec@1 {correct_1/batch_size*100:.3f} ({correctes/count*100:.3f})')

            if i == num_batches-1 and epoch == args.epochs - 1:
                validate(test_loader, model, criterion, device)
                model.train()

def get_state_dict(args, i, device='cpu'):
    if args.arch == 'clip-vit-b32':
        file = os.path.join(args.model_location, f'model_{i}.pt')
    else:
        file = os.path.join(args.model_location, f'{i}.pt')
    # print (file)
    return torch.load(file, map_location=device, weights_only=True)
    
if __name__ == '__main__':
    args = parse_arguments()
    print (args)
    set_seed(args.randomseed)
    # DDP initialize backend
    rank = args.local_rank
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl') 
    args.world_size = torch.distributed.get_world_size()
    device = torch.device("cuda", rank)
       
    train_loader, val_loader, test_loader  = get_datasets(args)

    model = get_model(args).to(device)
    
    layers = [0]
    for param in model.parameters():
        arr_shape = param.data.shape
        size = arr_shape.numel()
        # block-wise
        if layers[-1] > args.layer_size:
            layers.append(size)
        else:
            layers[-1] += size
    print (layers)

    models = [i for i in range(args.params_start, args.params_end)]
    NUM_MODELS = len(models)
    D = sum([p.data.shape.numel() for p in model.parameters()])
    print ('NUM_MODELS:', NUM_MODELS, 'models:', models)
    
    model = DDP(model, device_ids=[rank], output_device=rank \
                  , find_unused_parameters=True)
    
    criterion = nn.CrossEntropyLoss().to(device)
    center_P = 0
    for i in range(len(models)):
        state_dict = get_state_dict(args, models[i], device)
        try:
            model.module.load_state_dict(state_dict)
        except RuntimeError:
            model.load_state_dict(state_dict)
        center_P += get_model_param_vec_torch(model)
    
    center_P /= NUM_MODELS
    update_param(model, center_P)
    center = get_model_param_vec_torch(model)
    seq = [i for i in range(0, len(models))]
    print (seq)
    
    SWA = 0
    SWA_num = 0
    
    exp_name = f"{args.params_start}_{args.params_end}_"
    
    for models_epoch in range(args.models_epochs):
        # np.random.shuffle(models)
        print (models)
        st = 0
        print ('NUM_MODELS:', NUM_MODELS)
        while st < NUM_MODELS:
            ed = min(NUM_MODELS, st + args.model_batch)
            num_models_batch = ed - st
            models_per_gpu = num_models_batch // args.world_size
            remaining = num_models_batch % args.world_size
            parts = [models_per_gpu + (i < remaining) for i in range(args.world_size)]
            
            start, end = st + sum(parts[:rank]), st + sum(parts[:rank + 1])
            P = torch.zeros((parts[rank], D), device='cpu')
            
            # center = get_model_param_vec_torch(model)
            for i in range(start, end):
                state_dict = get_state_dict(args, models[seq[i]], device)
                # model.load_state_dict({k.replace('module.',''):v for k,v in state_dict.items()})
                try:
                    model.module.load_state_dict(state_dict)
                except RuntimeError:
                    model.load_state_dict(state_dict)
                P[i - start, :] = get_model_param_vec_torch(model)
            del state_dict
            
            update_param(model, center)
            P -= center_P.cpu()
            P = P.cpu()
                
            idx = 0
            layer_norm = []
            for size in layers:
                norm_layer = torch.norm(P[:, idx:idx+size], dim=1)
                layer_norm.append(norm_layer.cpu().numpy())
                P[:, idx:idx+size] /= norm_layer.reshape(-1, 1)
                idx += size
            layer_norm = torch.from_numpy(np.array(layer_norm))
            torch.save(layer_norm,  os.path.join(args.model_location, exp_name + 'layernorm.pt'))
            print ('rank:', rank, P.shape, P.dtype)
            
            idx = 0
            Q = []
            for size in layers:
                quantized_tensor, scale, zero_point = quantize_to_int8(P[:, idx:idx+size])
                Q.append([quantized_tensor.to(device), scale, zero_point])
                idx += size
            
            del P
            torch.cuda.empty_cache()
            
            torch.cuda.empty_cache()        
            dist.barrier() # Synchronizes all processes
            X = torch.zeros((parts[rank], len(layers))).to(device)
            X = Variable(X, requires_grad=True)
            X.sum().backward()
            print ('X:', X.shape)
            
            if args.optimizer == 'sgd':
                optimizer = torch.optim.SGD([X], lr=args.lr, momentum=0.9, weight_decay=args.wd)
            elif args.optimizer == 'adamw':
                optimizer = torch.optim.AdamW([X], lr=args.lr, weight_decay=args.wd)
            
            train_loader = val_loader
            print (len(train_loader))
            num_batches = len(train_loader)
            if args.local_rank == 0:
                print (optimizer)
                print ('num_batches:', num_batches)

            steps = num_batches * args.epochs
            lr_scheduler = cosine_lr(optimizer, args.lr, int(steps * 0.2), steps)
            train(model, optimizer, center, Q, train_loader, test_loader, args, device)
            
            
            torch.save(X.data, os.path.join(args.model_location, exp_name + 'X.pt'))
            
            SWA += get_model_param_vec_torch(model)
            SWA_num += 1
            print ('test SWA (not continued):------------------------------')
            center = get_model_param_vec_torch(model)
            update_param(model, SWA / SWA_num)
            validate(test_loader, model, criterion, device)
            
            del Q
            dist.barrier()
            torch.cuda.empty_cache()
            st = ed
