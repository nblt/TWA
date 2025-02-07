import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import time 
import clip
import torch
import random
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


from torchvision.datasets import CIFAR10
from torchvision.transforms import *
from utils import get_model_from_sd_cpu

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC


# import psutil
# import pynvml
# pynvml.nvmlInit()

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
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--blocks", type=int, default=3)
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
    parser.add_argument("--local_rank", type=int)
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
    
def set_seed(seed=1): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def reduce_value(value, op=dist.ReduceOp.SUM):
    world_size = dist.get_world_size()
    if world_size < 2:  # single GPU
        return value
 
    with torch.no_grad():
        dist.all_reduce(value, op)
        return value

# def get_cuda_info():
#     handle = pynvml.nvmlDeviceGetHandleByIndex(0)   #gpu_id
#     meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
#     print(meminfo)

#     info = psutil.virtual_memory()
#     print(u'内存使用：',psutil.Process(os.getpid()).memory_info().rss)
#     print(u'总内存：',info.total)
#     print(u'内存占比：',info.percent)
#     print(u'cpu个数：',psutil.cpu_count())
                        
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
        
    
def validate(test_loader, model, criterion, device):
    # Run evaluation 

    batch_time = AverageMeter()
    losses = AverageMeter()
    correctes = 0
    count = 0
    world_size = dist.get_world_size()

    # Switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
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

            reduce_value(loss * input.size(0))
            losses.update(loss.item(), batch_size)

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

def P_Step(model, optimizer, X, P, layers, center):
    grad = get_model_grad_vec_torch(model)

    idx = 0
    for i, size in enumerate(layers):
        X.grad.data[:, i] = torch.mm(P[:,  idx:idx+size], grad[idx:idx+size].reshape(-1,1)).reshape(-1)
        idx += size

    optimizer.step()
    param_proj = torch.zeros_like(grad).to(grad)

    idx = 0
    for i, size in enumerate(layers):
        param_proj[idx:idx+size] = torch.mm(P[:, idx:idx+size].transpose(0, 1), X.data[:, i].reshape(-1, 1)).reshape(-1)
        idx += size
    
    reduce_value(param_proj)
    update_param(model, param_proj + center)

def get_dataset(args, preprocess=None):
    if args.datasets == 'CIFAR10':
        print ('cifar10 dataset!')
        # normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        # train_dataset = datasets.CIFAR10(root='./datasets/', train=True, transform=transforms.Compose([
        #         transforms.RandomHorizontalFlip(),
        #         transforms.RandomCrop(32, 4),
        #         transforms.ToTensor(),
        #         normalize,
        #     ]), download=True)

        # val_dataset = datasets.CIFAR10(root='./datasets/', train=False, transform=transforms.Compose([
        #         transforms.ToTensor(),
        #         normalize,
        #     ]))

        def _convert_image_to_rgb(image):
            return image.convert("RGB")


        def _transform(n_px):
            return Compose([
                Resize(n_px, interpolation=BICUBIC),
                CenterCrop(n_px),
                _convert_image_to_rgb,
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])

        transforms = _transform(224)

        idxs = np.load('cifar1098_idxs.npy').astype('int')
        indices = []
        for i in range(len(idxs)):
            if idxs[i]:
                indices.append(i)
        print (len(indices))
        val = CIFAR10(root='./data', train=True, download=True, transform=transforms)
        train_dataset = Subset(val, indices)
        val_dataset = CIFAR10(root='./data', train=False, download=True, transform=transforms)


    elif args.datasets == 'CIFAR100':
        print ('cifar100 dataset!')
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        train_dataset = datasets.CIFAR100(root='./datasets/', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True)

        val_dataset = datasets.CIFAR100(root='./datasets/', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
    
    elif args.datasets == 'ImageNet':
        traindir = os.path.join(args.data_location, 'ILSVRC2012_img_train')
        valdir = os.path.join(args.data_location, 'ILSVRC2012_img_val')
        import torchvision.transforms as transforms
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(traindir, preprocess)
        val_dataset = datasets.ImageFolder(valdir, preprocess)
        
        ### train on validation set
        idx_file = 'imagenet_98_idxs.npy'
        assert os.path.exists(idx_file)
        with open(idx_file, 'rb') as f:
            idxs = np.load(f)

        idxs = idxs.astype('int')
        idxs = np.where(idxs)[0]
        if args.local_rank == 0:
            print (idxs, len(idxs))
            print('shuffling val set.')
            print (len(idxs))

        train_dataset = Subset(train_dataset, idxs)
        
    batch_size_per_GPU = args.batch_size // world_size

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size_per_GPU, sampler=train_sampler,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size_per_GPU, sampler=val_sampler,
        num_workers=args.workers)

    return train_loader, val_loader
    
def train(model, optimier, center, P, train_loader, test_loader, args, device):
    # criterion = nn.CrossEntropyLoss().to(device)
    if args.labelsmooth:
        print ('label smooth: 0.1')
        criterion = LabelSmoothCELoss().to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)    

    # # validate swa performance
    # bn_update(train_loader, model)
    validate(test_loader, model, criterion, device)

    # if args.local_rank == 0: get_cuda_info()

    end = time.time()
    _step = 0
    for epoch in range(args.epochs):
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train_loader.sampler.set_epoch(epoch)
        count = 0
        correctes = 0

        # Switch to train mode
        model.train()
        for i, (input, target) in enumerate(train_loader):
            target = target.to(device)
            input = input.to(device)

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

            P_Step(model, optimizer, X, P, layers, center)
            
            # Measure accuracy and record loss
            _, pred = output.topk(1, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            correct_1 = correct[:1].view(-1).float().sum(0)
            reduce_value(correct_1)
            correctes += correct_1

            reduce_value(loss * input.size(0))
            losses.update(loss.item(), batch_size)

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if args.local_rank == 0 and (i % args.print_freq == 0 or i == num_batches-1):
                print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                    f'Prec@1 {correct_1/batch_size*100:.3f} ({correctes/count*100:.3f})')

        validate(test_loader, model, criterion, device)

def get_state_dict(args, models, i, device='cpu'):
    file = os.path.join(args.model_location, f'{models[i]}.pt')
    print (file)
    return torch.load(file, map_location=device)
    
if __name__ == '__main__':
    args = parse_arguments()
    print (args)
    set_seed(args.randomseed)
    NUM_MODELS = args.params_end - args.params_start
    print ('NUM_MODELS:', NUM_MODELS)
    
    # DDP initialize backend
    rank = args.local_rank
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    world_size = torch.distributed.get_world_size()
    device = torch.device("cuda", rank)

    # model = models.__dict__[args.arch]()
    # model = get_model(args).to(device)
    INDIVIDUAL_MODEL_RESULTS_FILE = 'individual_model_results.jsonl'
    individual_model_db = pd.read_json(INDIVIDUAL_MODEL_RESULTS_FILE, lines=True)
    models = individual_model_db["model_name"]
    base_model, preprocess = clip.load('ViT-B/32', 'cpu', jit=False)
    model = get_model_from_sd_cpu(get_state_dict(args, models, 0), base_model).to(device)

    # Get model names
    train_loader, test_loader  = get_dataset(args, preprocess)
    
    layer_sizes = [p.data.shape.numel() for p in model.parameters()]
    args.layer_size = sum(layer_sizes) // args.blocks
    # args.layer_size = (sum(layer_sizes) - layer_sizes[-1]) // (args.blocks - 1)
    layers = [0]
    # for size in layer_sizes[:-1]:
    for size in layer_sizes:
        if layers[-1] + size > args.layer_size:
            layers.append(size)
        else:
            layers[-1] += size
    # layers.append(layer_sizes[-1])
    print (layers)

    models_per_gpu = NUM_MODELS // world_size
    remaining = NUM_MODELS % world_size
    parts = [models_per_gpu + (i < remaining) for i in range(world_size)]
    
    start, end = sum(parts[:rank]) + args.params_start, sum(parts[:rank + 1]) + args.params_start
    P = torch.zeros((parts[rank], sum(layer_sizes)), device=device)
    for i in range(start, end):
        state_dict = get_state_dict(args, models, i, device)
        model.load_state_dict({k.replace('module.',''):v for k,v in state_dict.items()})
        # model.load_state_dict(state_dict)
        P[i - start, :] = get_model_param_vec_torch(model)

    center = P.sum(axis=0)
    reduce_value(center)
    center /= NUM_MODELS

    P -= center
    idx = 0
    for size in layers:
        P[:, idx:idx+size] /= torch.norm(P[:, idx:idx+size], dim=1).reshape(-1, 1)
        idx += size
    print ('rank:', rank, P.shape, P.dtype)
    # print (P, center)
    
    torch.cuda.empty_cache()        
    dist.barrier() # Synchronizes all processes

    update_param(model, center)
    model = DDP(model, device_ids=[rank], output_device=rank \
                  , find_unused_parameters=True)

    # Run one train epoch
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    X = torch.zeros((parts[rank], len(layers))).to(device)
    X = Variable(X, requires_grad=True)
    X.sum().backward()
    
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD([X], lr=args.lr, momentum=0.9, weight_decay=args.wd)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW([X], lr=args.lr, weight_decay=args.wd)
        
    num_batches = len(train_loader)
    if args.local_rank == 0:
        print (optimizer)
        print ('num_batches:', num_batches)

    steps = num_batches * args.epochs
    lr_scheduler = cosine_lr(optimizer, args.lr, int(steps * 0.2), steps)
    # lr_scheduler = cosine_lr(optimizer, args.lr, int(steps * 0), steps)
    train(model, optimizer, center, P, train_loader, test_loader, args, device)


