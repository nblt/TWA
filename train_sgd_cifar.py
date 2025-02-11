import argparse
import os
import time
import numpy as np
import random
import sys
from timm.data import Mixup
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torchvision.transforms import v2

from utils import get_datasets, get_model, adjust_learning_rate, set_seed, Logger

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Parse arguments
parser = argparse.ArgumentParser(description='Regular SGD training')
parser.add_argument('--EXP', metavar='EXP', help='experiment name', default='SGD')
parser.add_argument('--arch', '-a', metavar='ARCH',
                    help='The architecture of the model')
parser.add_argument('--datasets', metavar='DATASETS', default='CIFAR10', type=str,
                    help='The training datasets')
parser.add_argument('--optimizer',  metavar='OPTIMIZER', default='sgd', type=str,
                    help='The optimizer for training')
parser.add_argument('--schedule',  metavar='SCHEDULE', default='step', type=str,
                    help='The schedule for training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50 iterations)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--wandb', dest='wandb', action='store_true',
                    help='use wandb to monitor statisitcs')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--log-dir', dest='log_dir',
                    help='The directory used to save the log',
                    default='save_temp', type=str)
parser.add_argument('--log-name', dest='log_name',
                    help='The log file name',
                    default='log', type=str)
parser.add_argument('--randomseed', 
                    help='Randomseed for training and initialization',
                    type=int, default=1)
parser.add_argument('--split', dest='split', action='store_true',
                    help='use split dataset')
parser.add_argument('--val_ratio', default=0, type=int, metavar='R',
                    help='val ratio for training')
parser.add_argument('--randaug', dest='randaug', action='store_true',
                    help='use randaug data augmentation')
parser.add_argument('--randn',  default=2, type=int,
                    metavar='N', help='random augmentation n')
parser.add_argument('--randm',  default=9, type=int,
                    metavar='M', help='random augmentation m')

parser.add_argument('--img_size', type=int, default=224, help="Resolution size")
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')
parser.add_argument("--local-rank", default=-1, type=int)

parser.add_argument('--ddp', dest='ddp', action='store_true',
                    help='ddp training')
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='finetune training')
parser.add_argument('--label_smooth', type=float, default=0.0, 
                        help='Label smoothing rate (default: 0.)')

parser.add_argument("--gradient_acc", type=int, default=1)
parser.add_argument('--mixup', type=float, default=0, help='use mixup data augmentation')
            
best_prec1 = 0


# Record training statistics
train_loss = []
train_err = []
test_loss = []
test_err = []
arr_time = []

p0 = None

args = parser.parse_args()


mixup_args = {
    'mixup_alpha' : args.mixup,
    'cutmix_alpha' : 0,
    'prob' : 1,
    'switch_prob' : 0,
    'mode': 'batch',
    'label_smoothing': args.label_smooth,
    'num_classes' : 1000
}
# print (mixup_args)

mixup_fn = Mixup(**mixup_args)

if args.wandb:
    import wandb
    wandb.init(project="TWA", entity="XXX")
    date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
    wandb.run.name = args.EXP + date


def get_model_param_vec(model):
    # Return the model parameters as a vector

    vec = []
    for name,param in model.named_parameters():
        vec.append(param.data.detach().reshape(-1))
    return torch.cat(vec, 0)


def get_model_grad_vec(model):
    # Return the model gradient as a vector

    vec = []
    for name,param in model.named_parameters():
        vec.append(param.grad.detach().reshape(-1))
    return torch.cat(vec, 0)

def update_grad(model, grad_vec):
    idx = 0
    for name,param in model.named_parameters():
        arr_shape = param.grad.shape
        size = arr_shape.numel()
        param.grad.data = grad_vec[idx:idx+size].reshape(arr_shape)
        idx += size

def update_param(model, param_vec):
    idx = 0
    for name,param in model.named_parameters():
        arr_shape = param.data.shape
        size = arr_shape.numel()
        param.data = param_vec[idx:idx+size].reshape(arr_shape)
        idx += size

sample_idx = 0

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

def step_lr(optimizer, base_lrs, warmup_length, steps, milestones, factor=0.1):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)
    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                lr = base_lr
                for _ in milestones:
                    if step >= _:
                        lr = lr * factor
                    else:
                        break
            assign_learning_rate(param_group, lr)
    return _lr_adjuster

def main():

    global args, best_prec1, p0, sample_idx
    global param_avg, train_loss, train_err, test_loss, test_err, arr_time, running_weight
    
    set_seed(args.randomseed)

    # Check the save_dir exists or not
    print ('save dir:', args.save_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Check the log_dir exists or not
    print ('log dir:', args.log_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    sys.stdout = Logger(os.path.join(args.log_dir, args.log_name))

    # Define model
    if args.ddp:
        rank = args.local_rank
        torch.cuda.set_device(rank)
        dist.init_process_group(backend='nccl') 
        args.world_size = torch.distributed.get_world_size()
        device = torch.device("cuda", rank)
        print (device)
        model = get_model(args).to(device)
        model = torch.nn.parallel.DistributedDataParallel(model\
                  , find_unused_parameters=True)
    else:
        device = torch.device("cuda")
        model = get_model(args).to(device)

    # Optionally resume from a checkpoint
    if args.resume:
        # if os.path.isfile(args.resume):
        if os.path.isfile(os.path.join(args.save_dir, args.resume)):
            
            # model.load_state_dict(torch.load(os.path.join(args.save_dir, args.resume)))

            print ("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            print ('from ', args.start_epoch)
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print ("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print ("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True


    # Prepare Dataloader
    # if args.split:
    #     train_loader, _, val_loader = get_datasets_split(args)
    # else:
    #     train_loader, val_loader = get_datasets(args)
        
    train_loader, _, val_loader = get_datasets(args)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    if args.half:
        model.half()
        criterion.half()
    
    print ('optimizer:', args.optimizer)

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, 
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, 
                                    weight_decay=args.weight_decay)
    print (optimizer)
    if args.schedule == 'step':
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], last_epoch=args.start_epoch - 1)
        
        num_baches = len(train_loader)
        print ('warn up: {} total: {}'.format(8*num_baches, num_baches*args.epochs))
        if args.datasets == 'ImageNet':
            lr_scheduler = step_lr(optimizer, args.lr, 8*num_baches, num_baches*args.epochs, [num_baches*30, num_baches*60])
        else:
            lr_scheduler = step_lr(optimizer, args.lr, 8*num_baches, num_baches*args.epochs, [num_baches*100, num_baches*150])

    elif args.schedule == 'cosine':
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        num_baches = len(train_loader)
        print ('warn up: {} total: {}'.format(8 * num_baches, num_baches * args.epochs))
        lr_scheduler = cosine_lr(optimizer, args.lr, 8 * num_baches, num_baches * args.epochs)
        
    print (lr_scheduler)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    is_best = 0
    print ('Start training: ', args.start_epoch, '->', args.epochs)

    if not args.ddp or args.ddp and args.local_rank == 0:
        # DLDR sampling
        torch.save(model.state_dict(), os.path.join(args.save_dir,  str(0) +  '.pt'))
    
    cnt = 0
    param_sum = 0
    p0 = get_model_param_vec(model)
    args.steps = 0
    lr_scheduler(0)

    for epoch in range(args.start_epoch, args.epochs):
        if args.ddp:
            train_loader.sampler.set_epoch(epoch)

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, lr_scheduler, epoch, device)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, device)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        
        if not args.ddp or args.ddp and args.local_rank == 0:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'model.th'))

            # DLDR sampling
            sample_idx += 1
            torch.save(model.state_dict(), os.path.join(args.save_dir,  str(sample_idx) +  '.pt'))

    p1 = get_model_param_vec(model)
    torch.save(p1 - p0, "delta.pt")
    
    print ('train loss: ', train_loss)
    print ('train err: ', train_err)
    print ('test loss: ', test_loss)
    print ('test err: ', test_err)

    print ('time: ', arr_time)

running_weight = None
iters = 0

def train(train_loader, model, criterion, optimizer, lr_scheduler, epoch, device):
    """
    Run one train epoch
    """
    global train_loss, train_err, arr_time, p0, sample_idx, iters
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()    
    
    total_loss, total_err = 0, 0
    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)
        
        lr_scheduler(args.steps)
        args.steps += 1

        target = target.to(device)
        input_var = input.to(device)
        target_var = target
        if args.half:
            input_var = input_var.half()
            
        if args.mixup > 0:
            input_var, target = mixup_fn(input_var, target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        loss.backward()
        total_loss += loss.item() * input_var.shape[0]
        total_err += (output.max(dim=1)[1] != target_var).sum().item()
        if 'vit' in args.arch:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        iters += 1
        if iters % args.gradient_acc == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            
        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (not args.ddp or args.ddp and args.local_rank == 0) and i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))

    print ('Total time for epoch [{0}] : {1:.3f}'.format(epoch, batch_time.sum))

    train_loss.append(total_loss / len(train_loader.dataset))
    train_err.append(total_err / len(train_loader.dataset)) 
    if args.wandb:
        wandb.log({"train loss": total_loss / len(train_loader.dataset)})
        wandb.log({"train acc": 1 - total_err / len(train_loader.dataset)})
    
    arr_time.append(batch_time.sum)

def reduce_value(value, op=dist.ReduceOp.SUM):
    if not args.ddp or dist.get_world_size() < 2:
        return value
    args.world_size = dist.get_world_size()
 
    with torch.no_grad():
        dist.all_reduce(value, op)
        return value

def validate(test_loader, model, criterion, device):
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

            if (not args.ddp or args.ddp and args.local_rank == 0) and i % args.print_freq == 0:
                print(f'Test: [{i}/{len(test_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Prec@1 {correct_1/batch_size*100:.3f} ({correctes/count*100:.3f})')

    print(f' * Prec@1 {correctes/count*100:.3f}')

    return correctes/count*100

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
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
    """Computes the precision@k for the specified values of k"""
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
