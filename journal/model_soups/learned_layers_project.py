import argparse
import os
import wget
import torch
import clip
import os
import json
import operator

import sys
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.autograd import Variable

from datasets import ImageNet2p, ImageNet, ImageNetSketch, ImageNetR, ObjectNet, ImageNetA, ImageNet2pShuffled
from utils import get_model_from_sd, test_model_on_dataset, maybe_dictionarize_batch

import time 
import torch.nn as nn
import torch.optim as optim

from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
    
def set_seed(seed=1): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('~/data'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--model-location",
        type=str,
        default=os.path.expanduser('~/ssd/checkpoints/soups'),
        help="Where to download the models.",
    )
    parser.add_argument(
        "--download-models", action="store_true", default=False,
    )
    parser.add_argument(
        "--eval-individual-models", action="store_true", default=False,
    )
    parser.add_argument(
        "--uniform-soup", action="store_true", default=False,
    )
    parser.add_argument(
        "--greedy-soup", action="store_true", default=False,
    )
    parser.add_argument(
        "--twa-soup", action="store_true", default=False,
    )
    parser.add_argument(
        "--plot", action="store_true", default=False,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )
    
    parser.add_argument('--epochs', default=5, type=int, metavar='N',
                        help='number of total epochs to run')

    return parser.parse_args()


def get_model_param_vec(model):
    """
    Return model parameters as a vector
    """
    vec = []
    for name,param in model.named_parameters():
        vec.append(param.detach().cpu().numpy().reshape(-1))
    return np.concatenate(vec, 0)

def get_model_grad_vec_torch(model):
    """
    Return model grad as a vector
    """
    vec = []
    for name,param in model.named_parameters():
        if param.grad is not None:
            vec.append(param.grad.clone().reshape(-1).cpu())
        else:
            vec.append(torch.zeros_like(param).reshape(-1).cpu())
    return torch.cat(vec, 0)

def update_grad(model, grad_vec):
    """
    Update model grad
    """
    idx = 0
    for name,param in model.named_parameters():
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
    for name,param in model.named_parameters():
        arr_shape = param.data.shape
        size = arr_shape.numel()
        param.data = param_vec[idx:idx+size].reshape(arr_shape).clone()
        idx += size

def P_SGD(model, optimizer, P, grad):
    # print (P.shape, grad.shape)
    gk = torch.mm(P, grad.reshape(-1,1))
    grad_proj = torch.mm(P.transpose(0, 1), gk)
    
    update_grad(model, grad_proj.reshape(-1).cuda())

    optimizer.step()

def P_Adam(model, optimizer, X, P, grad, layers):
    # print (P.shape, grad.shape)
    
    idx = 0
    for i, size in enumerate(layers):
        X.grad.data[:, i] = torch.mm(P[:,  idx:idx+size], grad[idx:idx+size].reshape(-1,1)).reshape(-1)
        idx += size
        
    Y = torch.nn.functional.softmax(X.data, dim=0)
    # print (Y.shape)
    # print (1 - Y[:, 0])
    # g = .clone()
    X.grad.data = X.grad.data * Y
    X.grad.data = X.grad.data - Y * X.grad.data.sum(0)
    # X.grad.data = X.grad.data * (Y * (1 - Y))
    # print ('grad:')
    # print (X.grad.data.shape)
    # print (X.grad.data)
    # print ('data:')
    # print (X.data)
    # sys.exit()
    
    optimizer.step()
    
    Y = torch.nn.functional.softmax(X.data, dim=0)
    param_proj = torch.zeros_like(grad)

    idx = 0
    for i, size in enumerate(layers):
        param_proj[idx:idx+size] = torch.mm(P[:, idx:idx+size].transpose(0, 1), Y.data[:, i].reshape(-1, 1)).reshape(-1)
        idx += size
    update_param(model, param_proj.cuda())

def get_imagenet_acc(model, test_dset):
    with torch.no_grad():
        correct = 0.
        n = 0
        end = time.time()
        for i, batch in enumerate(test_dset.test_loader):
            batch = maybe_dictionarize_batch(batch)
            inputs, labels = batch['images'].cuda(), batch['labels'].cuda()
            data_time = time.time() - end
            end = time.time()
            logits = model(inputs)
            loss = criterion(logits, labels)
            # pred = logits.argmax(dim=1, keepdim=True).to(device)
            pred = logits.argmax(dim=1, keepdim=True).cuda()
            y = labels
            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)

            batch_time = time.time() - end
            percent_complete = 100.0 * i / len(test_dset.test_loader)
            if ( i % 200 ) == 0:
                print(
                    f"Train Epoch: {0} [{percent_complete:.0f}% {i}/{len(test_dset.test_loader)}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )
            end = time.time()
        acc = correct / float(n)
        print('Top-1', acc)
    return acc

if __name__ == '__main__':
    args = parse_arguments()
    # NUM_MODELS = 72
    NUM_MODELS = 36
    INDIVIDUAL_MODEL_RESULTS_FILE = 'individual_model_results.jsonl'
    UNIFORM_SOUP_RESULTS_FILE = 'uniform_soup_results.jsonl'
    GREEDY_SOUP_RESULTS_FILE = 'greedy_soup_results.jsonl'
    TWA_SOUP_RESULTS_FILE = 'twa_soup_results.jsonl'
    
    set_seed(1)
    
    # Step 1: Download models.
    if args.download_models:
        if not os.path.exists(args.model_location):
            os.mkdir(args.model_location)
        for i in range(NUM_MODELS):
            print(f'\nDownloading model {i} of {NUM_MODELS - 1}')
            wget.download(
                f'https://github.com/mlfoundations/model-soups/releases/download/v0.0.2/model_{i}.pt',
                out=args.model_location
                )

    model_paths = [os.path.join(args.model_location, f'model_{i}.pt') for i in range(NUM_MODELS)]
    
    # Step 4: TWA Soup.
    if args.twa_soup:
        if os.path.exists(TWA_SOUP_RESULTS_FILE):
            os.remove(TWA_SOUP_RESULTS_FILE)
        
        # Get model names
        individual_model_db = pd.read_json(INDIVIDUAL_MODEL_RESULTS_FILE, lines=True)
        models = individual_model_db["model_name"]
        
        base_model, preprocess = clip.load('ViT-B/32', 'cpu', jit=False)
    
        print ('here!')

        held_out_val_set = ImageNet2p(preprocess, args.data_location, args.batch_size, args.workers)
        train_dset = ImageNet2pShuffled(preprocess, location=args.data_location, batch_size=args.batch_size, num_workers=args.workers)
        test_dset = ImageNet(preprocess, location=args.data_location, batch_size=args.batch_size, num_workers=args.workers)
        W = []
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        W = np.zeros((NUM_MODELS, 113961705))
        model = None
        for i in range(NUM_MODELS):
            file = os.path.join(args.model_location, f'{models[i]}.pt')
            print (file)
            state_dict = torch.load(file, map_location=device)
            if model is None:
                model = get_model_from_sd(state_dict, base_model)
            else:
                model.load_state_dict(state_dict)
            W[i, :] = get_model_param_vec(model)
        print (W.shape)

        P = torch.from_numpy(W).float()
        center = torch.from_numpy(W.mean(axis=0)).float().cuda()
        update_param(model, center)
        del W
        
        n_dim = P.shape[0]
        n_components = n_dim
        
        layers = [0]
        for param in model.parameters():
            arr_shape = param.data.shape
            size = arr_shape.numel()
            # layers.append(size)
            # block-wise
            if layers[-1] > 30000000:
                layers.append(size)
            else:
                layers[-1] += size
        print (layers)

        # Run one train epoch
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        # Define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().cuda()    
        count = len(layers)

        X = torch.ones((n_dim, count)) / n_dim
        X = Variable(X, requires_grad = True)
        X.sum().backward()
        optimizer = torch.optim.AdamW([X], lr=args.lr, weight_decay=0.0)
        print (optimizer)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        optimizer_model = optim.SGD(model.parameters(), lr=0)
        optimizer_model.zero_grad()

        # Switch to train mode
        model.train()

        end = time.time()
        # for i, (input, target) in enumerate(train_loader):
        epoch = 0
        num_batches = len(train_dset.train_loader)
        print ('num_batches:', num_batches)
        for epoch in range(args.epochs):
            print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))

            for i, batch in enumerate(train_dset.train_loader):

                # Measure data loading time
                data_time.update(time.time() - end)

                # Load batch data to cuda
                batch = maybe_dictionarize_batch(batch)
                input_var, target_var = batch['images'].cuda(), batch['labels'].cuda()
                # input_var = input_var.to(torch.float32)

                # Compute output
                output = model(input_var)
                loss = criterion(output, target_var)

                # Compute gradient and do SGD step
                optimizer_model.zero_grad()
                loss.backward()

                gk = get_model_grad_vec_torch(model)
                P_Adam(model, optimizer, X, P, gk, layers)
                # sys.exit()
                # optimizer.step()

                # Measure accuracy and record loss
                prec1 = accuracy(output.data, target_var)[0]
                losses.update(loss.item(), input_var.size(0))
                top1.update(prec1.item(), input_var.size(0))

                # Measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                
                if i % 50 == 0 or i == num_batches-1:
                    print('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                            epoch, i, len(train_dset.train_loader), batch_time=batch_time,
                            data_time=data_time, loss=losses, top1=top1))

                    if i == num_batches-1:
                        held_out_val_accuracy = test_model_on_dataset(model, held_out_val_set)
                        print ('held_out_val_accuracy:', held_out_val_accuracy)
                        acc = get_imagenet_acc(model, test_dset)
                        print('Accuracy is', 100 * acc)
            
            # lr_scheduler.step()
