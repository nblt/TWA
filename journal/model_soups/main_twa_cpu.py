import argparse
import os
import wget
import torch
import clip
import os
import json
import operator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    NUM_MODELS = 72
    INDIVIDUAL_MODEL_RESULTS_FILE = 'individual_model_results.jsonl'
    UNIFORM_SOUP_RESULTS_FILE = 'uniform_soup_results.jsonl'
    GREEDY_SOUP_RESULTS_FILE = 'greedy_soup_results.jsonl'
    TWA_SOUP_RESULTS_FILE = 'twa_soup_results.jsonl'
    
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

    # # Step 2: Evaluate individual models.
    # if args.eval_individual_models or args.uniform_soup or args.greedy_soup:
    #     base_model, preprocess = clip.load('ViT-B/32', 'cpu', jit=False)

    if args.eval_individual_models:
        if os.path.exists(INDIVIDUAL_MODEL_RESULTS_FILE):
            os.remove(INDIVIDUAL_MODEL_RESULTS_FILE)
        for j, model_path in enumerate(model_paths):
            assert os.path.exists(model_path)
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            model = get_model_from_sd(state_dict, base_model)

            results = {'model_name' : f'model_{j}'}
            # Note: ImageNet2p is the held-out minival set from ImageNet train that we use.
            # It is called 2p for 2 percent of ImageNet, or 26k images.
            # See utils on how this dataset is handled slightly differently.
            for dataset_cls in [ImageNet2p, ImageNet, ImageNetV2, ImageNetSketch, ImageNetR, ObjectNet, ImageNetA]:

                print(f'Evaluating model {j} of {NUM_MODELS - 1} on {dataset_cls.__name__}.')

                dataset = dataset_cls(preprocess, args.data_location, args.batch_size, args.workers)
                accuracy = test_model_on_dataset(model, dataset)
                results[dataset_cls.__name__] = accuracy
                print(accuracy)

            with open(INDIVIDUAL_MODEL_RESULTS_FILE, 'a+') as f:
                f.write(json.dumps(results) + '\n')

    # Step 3: Uniform Soup.
    if args.uniform_soup:
        if os.path.exists(UNIFORM_SOUP_RESULTS_FILE):
            os.remove(UNIFORM_SOUP_RESULTS_FILE)

        # create the uniform soup sequentially to not overload memory
        for j, model_path in enumerate(model_paths):

            print(f'Adding model {j} of {NUM_MODELS - 1} to uniform soup.')

            assert os.path.exists(model_path)
            state_dict = torch.load(model_path)
            if j == 0:
                uniform_soup = {k : v * (1./NUM_MODELS) for k, v in state_dict.items()}
            else:
                uniform_soup = {k : v * (1./NUM_MODELS) + uniform_soup[k] for k, v in state_dict.items()}

        model = get_model_from_sd(uniform_soup, base_model)

        results = {'model_name' : f'uniform_soup'}
        for dataset_cls in [ImageNet2p, ImageNet, ImageNetV2, ImageNetSketch, ImageNetR, ObjectNet, ImageNetA]:

            print(f'Evaluating on {dataset_cls.__name__}.')

            dataset = dataset_cls(preprocess, args.data_location, args.batch_size, args.workers)
            accuracy = test_model_on_dataset(model, dataset)
            results[dataset_cls.__name__] = accuracy
            print(accuracy)
       
        with open(UNIFORM_SOUP_RESULTS_FILE, 'a+') as f:
            f.write(json.dumps(results) + '\n')

    
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
        # device = 'cpu'
        # NUM_MODELS = 5
        W = np.zeros((NUM_MODELS, 113961705))
        model = None
        for i in range(NUM_MODELS):
        # for i in range(40, 72):
            file = os.path.join(args.model_location, f'{models[i]}.pt')
            print (file)
            state_dict = torch.load(file, map_location=device)
            if model is None:
                model = get_model_from_sd(state_dict, base_model)
            else:
                model.load_state_dict(state_dict)
            W[i, :] = get_model_param_vec(model)
        # W = np.array(W)
        print (W.shape)

        P = torch.from_numpy(W).float()
        center = torch.from_numpy(W.mean(axis=0)).float().cuda()
        update_param(model, center)
        del W
        
        n_dim = P.shape[0]
        n_components = n_dim
        for i in range(n_dim):
            if i > 0:
                tmp = torch.mm(P[:i, :], P[i].reshape(-1, 1))
                P[i] -= torch.mm(P[:i, :].T, tmp).reshape(-1)
            tmp = torch.norm(P[i])
            P[i] /= tmp

        # Run one train epoch
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        
        # Define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().cuda()    
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0000)
        # model.cuda()
        
        # Switch to train mode
        model.train()

        end = time.time()
        # for i, (input, target) in enumerate(train_loader):
        epoch = 0
        num_batches = len(train_dset.train_loader)
        print ('num_batches:', num_batches)
        for epochs in range(5):
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
                optimizer.zero_grad()
                loss.backward()

                gk = get_model_grad_vec_torch(model)
                P_SGD(model, optimizer, P, gk)
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


        # # Now, iterate through all models and consider adding them to the greedy soup.
        # for i in range(1, NUM_MODELS):
        #     print(f'Testing model {i} of {NUM_MODELS}')

        #     # Get the potential greedy soup, which consists of the greedy soup with the new model added.
        #     new_ingredient_params = torch.load(os.path.join(args.model_location, f'{sorted_models[i]}.pt'))
        #     num_ingredients = len(greedy_soup_ingredients)
        #     potential_greedy_soup_params = {
        #         k : greedy_soup_params[k].clone() * (num_ingredients / (num_ingredients + 1.)) + 
        #             new_ingredient_params[k].clone() * (1. / (num_ingredients + 1))
        #         for k in new_ingredient_params
        #     }

        #     # Run the potential greedy soup on the held-out val set.
        #     model = get_model_from_sd(potential_greedy_soup_params, base_model)
        #     held_out_val_accuracy = test_model_on_dataset(model, held_out_val_set)

        #     # If accuracy on the held-out val set increases, add the new model to the greedy soup.
        #     print(f'Potential greedy soup val acc {held_out_val_accuracy}, best so far {best_val_acc_so_far}.')
        #     if held_out_val_accuracy > best_val_acc_so_far:
        #         greedy_soup_ingredients.append(sorted_models[i])
        #         best_val_acc_so_far = held_out_val_accuracy
        #         greedy_soup_params = potential_greedy_soup_params
        #         print(f'Adding to soup. New soup is {greedy_soup_ingredients}')

        # # Finally, evaluate the greedy soup.
        # model = get_model_from_sd(greedy_soup_params, base_model)
        # results = {'model_name' : f'greedy_soup'}
        # for dataset_cls in [ImageNet2p, ImageNet, ImageNetV2, ImageNetSketch, ImageNetR, ObjectNet, ImageNetA]:
        #     print(f'Evaluating on {dataset_cls.__name__}.')
        #     dataset = dataset_cls(preprocess, args.data_location, args.batch_size, args.workers)
        #     accuracy = test_model_on_dataset(model, dataset)
        #     results[dataset_cls.__name__] = accuracy
        #     print(accuracy)

        # with open(GREEDY_SOUP_RESULTS_FILE, 'a+') as f:
        #     f.write(json.dumps(results) + '\n')

    # # Step 5: Plot.
    # if args.plot:
    #     individual_model_db = pd.read_json(INDIVIDUAL_MODEL_RESULTS_FILE, lines=True)
    #     individual_model_db['OOD'] = 1./5 * (individual_model_db['ImageNetV2'] + 
    #         individual_model_db['ImageNetR'] + individual_model_db['ImageNetSketch'] + 
    #         individual_model_db['ObjectNet'] + individual_model_db['ImageNetA'])
    #     uniform_soup_db = pd.read_json(UNIFORM_SOUP_RESULTS_FILE, lines=True)
    #     uniform_soup_db['OOD'] = 1./5 * (uniform_soup_db['ImageNetV2'] + 
    #         uniform_soup_db['ImageNetR'] + uniform_soup_db['ImageNetSketch'] + 
    #         uniform_soup_db['ObjectNet'] + uniform_soup_db['ImageNetA'])
    #     greedy_soup_db = pd.read_json(GREEDY_SOUP_RESULTS_FILE, lines=True)
    #     greedy_soup_db['OOD'] = 1./5 * (greedy_soup_db['ImageNetV2'] + 
    #         greedy_soup_db['ImageNetR'] + greedy_soup_db['ImageNetSketch'] + 
    #         greedy_soup_db['ObjectNet'] + greedy_soup_db['ImageNetA'])

    #     fig = plt.figure(constrained_layout=True, figsize=(8, 6))
    #     ax = fig.subplots()

    #     ax.scatter(
    #         greedy_soup_db['ImageNet'], 
    #         greedy_soup_db['OOD'], 
    #         marker='*', 
    #         color='C4',
    #         s=400,
    #         label='Greedy Soup',
    #         zorder=10
    #     )

    #     ax.scatter(
    #         uniform_soup_db['ImageNet'], 
    #         uniform_soup_db['OOD'], 
    #         marker='o', 
    #         color='C0',
    #         s=200,
    #         label='Uniform Soup',
    #         zorder=10
    #     )

    #     ax.scatter(
    #         individual_model_db['ImageNet'].values[0], 
    #         individual_model_db['OOD'].values[0], 
    #         marker='h', 
    #         color='slategray',
    #         s=150,
    #         label='Initialization (LP)',
    #         zorder=10
    #     )

    #     ax.scatter(
    #         individual_model_db['ImageNet'].values[1:], 
    #         individual_model_db['OOD'].values[1:], 
    #         marker='d', 
    #         color='C2',
    #         s=130,
    #         label='Various hyperparameters',
    #         zorder=10
    #     )

    #     ax.set_ylabel('Avg. accuracy on 5 distribution shifts', fontsize=16)
    #     ax.set_xlabel('ImageNet Accuracy (top-1%)', fontsize=16)
    #     ax.grid()
    #     ax.legend(fontsize=13)
    #     plt.savefig('figure.png', bbox_inches='tight')