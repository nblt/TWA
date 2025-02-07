import yaml
import os
import re
import argparse
import time
import torch
import utils
from torch.autograd import Variable
import numpy as np
import youtokentome as yttm
from tqdm import tqdm
import argparse
from models import *
import _data as data
from utils import *
import glob
import shutil
import logging
from torch.utils.tensorboard import SummaryWriter


UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

def get_model_param_vec_torch(model):
    """
    Return model grad as a vector
    """
    vec = []
    for param in model.parameters():
        vec.append(param.data.detach().reshape(-1))
    return torch.cat(vec, 0)

def get_model_grad_vec_torch(model):
    """
    Return model grad as a vector
    """
    vec = []
    for param in model.parameters():
        if param.grad is not None:
            vec.append(param.grad.detach().reshape(-1))
        else:
            vec.append(torch.zeros_like(param).reshape(-1).to(param))
    return torch.cat(vec, 0)

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

def create_path(path):
	if os.path.isdir(path) is False:
		os.makedirs(path)
	else:
		if len(glob.glob(f"{path}/*.log", recursive=True)) > 0:
			file = glob.glob(f"{path}/*.log", recursive=True)[0]
			with open(file, 'r') as f:
				text = f.read()
			if "Step: 99999/100000" in text:
				print("File exists")
				quit()
			else:
				shutil.rmtree(path)
				os.makedirs(path)
				print("Removing old files")
		else:
			shutil.rmtree(path)
			os.makedirs(path)
			print("Removing old files")
				
	return 

def P_Step(model, optimizer, X, P, layers, center):
	grad = get_model_grad_vec_torch(model)
	
	with torch.no_grad():
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

		update_param(model, param_proj + center)

def get_state_dict(args, i, device='cpu'):
    file = os.path.join(args.cp_dir, f'{i}.pt')
    print (file)
    return torch.load(file, map_location=device)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	# optim config
	parser.add_argument("--ts", default=100000, type=int, help="Total number of epochs.")
	parser.add_argument("--lr", default=0.0005, type=float, help="Base learning rate at the start of the training.")
	parser.add_argument("--wd", default=0.0001, type=float, help="L2 weight decay.")
	parser.add_argument("--bs", default=256, type=int, help="Batch size used in the training and validation loop.")
	parser.add_argument("--dp", default=0.1, type=float)

	# twa
	parser.add_argument("--layer-size", type=int, default=0)
	parser.add_argument('--epochs', default=5, type=int, help='number of total epochs to run')
	parser.add_argument('--labelsmooth', dest='labelsmooth', action='store_true', help='use label smooth (0.1)')
	parser.add_argument('--params_start', default=0, type=int, help='which idx start for TWA') 
	parser.add_argument('--params_end', default=51, type=int, help='which idx end for TWA')

	# seed
	parser.add_argument("--seed", default=123, type=int, help="seed")
	parser.add_argument("--print_freq", type=int, default=1000, help="print frequency")

	args = parser.parse_args()

	initialize(args.seed)

	# initialize directory
	args.cp_dir = f"checkpoints/adam/run_ms_{args.seed}/"
	# create_path(args.cp_dir)
	# for file in glob.glob("**/*.py", recursive=True):
	# 	if "checkpoints" in file or "data" in file or "results" in file:
	# 		continue
	# 	os.makedirs(os.path.dirname(f"{args.cp_dir}/codes/{file}"), exist_ok=True)
	# 	shutil.copy(file, f"{args.cp_dir}/codes/{file}")

	# initialize logging
	train_log = os.path.join(args.cp_dir, time.strftime("%Y%m%d-%H%M%S") + '.log')
	logging.basicConfig(
		format="%(name)s: %(message)s",
		level="INFO",
		handlers=[
			logging.FileHandler(train_log),
			logging.StreamHandler()
		]
	)

	writer = SummaryWriter(log_dir=args.cp_dir)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	bpe_model = yttm.BPE(model="data/bpe.32000.model")
	logging.info(f"Vocab length: {bpe_model.vocab_size()}")

	model = Seq2SeqTransformer(
		num_encoder_layers=3,
		num_decoder_layers=3,
		emb_size=512,
		nhead=8,
		vocab_size=bpe_model.vocab_size(),
		dim_feedforward=512,
		dropout=args.dp
	)
	# model = model.to(device)
	model = model.cuda()

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

	D = sum(layers)
	N = args.params_end - args.params_start
	P = torch.zeros((N, D), device=device)
    
	for i in range(args.params_start, args.params_end):
		state_dict = get_state_dict(args, i, device)
		model.load_state_dict({k.replace('module.',''):v for k,v in state_dict.items()})
		P[i - args.params_start, :] = get_model_param_vec_torch(model)
	del state_dict
	
	center = P.sum(axis=0)
	center /= N

	P -= center
	idx = 0
	for size in layers:
		P[:, idx:idx+size] /= torch.norm(P[:, idx:idx+size], dim=1).reshape(-1, 1)
		idx += size
  
	torch.cuda.empty_cache()
	update_param(model, center)

	X = torch.zeros((N, len(layers))).to(device)
	print ('X:', X.shape)
	X = Variable(X, requires_grad=True)
	X.sum().backward()
	optimizer = torch.optim.Adam([X], lr=args.lr, weight_decay=args.wd)

	criterion = torch.nn.CrossEntropyLoss(ignore_index=1)
	optimizer_base = torch.optim.Adam(
		model.parameters(),
		lr=0,
		betas=(0.9, 0.999), 
		eps=1e-08,
		weight_decay=args.wd,
		amsgrad=False)
	# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
	# 													   mode='min',
	# 													   factor=0.5,
	# 													   patience=1,
	# 													   threshold=0.01,
	# 													   min_lr=1e-6)

	# Create dataset
	dset_loaders = {
		'train': data.load("data/",
						   split='train',
						   batch_size=args.bs,
						   bpe_model=bpe_model,
						   workers=8),
		'val': data.load("data/",
					     split='dev',
					     batch_size=args.bs,
					     shuffle=False,
					     bpe_model=bpe_model),
		'test': data.load("data/",
					     split='test',
					     batch_size=1,
					     shuffle=False,
					     bpe_model=bpe_model)
	} 

	best_loss = float('inf')
	step = 0
	epoch = 0
	while (step < args.ts):
		trainloss = AverageMeter()
		model.train()
		
		for batch_idx, batch in enumerate(dset_loaders['train']):
			inputs, targets = (b.to(device) for b in batch)
			targets_input = targets[:-1, :]
			
			optimizer_base.zero_grad()

			inputs_mask, targets_mask, src_padding_mask, tgt_padding_mask = create_mask(inputs, targets_input, device)
			outputs = model(inputs, targets_input, inputs_mask, targets_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

			targets_out = targets[1:, :]
			batch_loss = criterion(outputs.reshape(-1, outputs.shape[-1]), targets_out.reshape(-1))
			batch_loss.backward()

			P_Step(model, optimizer, X, P, layers, center)

			trainloss.update(batch_loss.item(), inputs.shape[1])

			if batch_idx % args.print_freq == 0:
				logging.info('Batch: {0}/{1}, Train_Loss = {2}'.format(batch_idx, len(dset_loaders['train']), trainloss.avg))

			if step % 1000 == 0:
				torch.save(model.state_dict(), f"{args.cp_dir}/"+str(step // args.print_freq)+'.pt')
				
			step+=1
			if (step >= args.ts):
				break

			# break
			
		writer.add_scalar('Train/train_loss', trainloss.avg, step)
		writer.add_scalar('Params/lr', optimizer.param_groups[0]['lr'], step)
		
		# inputs = "Dies ist eines der wichtigsten Skigebiete in den Pyrenäen und auf der Halbinsel ."
		# inputs = translate(model, inputs, bpe_model, device)
		# logging.info(f"Translated Sentence \n {inputs}")

		model.eval()
		valloss = AverageMeter()

		with torch.no_grad():
			for batch_idx, batch in enumerate(dset_loaders['val']):
				inputs, targets = (b.to(device) for b in batch)
				targets_input = targets[:-1, :]

				inputs_mask, targets_mask, src_padding_mask, tgt_padding_mask = create_mask(inputs, targets_input, device)
				outputs = model(inputs, targets_input, inputs_mask, targets_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
				
				targets_out = targets[1:, :]
				batch_loss = criterion(outputs.reshape(-1, outputs.shape[-1]), targets_out.reshape(-1))
			
				valloss.update(batch_loss.item(), inputs.shape[1])
					
				if batch_idx % args.print_freq == 0:
					logging.info('Batch: {0}/{1}, Val_Loss = {2}'.format(batch_idx, len(dset_loaders['val']), valloss.avg))

		writer.add_scalar('Val/val_loss', valloss.avg, step)
		logging.info('Step: {0}/{1}, Train_Loss = {2}, Val_Loss = {3}'.format(step, args.ts, trainloss.avg, valloss.avg))

		if valloss.avg < best_loss:
			torch.save(model.state_dict(), f"{args.cp_dir}/twa_best_model.pth.tar")
			best_loss = valloss.avg
		
		# scheduler.step(valloss.avg)
		# epoch+=1
		# if epoch % 2 == 0:
  

	# with torch.no_grad():
	# 	score = compute_bleu(model, dset_loaders['val'], bpe_model, device)
	# # writer.add_scalar('val/bleu_score', score,  step)
	# logging.info('Bleu_score = {0}'.format(score))
  
	with torch.no_grad():
		score = compute_bleu(model, dset_loaders['test'], bpe_model, device)
	# writer.add_scalar('Test/bleu_score', score,  step)
	logging.info('Bleu_score = {0}'.format(score))

		