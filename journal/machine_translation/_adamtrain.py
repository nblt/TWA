import yaml
import os
import re
import argparse
import time
import torch
import utils
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


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	# optim config
	parser.add_argument("--ts", default=100000, type=int, help="Total number of epochs.")
	parser.add_argument("--lr", default=0.0005, type=float, help="Base learning rate at the start of the training.")
	parser.add_argument("--wd", default=0.0001, type=float, help="L2 weight decay.")
	parser.add_argument("--bs", default=256, type=int, help="Batch size used in the training and validation loop.")
	parser.add_argument("--dp", default=0.1, type=float)


	# seed
	parser.add_argument("--seed", default=123, type=int, help="seed")
	parser.add_argument("--print_freq", type=int, default=1000, help="print frequency")

	args = parser.parse_args()

	initialize(args.seed)

	# initialize directory
	args.cp_dir = f"checkpoints/adam/run_ms_{args.seed}/"
	create_path(args.cp_dir)
	for file in glob.glob("**/*.py", recursive=True):
		if "checkpoints" in file or "data" in file or "results" in file:
			continue
		os.makedirs(os.path.dirname(f"{args.cp_dir}/codes/{file}"), exist_ok=True)
		shutil.copy(file, f"{args.cp_dir}/codes/{file}")

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
	model = model.to(device)

	criterion = torch.nn.CrossEntropyLoss(ignore_index=1)
	optimizer = torch.optim.Adam(
		model.parameters(),
		lr=args.lr,
		betas=(0.9, 0.999), 
		eps=1e-08,
		weight_decay=args.wd,
		amsgrad=False)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
														   mode='min',
														   factor=0.5,
														   patience=1,
														   threshold=0.01,
														   min_lr=1e-6)

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
					     split='dev',
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
			
			optimizer.zero_grad()

			inputs_mask, targets_mask, src_padding_mask, tgt_padding_mask = create_mask(inputs, targets_input, device)
			outputs = model(inputs, targets_input, inputs_mask, targets_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

			targets_out = targets[1:, :]
			batch_loss = criterion(outputs.reshape(-1, outputs.shape[-1]), targets_out.reshape(-1))
			batch_loss.backward()

			optimizer.step()
			trainloss.update(batch_loss.item(), inputs.shape[1])

			if batch_idx % args.print_freq == 0:
				logging.info('Batch: {0}/{1}, Train_Loss = {2}'.format(batch_idx, len(dset_loaders['train']), trainloss.avg))

			if step % 1000 == 0:
				torch.save(model.state_dict(), f"{args.cp_dir}/"+str(step // args.print_freq)+'.pt')
				
			step+=1

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
			torch.save(model.state_dict(), f"{args.cp_dir}/best_model.pth.tar")
			best_loss = valloss.avg
		
		scheduler.step(valloss.avg)
		epoch+=1
		if epoch % 2 == 0:
			with torch.no_grad():
				score = compute_bleu(model, dset_loaders['test'], bpe_model, device)
			writer.add_scalar('Test/bleu_score', score,  step)
			logging.info('Bleu_score = {0}'.format(score))

		