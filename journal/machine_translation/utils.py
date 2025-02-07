from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
import random
from torchtext.data.metrics import bleu_score
from tqdm import tqdm 

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
SRC_LANGUAGE, TGT_LANGUAGE = 'de', 'en'


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


def initialize(seed: int):
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

	torch.backends.cudnn.enabled = True
	torch.backends.cudnn.benchmark = True
	torch.backends.cudnn.deterministic = False


def generate_square_subsequent_mask(sz, device):
	mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
	mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
	return mask


def create_mask(src, tgt, device):
	src_seq_len = src.shape[0]
	tgt_seq_len = tgt.shape[0]

	tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
	src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

	src_padding_mask = (src == PAD_IDX).transpose(0, 1)
	tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
	return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol, device):
	src = src.to(device)
	src_mask = src_mask.to(device)

	memory = model.encode(src, src_mask)
	ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
	for i in range(max_len-1):
		memory = memory.to(device)
		tgt_mask = generate_square_subsequent_mask(ys.size(0), device).type(torch.bool)
		out = model.decode(ys, memory, tgt_mask)
		out = out.transpose(0, 1)
		prob = model.generator(out[:, -1])
		_, next_word = torch.max(prob, dim=1)
		next_word = next_word.item()

		ys = torch.cat([ys,
						torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
		if next_word == EOS_IDX:
			break
	return ys


# actual function to translate input sentence into target language
def translate(model, text_transform, vocab_transform, src_sentence, device):
	model.eval()
	src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
	num_tokens = src.shape[0]

	src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
	tgt_tokens = greedy_decode(
		model, src, src_mask, max_len=num_tokens + 10, start_symbol=BOS_IDX, device=device).flatten()
	return vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))[1:-1]


def compute_bleu(model, dsets, text_transform, vocab_transform, device):
	all_targets = []
	all_outputs = []

	for inputs, targets in tqdm(dsets, desc="Computing Bleu Score"):
		targets = text_transform[TGT_LANGUAGE](targets.rstrip("\n"))
		targets = vocab_transform[TGT_LANGUAGE].lookup_tokens(list(targets.cpu().numpy()))[1:-1]

		prediction = translate(model, text_transform, vocab_transform, inputs, device)

		if len(prediction) < 4:
			continue

		all_targets.append([targets])
		all_outputs.append(prediction)

	return bleu_score(all_outputs, all_targets)


if __name__ == "__main__":
	# Test for Debug
	inputs = "ein pferd geht unter einer brücke neben einem boot."
	inputs = translate(model, inputs)
	logging.info(f"Translated Sentence \n {inputs}")
		