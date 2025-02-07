from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
import random
from torchtext.data.metrics import bleu_score

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


def decode(s_in_batch, bpe_model):
    """
    Args:
        s_in_batch (list(list(integer))): A mini-batch of sentences in subword IDs.

    Returns:
        s_out_batch (list(list(string))): A mini-batch of sentences in words.
    """
    s_out_batch = []
    for s in s_in_batch:
        if s == 2: # <bos>
            continue
        elif s == 3: # <eos>
            break

        s_out_batch.append(s)
    
    s_out_batch = bpe_model.decode(s_out_batch)   # list(string)
    s_out_batch = [s.split() for s in s_out_batch]   # list(list(string))
    return s_out_batch


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
@torch.no_grad()
def greedy_decode(model, src, src_mask, max_len, device):
	memory = model.encode(src, src_mask)
	ys = torch.ones(1, 1).fill_(BOS_IDX).type(torch.long).to(device)
	
	for i in range(max_len-1):
		tgt_mask = generate_square_subsequent_mask(ys.size(0), device).type(torch.bool)

		out = model.decode(ys, memory, tgt_mask)
		out = out.transpose(0, 1)
		prob = model.generator(out[:, -1])

		_, next_word = torch.max(prob, dim=1)
		next_word = next_word.item()

		ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
		
		if next_word == EOS_IDX:
			break
	
	return ys


# actual function to translate input sentence into target language
@torch.no_grad()
def translate(model, inputs, bpe_model, device):
	model.eval()
	
	inputs = bpe_model.encode(inputs, bos=True, eos=True)
	inputs = torch.LongTensor(inputs).to(device)

	num_tokens = inputs.shape[0]
	inputs_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool).to(device)

	outputs = greedy_decode(model, inputs, inputs_mask, max_len=num_tokens + 10, device=device).flatten()

	outputs = decode(outputs.tolist(), bpe_model)[0]

	return " ".join(outputs)


@torch.no_grad()
def compute_bleu(model, dsets, bpe_model, device):
	all_targets = []
	all_outputs = []

	for batch in dsets:
		inputs, targets = batch
		inputs, targets = inputs.to(device), targets.to(device)
		targets = targets[1:-1,:]
		
		num_tokens = inputs.shape[0]
		inputs_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool).to(device)
		
		prediction = greedy_decode(model, inputs, inputs_mask, max_len=num_tokens + 10, device=device).flatten()
		prediction = decode(prediction.tolist(), bpe_model)
		if len(prediction) != 0: 
			prediction = prediction[0]
		else: 
			prediction = [""]

		targets = decode(targets.flatten().tolist(), bpe_model)
		if len(targets) != 0: 
			targets = targets[0]
		else: 
			targets = [""]

		all_targets.append([targets])
		all_outputs.append(prediction)

	return bleu_score(all_outputs, all_targets)


if __name__ == "__main__":
	# Test for Debug
	inputs = "ein pferd geht unter einer brücke neben einem boot."
	inputs = translate(model, inputs)
	logging.info(f"Translated Sentence \n {inputs}")
		