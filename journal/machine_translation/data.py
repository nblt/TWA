from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k, IWSLT2016, IWSLT2017
from torchtext.experimental.datasets.raw import WMT14
from typing import Iterable, List
from torch.utils.data import DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence

from torchnlp.datasets import wmt_dataset, imdb_dataset

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
SRC_LANGUAGE, TGT_LANGUAGE = 'de', 'en'


# helper function to club together sequential operations
def sequential_transforms(*transforms):
	def func(txt_input):
		for transform in transforms:
			txt_input = transform(txt_input)
		return txt_input
	return func


# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]), 
                      torch.tensor(token_ids), 
                      torch.tensor([EOS_IDX])))


def get_transforms(path):

    token_transform = torch.load(f"{path}/token_transform.pth")
    vocab_transform = torch.load(f"{path}/vocab_transform.pth")
    

    # src and tgt language text transforms to convert raw strings into tensors indices
    text_transform = {}
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                                   vocab_transform[ln], #Numericalization
                                                   tensor_transform) # Add BOS/EOS and create tensor

    return vocab_transform, text_transform


if __name__ == "__main__":
	vocab_transform, text_transform = get_transforms()

	transforms = {
		'vocab_transform': vocab_transform,
		'text_transform': text_transform
	}

	torch.save(transforms, "data/transforms.pth")

	dsets = wmt_dataset(directory="data/wmt/", train=True)
	batch = next(dsets)
	src_batch, tgt_batch = [], []
	src_sample, tgt_sample = batch
	print(src_sample)
	batch = next(dsets)
	src_batch, tgt_batch = [], []
	src_sample, tgt_sample = batch
	print(src_sample, tgt_sample)
	batch = next(dsets)
	src_batch, tgt_batch = [], []
	src_sample, tgt_sample = batch
	print(src_sample)
	# src_batch.append(text_transform['de'](src_sample.rstrip("\n")))
	# tgt_batch.append(text_transform['en'](tgt_sample.rstrip("\n")))
	# src_batch = pad_sequence(src_batch, padding_value=1)
	# tgt_batch = pad_sequence(tgt_batch, padding_value=1)
	# print(tgt_batch, tgt_batch[:-1,:])
	# targets_input = tgt_batch[:-1, :]

	# print(src_batch.shape, tgt_batch.shape, targets_input.shape)
		
	# from utils import *
	# inputs_mask, targets_mask, src_padding_mask, tgt_padding_mask = create_mask(src_batch, targets_input, torch.device('cpu'))
	# print(inputs_mask, targets_mask)

