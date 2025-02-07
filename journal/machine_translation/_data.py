import torch
import os
import youtokentome as yttm
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3


class Translation(Dataset):
	def __init__(self, directory, split, bpe):
		self.bpe = bpe
		with open(os.path.join(directory, '%s.de'%split)) as f:
			source = [l.strip() for l in f.readlines()]
		with open(os.path.join(directory, '%s.en'%split)) as f:
			target = [l.strip() for l in f.readlines()]
		assert len(source) == len(target)
		self.data = tuple(zip(source, target))
		
	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		"""
		Returns:
			x, y (list(string)): Source/target sentence.
		"""
		x, y = self.data[idx]
		return x, y

	def generateBatch(self, batch):
		"""
		Generate a mini-batch of data. For DataLoader's 'collate_fn'.

		Args:
			batch (list(tuple)): A mini-batch of (source, target) sentence pairs.

		Returns:
			xs, ys (torch.LongTensor, [batch_size, (padded) seq_length]): A mini-batch of encoded source/target sentences.
		"""
		xs, ys = zip(*batch)
		# Encode (sentence --> subword IDs)
		xs = self.bpe.encode(xs, bos=True, eos=True)
		ys = self.bpe.encode(ys, bos=True, eos=True)
		# Transform data type from list to PyTorch tensor
		xs = [torch.LongTensor(x) for x in xs]
		ys = [torch.LongTensor(y) for y in ys]
		# Padding
		xs = rnn_utils.pad_sequence(xs, padding_value=PAD_IDX)   # [(padded) src_length, batch_size]
		ys = rnn_utils.pad_sequence(ys, padding_value=PAD_IDX)   # [(padded) tgt_length, batch_size]
		return xs, ys


def load(directory, split, batch_size, bpe_model, workers=0, shuffle=True):
	"""
	Args:
		directory (string): Directory of dataset.
		split (string): Which of the subset of data to take. One of 'train', 'dev' or 'test'.
		batch_size (integer): Batch size.
		bpe_model (youtokentome.BPE): Byte-pair encoding model.
		workers (integer): How many subprocesses to use for data loading.
		shuffle (bool): Shuffle the dataset or not.

	Returns:
		loader (DataLoader): A DataLoader can generate batches of (source, target) pairs.
	"""
	assert split in ['train', 'dev', 'test']
	print ("Loading %s dataset ..." % split.upper())
	dataset = Translation(directory, split, bpe_model)
	print ("%s set size:"%split.upper(), len(dataset))
	loader = DataLoader(dataset,
						batch_size=batch_size,
						collate_fn=dataset.generateBatch,
						shuffle=shuffle,
						num_workers=workers,
						pin_memory=True)
	return loader


def inspect_data():
	"""
	Load a few samples and check the functionality of input pipeline.
	"""
	import utils
	os.environ["CUDA_VISIBLE_DEVICES"] = ''

	DIR = "data/"
	BPE_MODEL = "data/bpe.32000.model"
	BATCH_SIZE = 2
	SPLIT = 'dev'
	SHUFFLE = False

	bpe_model = yttm.BPE(model=BPE_MODEL)
	loader = load(DIR, SPLIT, BATCH_SIZE, bpe_model, shuffle=SHUFFLE)

	xs, ys = next(iter(loader))
	xs, ys = xs.transpose(0, 1), ys.transpose(0, 1)

	xs = utils.decode(xs.tolist(), bpe_model)
	ys = utils.decode(ys.tolist(), bpe_model)
	for i in range(BATCH_SIZE):
		print ("\n", i+1)
		print (' '.join(xs[i]))
		print (' '.join(ys[i]))
		

if __name__ == '__main__':
	inspect_data()
