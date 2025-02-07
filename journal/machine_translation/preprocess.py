from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k, IWSLT2016, IWSLT2017
from torchtext.experimental.datasets.raw import WMT14
from typing import Iterable, List
from torch.utils.data import DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
import os 

from torchnlp.datasets import wmt_dataset, imdb_dataset


UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
SRC_LANGUAGE, TGT_LANGUAGE = 'de', 'en'


def yield_tokens(token_transform, data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])


def get_token_transform(path):
    token_transform = {
        SRC_LANGUAGE: get_tokenizer('spacy', language='de_core_news_sm'),
        TGT_LANGUAGE: get_tokenizer('spacy', language='en_core_web_sm')
    }

    torch.save(token_transform, f"{path}/token_transform.pth")

    return token_transform


def get_vocab_transform(path):
    if os.path.isdir(path) is False:
        os.makedirs(path)

    token_transform = get_token_transform(path)

    special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
    vocab_transform = {}
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        train_iter = wmt_dataset(directory="data/wmt/", train=True)
        vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(token_transform, train_iter, ln),
                                                        min_freq=2,
                                                        specials=special_symbols,
                                                        special_first=True)
        vocab_transform[ln].set_default_index(UNK_IDX)

    torch.save(vocab_transform, f"{path}/vocab_transform.pth")
    print(len(vocab_transform['en']), len(vocab_transform['de']))
    return



if __name__ == "__main__":
    get_vocab_transform("data/iwslt17")

