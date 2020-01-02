import dynet as dy
import time
from tqdm import tqdm
import numpy as np
import time
from copy import deepcopy

from typing import List, Set, Dict, Tuple
from collections import defaultdict
import sys
import logging

from utils import *
from dynet_modules import *
from training import *


if __name__ == "__main__":
    MIN_COUNT = 1
    BATCH_SIZE=128
    EMB_SIZE=128
    percent_masking = 0.1
    LR = 0.1

    sents_tokenized = []
    vocab = Vocabulary('en', MIN_COUNT)
    for part in ['example_ud_sent.txt']:
        sents = read_sentences(part)
        vocab.fit_sentences(sents)
        sents_tokenized.extend(vocab.tokenize_sentences(sents))
    
    logging.basicConfig(filename='training.log',level=logging.DEBUG)
    logging.info(f'{vocab.name}, emb={EMB_SIZE}, minc={MIN_COUNT}, bs={BATCH_SIZE}, lr={LR}')
    

    logging.info(f'n_words={vocab.n_words}, n_vocab={vocab.n_vocab}, num_sent={len(sents_tokenized)}')
    data_train = [prepare_tree(sent) for sent in sents_tokenized] 

    
    model = TreeLSTMLM(vocab.n_vocab, vocab.n_vocab, EMB_SIZE)
    trainer = dy.AdamTrainer(model.model)

    for i in range(0,10):
        msg = train_minibatches(model, trainer, data_train, BATCH_SIZE, percent_masking, True)
        logging.info(f'Epoch {i} ' + msg)
        model.model.save(f'{vocab.name},emb={EMB_SIZE},minc={MIN_COUNT},bs={BATCH_SIZE},lr={LR},epoch={i}')

