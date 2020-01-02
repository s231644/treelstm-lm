from typing import List, Set, Dict, Tuple
from collections import defaultdict
import numpy as np

def read_sentence(file_name: str) -> List[dict]:
    sentence = [{'token': '[ROOT]', 'parent': -1}]
    with open(file_name, 'r') as f:
        for l in f:
            l = l.strip()
            if l.startswith('#'):
                continue
            if not l:
                break
            l = l.split('\t')
            sentence.append({'token': (l[2].lower(), l[3]), 'parent': int(l[6])})
    return sentence


def read_sentences(file_name: str) -> List[dict]:
    sentences = []
    with open(file_name, 'r') as f:
        for l in f:
            l = l.strip()
            if l.startswith('#'):
                # a new sentence begins
                sentence = [{'token': '[ROOT]', 'parent': -1}]
                continue
            if not l:
                sentences.append(sentence)
                continue
            l = l.split('\t')
            sentence.append({'token': (l[2].lower(), l[3]), 'parent': int(l[6])})
    return sentences


class Vocabulary:
    def __init__(self, name='en', min_freq=1):
        self.name = name
        self.min_freq = min_freq
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 0
        self.n_vocab = 3
        self.token2id = defaultdict(lambda: 2)
        self.id2token = ['[ROOT]', '[MASK]', '[UNK]']
        self.token2id['[ROOT]'] = 0
        self.token2id['[MASK]'] = 1
        self.token2id['[UNK]'] = 2
        self.add_word('[ROOT]')
        self.add_word('[MASK]')
        self.add_word('[UNK]')
        
    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
        if self.word2count[word] >= self.min_freq and word not in self.token2id:
            self.token2id[word] = self.n_vocab
            self.id2token.append(word)
            self.n_vocab += 1
            
    def add_sentence(self, sentence: List[dict]):
            for word in sentence:
                self.add_word(word['token'])

    def fit_sentences(self, sentences: List[List[dict]]):
        for sentence in sentences:
            self.add_sentence(sentence)

    def tokenize_sentence(self, sentence: List[dict]) -> List[dict]:
        tokenized = [{'token_id': self.token2id[token['token']], 
                      'parent': token['parent'],
                      'children': []} 
                     for token in sentence]
        for i, token in enumerate(tokenized):
            if token['parent'] == -1:
                continue
            tokenized[token['parent']]['children'].append(i)
        return tokenized
    
    def detokenize_sentence(self, tokenized: List[dict]) -> List[str]:
        return [self.id2token[token['token_id']] for token in tokenized]

    def tokenize_sentences(self, sentences: List[List[dict]]):
        return [self.tokenize_sentence(sentence) for sentence in sentences]


def prepare_tree(tokenized: List[dict]):
    # bfs
    children = [token['children'] for token in tokenized]
    parents = [[token['parent']] for token in tokenized]
    parents[0] = []
    tokens = [token['token_id'] for token in tokenized]
    
    q = [0]
    i = 0
    while i < len(q):
        for child in reversed(children[q[i]]):
            q.append(child)
        i += 1
    i = len(q)
    while i > 1:
        i -= 1
        par = parents[q[i]][0]
    
    tokens = np.array(tokens, dtype=np.int32)
    parents = np.array([np.array(item, dtype=np.int16) for item in parents])
    children = np.array([np.array(item, dtype=np.int16) for item in children])
    q = np.array(q, dtype=np.int16)
    
    return tokens, parents, children, q
