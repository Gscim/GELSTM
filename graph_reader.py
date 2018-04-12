'''
    to parse the giving graph file
    needed:
    
    graph_walk_data()
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import tensorflow as tf 

def _read_ent_rel(filename):
    with open(filename, 'r') as f:
        return f.read().strip().split()

def _build_vocab(filename):
    '''
    sorted by frequency of words(here words means nodes and relations)
    '''
    data = _read_ent_rel(filename)
    counter = collections.Counter(data)
    count_pairs = sorted(couter.items(), key=lambda x:(-x[1], x[0]))

    word, _ = list(zip(*count_pairs))
    word2id = dict(zip(words, range(len(words))))
    
    return word2id




