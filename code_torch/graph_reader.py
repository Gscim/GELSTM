'''
    functions needed to get an graph, 
    turns it into a id-represented formation
    an reader of the graph
    we got the graph input as an edgelist
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import graph
import collections
import json
import numpy as np 
import os
import sys

RWConf = {"num_paths": 10, "path_length": 20}

def _read_ent_rel(filename):
    with open(filename, 'r') as f:
        return f.read().strip().split()

def _build_vocab(filename):
    '''
    sort nodes and relations together by frequency
    '''
    filedata = read_ent_rel(filename)
    counter = collections.Counter(filedata)
    count_pairs = sorted(counter.items(), key=lambda x:(x[-1], x[0]))

    words, _ = list(zip(*count_pairs))
    word2id = dict(zip(words, range(len(words))))

    return word2id

def _graph_to_edgelist(in_filename, out_filename, word2id):
    with open(in_filename, 'r') as in_file, open(out_filename, 'w') as out_file:
        for line in in_file:
            obj, pred, subj = line.strip().split()[:3]
            print(word2id[obj], word2id[subj], word2id[pred], file=outfile)

def graph_walk_data(data_path=None):
    p_train_path = os.path.join(data_path, "train.txt")
    train_path = os.path.join(data_path, "train.edgelist")

    word2id = _build_vocab(p_train_path)
    _graph_to_edgelist(p_train_path, train_path, word2id)

    print("export word2id file....")
    json.dum(word2id, open(os.path.join(data_path, "word2id.json"), 'w'))
    print("word2id file exported.")

    G_train = graph.load_edgelist(train_path)
    train_walks = graph.build_deepwalk_corpus(G_train, list_exclude=[], num_paths=RWConf["num_paths"], path_length=RWConf["path_length"])

    vacabulary = len(word2id)

    return train_walks, vocabulary



