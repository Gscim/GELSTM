'''
    to parse the giving graph file
    needed:
    
    graph_walk_data()

    walk sequence: 
        num_paths = 30, path_length = 50

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import tensorflow as tf 

import json
import graph

def _read_ent_rel(filename):
    with open(filename, 'r') as f:
        return f.read().strip().split()

def _build_vocab(filename):
    '''
    sorted by frequency of words(here words means nodes and relations)
    '''
    data = _read_ent_rel(filename)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x:(-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word2id = dict(zip(words, range(len(words))))
    
    return word2id

def _graph_to_edgelist(in_filename, out_filename, word2id):
    with open(in_filename, 'r') as in_file, open(out_filename, 'w') as out_file:
        for line in in_file:
            obj, pred, subj = line.strip().split()[:3]
            print(word2id[obj], word2id[subj], word2id[pred], file=out_file)
    
    


def graph_walk_data(data_path=None):
    p_train_path = os.path.join(data_path, "train.txt")
    p_valid_path = os.path.join(data_path, "valid.txt")
    p_test_path = os.path.join(data_path, "test.txt")
    train_path = os.path.join(data_path, "train.edgelist")
    valid_path = os.path.join(data_path, "valid.edgelist")
    test_path = os.path.join(data_path, "test.edgelist")

    word2id = _build_vocab(p_train_path)
    _graph_to_edgelist(p_train_path, train_path, word2id)
    _graph_to_edgelist(p_valid_path, valid_path, word2id)
    _graph_to_edgelist(p_test_path, test_path, word2id)
    json.dump(word2id, open(os.path.join(data_path, "word2id.json"), 'w'))

    G_train = graph.load_edgelist(train_path)
    G_valid = graph.load_edgelist(valid_path)
    G_test = graph.load_edgelist(test_path)

    train_walks = graph.build_deepwalk_corpus(G_train, list_exclud=[], num_paths=30, path_length=50)
    valid_walks = graph.build_deepwalk_corpus(G_valid, list_exclud=[], num_paths=30, path_length=50)
    test_walks = graph.build_deepwalk_corpus(G_test, list_exclud=[], num_paths=30, path_length=50)
    vacabulary = len(word2id)

    return train_walks, valid_walks, test_walks, vacabulary

def graph_producer(walk_data, batch_size, num_steps, name=None):
    '''
        iterate on the walk_data
        chunks up walk_data into batches of examples 
        and return tensors that are drawn from these batches

        Args:
            raw_data: one of the ouput of graph_walk_data
            batch_size: int, the batch size
            num_steps: int, the number of unrolls
            name: the name of this operation
        
        Returns:
            A pair of Tensors, each shaped [batch_size, num_steps]. The second element
            of the tuple is the same data time-shifted to the right by one.
    '''
    with tf.name_scope(name, "GraphProducer", [walk_data, batch_size, num_steps]):
        walk_data = tf.convert_to_tensor(walk_data, name="walk_data", dtype=tf.int32)

        data_len = tf.size(walk_data)
        batch_len = data_len // batch_size
        data = tf.reshape(walk_data[0 : batch_size * batch_len], [batch_size, batch_len])

        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.assert_positive(epoch_size, message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")
        
        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data, [0, i * num_steps], [batch_size, (i + 1) * num_steps])
        x.set_shape([batch_size, num_steps])
        y = tf.strided_slice(data, [0, i * num_steps + 1], [batch_size, (i + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])
        return x, y


