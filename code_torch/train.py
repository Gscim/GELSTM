'''
    training a model
    using pytorch
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.optim as optim
import torch.autograd as autograd
from graph_lm import GLNet
import graph_reader as reader

train_conf = {
    "data_path": "../datasrc",
    "num_paths": 15, 
    "path_length": 20
    "embedding_dim": 64
    "hidden_dim": 8
    "num_epoch": 100
}

# make the curpus using random walk algorithm
# get the vocab size

walks_data, vocab_size = reader.graph_walk_data(train_conf["data_path"])

model = GLNet(vocab_size, train_conf["embedding_dim"], train_conf["hidden_dim"])
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

for epoch in range(train_conf["num_epoch"]):


    
