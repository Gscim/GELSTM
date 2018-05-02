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
import time

train_conf = {
    "data_path": "../datasrc",
    "num_paths": 15, 
    "path_length": 20
    "embedding_dim": 64
    "hidden_dim": 128
    "num_epoch": 100
    "learning_rate": 0.01
}

# make the curpus using random walk algorithm
# get the vocab size

walks_data, vocab_size = reader.graph_walk_data(train_conf["data_path"])

model = GLNet(vocab_size, train_conf["embedding_dim"], train_conf["hidden_dim"])
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

print(train_conf)
print("running model with train_conf")

for epoch in range(train_conf["num_epoch"]):
    start_ = time.time()

    for walk_seq in walks_data:
        # clear the grad before each seq
        model.zero_grad()

        in_walk = torch.tensor(walk_seq, dtype=torch.long)

        loss = model.loss_def(in_walk)

        loss.backward()
        optimizer.step()

    end_ = time.time()
    print("epoch", epoch, "cost time", (end_ - start_), "seconds")

with torch.no_grad():
    print("exporting embedding to file \"lstm_trained.embedding\"")
    torch.save(model.ne_embeds, train_conf["data_path"] + "torch_nn.embedding")
    
