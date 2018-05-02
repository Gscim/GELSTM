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
import logging

logging.basicConfig(level=logging.INFO, 
                    filename='log.txt', 
                    filemode='a', 
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s') 

train_conf = {
    "data_path": "../datasrc",
    "num_paths": 15, 
    "path_length": 20,
    "embedding_dim": 128,
    "hidden_dim": 256,
    "num_epoch": 100,
    "learning_rate": 0.01
}

# make the curpus using random walk algorithm
# get the vocab size

walks_data, vocab_size = reader.graph_walk_data(train_conf["data_path"])

model = GLNet(vocab_size, train_conf["embedding_dim"], train_conf["hidden_dim"])
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

print(train_conf)
print("running model with train_conf")
logging.info(train_conf)
logging.info("running model with train_conf")

for epoch in range(train_conf["num_epoch"]):
    start_ = time.time()

    for walk_seq in walks_data:
        # clear the grad before each seq
        model.zero_grad()

        in_walk = torch.tensor(walk_seq, dtype=torch.long)

        loss = model.loss_def(in_walk)

        loss.backward(retain_graph=True)
        optimizer.step()

    end_ = time.time()
    print("epoch", epoch, "cost time", (end_ - start_), "seconds")
    logging.info("epoch {} cost time {} seconds".format(epoch, (end_ - start_)))

with torch.no_grad():
    print("exporting embedding to file \"lstm_trained.embedding\"")
    logging.info("exporting embedding to file \"lstm_trained.embedding\"")
    torch.save(model.ne_embeds, train_conf["data_path"] + "torch_nn.embedding")    
