'''
    training a model
    using pytorch
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.optim as optim
import torch.nn as nn
import torch.autograd as autograd
from graph_lm import GLNet
import graph_reader as reader
import time
import logging
import random
logging.basicConfig(level=logging.INFO, 
                    filename=str(time.time()) + '.log', 
                    filemode='a', 
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s') 

train_conf = {
    "data_path": "../datasrc",
    "num_paths": 15, 
    "path_length": 20,
    "embedding_dim": 128,
    "hidden_dim": 256,
    "num_epoch": 300,
    "learning_rate": 0.01
}

loss_func = nn.CrossEntropyLoss().cuda()

# make the curpus using random walk algorithm
# get the vocab size
walks_data, vocab_size = reader.graph_walk_data(train_conf["data_path"])

logging.info("vocab size {}".format(vocab_size))

model = GLNet(vocab_size, train_conf["embedding_dim"], train_conf["hidden_dim"]).cuda()
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

#print(train_conf)
#print("running model with train_conf")
logging.info(train_conf)
logging.info("running model with train_conf")

#code blow not work because the walks_data is not aligned
#walks_data = torch.tensor(walks_data, dtype=torch.long).cuda()
'''
with torch.no_grad():
    walks_tensor = []
    for walk_ in walks_data:
        walks_tensor.append(torch.tensor(walk_, dtype=torch.long).cuda())
'''

logging.info("num of walk seqs {}".format(len(walks_data)))
logging.info("starting running epochs")
for epoch in range(train_conf["num_epoch"]):
    start_ = time.time()
    seqn = 0
    random.shuffle(walks_data)
    etloss = 0.0
    for in_walk in walks_data:
        # clear the grad before each seq
        if len(in_walk) <= 1:
            continue

        in_walk = torch.tensor(in_walk, dtype=torch.long).cuda()
        model.zero_grad()

        #logging.info("len = {}".format(len(in_walk)))
        #loss = model.loss_def(in_walk)

        _poss_out = model(in_walk)[:-1]
        _target = in_walk[1:]
        loss = loss_func(_poss_out, _target)

        etloss = etloss + loss.item()

        loss.backward(retain_graph=True)
        optimizer.step()

        #del loss
        #logging.info("seq {}".format(seqn))
        '''
        with torch.no_grad():
            if seqn % 1000 == 0:
                #end_ = time.time()
                #logging.info("100 cost time {}".format(end_-start_))
                #start_ = end_
                logging.info("seq {} finished".format(seqn))
            seqn = seqn + 1
        '''
        
    end_ = time.time()
    #print("epoch", epoch, "cost time", (end_ - start_), "seconds")
    logging.info("epoch {} cost time {} seconds".format(epoch, (end_ - start_)))
    logging.info("loss for epoch {} is {}".format(epoch, etloss))

with torch.no_grad():
    #print("exporting embedding to file \"lstm_trained.embedding\"")
    logging.info("exporting embedding to file \"lstm_trained.embedding\"")
    torch.save(model.ne_embeds, train_conf["data_path"] + "torch_nn.embedding")    
