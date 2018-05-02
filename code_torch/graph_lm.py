'''
    using pytorch
    this work is a trial version
    some details are not clearly setted yet
    need further research
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import graph_reader as reader 
import time
import numpy as np 
import torch

import torch.nn as nn

Config = {}

class GLNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(GLNet, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.ne_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        self.hidden = self.init_hidden()
        # add an linear at the output, out a shape of (sequence, vocab_size)
        self.linear_out = nn.Linear(hidden_dim, vocab_size)

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2), torch.randn(2, 1, self.hidden_dim // 2))

    def get_poss_of_out(self, sequence_):
        seqlen = len(sequence_)
        embeds = self.ne_embeds(sequence_).view(len(sequence_), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        
        lstm_out = lstm_out.view(seqlen, self.hidden_dim)
        lstm_poss = self.linear_out(lstm_out)
        softmax = nn.Softmax(dim=1)
        lstm_poss = softmax(lstm_poss)
        # lstm poss with a sahpe of (sequence, vocab)
        return lstm_poss
    
    def loss_def(self, sequence_):
        # to match the target with output
        poss_out = get_poss_of_out(sequence_)[:-1]
        seq = sequence_.type(torch.FloatTensor)
        target = sequence_[1:]

        cross_entropy_loss = nn.CrossEntropyLoss()
        loss_ = cross_entropy_loss(poss_out, target)

        return loss_

    def forward(self, sequence_):
        loss_ = self.loss_def(sequence_)
        return loss_


