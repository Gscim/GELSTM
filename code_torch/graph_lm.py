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
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.ne_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        self.hidden = self.init_hidden()
        # add an linear at the output, out a single number
        self.linear_out = nn.Linear(hidden_dim, 1)

    def init_hidden(self):
        return (torch.randn(2, 1, hidden_dim // 2), torch.randn(2, 1, hidden_dim // 2))

    def get_poss_of_out(self, sequence_):
        seqlen = len(sequence_)
        embeds = self.ne_embeds(sequence_).view(len(sequence_), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        
        lstm_out = lstm_out.view(seqlen, self.hidden_dim)
        lstm_poss = self.linear_out(lstm_out)

        return lstm_poss

    def forward(self, sequence_):
        lstm_out = get_poss_of_out(sequence_)
        return 

    def loss_def(self, sequence_):
        poss_out = get_poss_of_out(sequence_)

        return 