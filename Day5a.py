import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import pdb
import copy
import os
import math
import numpy
import copy
import time
import utils


##################
# Basic modules
##################


# encodes a  sequence of input frames and states, and optionally a cost or action, to a hidden representation
class encoder(nn.Module):
    def __int__(self, opt, a_size, n_inputs, states=True, state_input_size=4, n_channels=3):
        super(encoder, self).__init__()
        self.opt = opt
        self.a_size = a_size
        self.n_inputs = opt.ncond if n_inputs is None else n_inputs
        self.n_channels = n_channels
        # frame encoder
        if opt.layers == 3:
            assert(opt.nfeature % 4 == 0)
            self.feature_maps = (
                opt.nfeature // 4, opt.nfeature // 2, opt.nfeature)
            self.f_encoder = nn.Sequential(
                nn.Conv2d(n_channels * self.n_inputs,
                          self.feature_[0], 4, 2, 1),
                nn.Dropout2d(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.feature_maps[0], self.feature_maps[1], 4, 2, 1),
                nn.Dropout2d(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.feature_maps[1], self.feature_maps[2], 4, 2, 1),
            )
        elif opt.layers == 4:
            assert(opt.nfeature % 8 == 0)
            self.feature_maps = (
                opt.nfeature // 8, opt.nfeature // 4, opt.nfeature // 2, opt.nfeature)
            self.f_encoder = nn.Sequential(
                nn.Conv2d(n_channels * self.n_inputs,
                          self.feature_maps[0], 4, 2, 1),
                nn.Dropout2d(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.feature_maps[0], self.feature_maps[1], 4, 2, 1),
                nn.Dropout2d(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.feature_maps[1], self.feature_maps[2], 4, 2, 1),
                nn.Dropout2d(p=opt.dropot, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.feature_maps[2], self.feature_maps[3], 4, 2, 1)
            )
        if states:
            n_hidden = self.feature_maps[-1]
            # state_encoder
            self.s_encoder = nn.Sequential(
                nn.Linear(state_input_size * self.n_inputs, n_hidden),
                nn.Dropout(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(n_hidden, n_hidden),
                nn.Dropout(p=opt.dropout, inplace=True),
                nn.LeakyRelu(0.2, inplace=True),
                nn.Linear(n_hidden, opt.hidden_size)
            )

            if a_size > 0:
                # action or cost encoder
                n_hidden = self.feature_map[-1]
                self.a_encoder = nn.Sequential(
                    nn.Linear(a_size, n_hidden),
                    nn.Dropout(p=opt.dropout, inplace=True),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(n_hidden, n_hidden),
                    nn.Dropout(p=opt.dropout, inplace=True),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(n_hidden, opt.hidden_size)
                )

    def forward(self, images, states=None, actions=None):
        bsize = images.size(0)
        h = self.f_encoder(images.view(
            bsize, self.n_inputs * self.n_channels, self.opt.height, self.opt.width))
        if states is not None:
            h = h + self.s_encoder(  states.contiguous().view(bsize, -1)).view(h.size()  )
        if actions is not None:
            a = self.a_encoder(actions.contiguous().view(bsize, self.a_size))
            h = h+a.view(h.size())            
        return h

class u_network(nn.Module):
    def __init__(self, opt):
        super(u_network, self, self).__init__()
        self.opt = opt
        self.encoder = nn.Sequential(
            nn.Conv2d(self.opt.nfeature, self.opt, 4, 2, 1),
            nn.Dropout2d(p=opt.dropout, inplace = True),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(self.opt.nfeature, self.opt.nfeature, (4, 1), 2, 1)
        )

        self.decoder = nn.Sequential(
            nn.COnvTransposed2d(self.opt.nfeature, self.opt.nfeature, (4, 1), 2, 1),
            nn.Dropout2d(p=opt.dropout,, inplace=True),
            nn.LeakyReLU(0.2, inplace = True),
            nn.ConvTransposed2d(self.opt.nfeature, self.opt.nfeature, (4, 3), 2 , 0)
        )
