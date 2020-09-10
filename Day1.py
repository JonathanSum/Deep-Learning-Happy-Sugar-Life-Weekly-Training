import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random, pdb, copy, os, math, numpy, copy, time
import utils


##################
# Basic modules
##################


# encodes a  sequence of input frames and states, and optionally a cost or action, to a hidden representation
class encoder(nn.Module):
    def __int__(self, opt, a_size, n_inputs, state=True, state_input_size=4, n_channels=3):
        super(encoder, self).__init__()
        self.opt = opt
        self.a_size = a_size
        self.n_inputs = opt.ncond if n_inputs is None else n_inputs
        self.n_channels = n_channels
        #frame encoder
        if opt.layers == 3:
            assert(opt.nfeature % 4 ==0)
            self.feature_maps = ( opt.nfeature // 4, opt.nfeature // 2, opt.nfeature)
            self.f_encoder = nn.Sequential(
                nn.Conv2d(n_channels * self.n_inputs, self.feature_[0], 4, 2, 1),
                nn.Dropout2d(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.feature_maps[0], self.feature_maps[1], 4, 2, 1),
                nn.Dropout2d(p=opt.dropout, inplace = True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.feature_maps[1], self.feature_maps[2], 4, 2, 1),
            )
        elif opt.layers == 4:
            assert(opt.nfeature % 8 == 0)
            self.feature_maps = (opt.nfeature // 8, opt.nfeature //4, opt.nfeature // 2, opt.nfeature)
            self.f_encoder = nn.Sequential(
                nn.Conv2d(n_channels * self.n_inputs, self.feature_maps[0], 4, 2, 1),
                nn.Dropout2d(p=opt.dropout, inplace = True),
                nn.LeakyReLU(0.2, inplace = True),
                nn.Conv2d(self.feature_maps[0], self.feature_maps[1], 4, 2, 1),
                nn.Dropout2d(p=opt.dropout, inplace=True),
                nn.LeakyReLU(0.2, inplace = True),
                nn.Conv2d(self.feature_maps[1], self.feature_maps[2], 4, 2, 1),
                nn.Dropout2d(p=opt.dropot, inplace=True),
                nn.LeakyReLU(0.2, inplace = True),
                nn.Conv2d(self.feature_maps[2], self.feature_maps[3], 4, 2, 1)
                )
        if states:
            n_hidden = self.feature_maps[-1]
            # state_encoder
            self.s_encoder = nn.Sequential(
                nn.Linear(state_input_size * self.n_inputs, n_hidden)
                )
