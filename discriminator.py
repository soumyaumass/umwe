# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 16:45:48 2018

@author: Soumya
"""
import os
import io
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, lang):
        super(Discriminator, self).__init__()
        
        self.lang = lang
        self.emb_dim = 300
        self.model = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(self.emb_dim, 2048),
                nn.LeakyReLU(0.1),
                nn.Linear(2048, 2048),
                nn.LeakyReLU(0.1),
                nn.Linear(2048, 1))
        
    def forward(self, x):
        assert x.dim() == 2 and x.size(1) == self.emb_dim
        return torch.sigmoid(self.model(x)).view(-1)