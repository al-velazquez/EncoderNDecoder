#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 11:32:15 2022

@author: leonardo
"""

import torch
from AttentionClass import EncoderSelfAttn as encoder
from data_generation.square_sequence import generate_sequences

import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
import torch.nn as nn
from torch_geometric.nn import NNConv, Set2Set

#Creating data points (of squares) and their direction
points, directions=generate_sequences(n=128, seed=13)

x=torch.load("tensor_x.pt", map_location=torch.device("cpu"))
#x.size()
lin0=Linear(11, 64)
nn=Sequential(Linear(5,64), ReLU(), Linear(64, 64*64))
conv=NNConv(64, 64, nn, aggr="mean")
#se necesita edge_index y edge_attr 
edge_index=torch.load("tensor_edgeindex.pt", map_location=torch.device("cpu"))
edge_attr=torch.load("tensor_edgeattr.pt", map_location=torch.device("cpu"))
batch=torch.load("tensor_batch.pt", map_location=torch.device("cpu"))
#el buen forward
out=F.relu(lin0(x))
m=conv(out, edge_index, edge_attr)
m.size()
