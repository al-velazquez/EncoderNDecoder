#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 16:30:04 2022

@author: leonardo

Aquí estaré haciendo las pruebas
hopefully it does work super nice :d

"""
import torch
from AttentionClass import EncoderSelfAttn as encoder
from data_generation.square_sequence import generate_sequences

import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
import torch.nn as nn
from torch_geometric.nn import NNConv, Set2Set

#aquí va  x
x=torch.load("tensor_x.pt", map_location=torch.device("cpu"))

#y luego las operaciones anteriores a la convolución
lin0=Linear(11, 64)
nn=Sequential(Linear(5,64), ReLU(), Linear(64, 64*64))
conv=NNConv(64, 64, nn, aggr="mean")

#primer pedacito del forward()
out=F.relu(lin0(x))

#se necesita el edge_index y edge_attr
edge_index=torch.load("tensor_edgeindex.pt", map_location=torch.device("cpu"))
edge_attr=torch.load("tensor_edgeattr.pt", map_location=torch.device("cpu"))

#se encuentra una primera convolución
m=conv(out, edge_index, edge_attr)
#se carga la lista de batch
batch=torch.load("tensor_batch.pt", map_location=torch.device("cpu"))
#la función que separa por moléculas
def batch_sequences_features(x, batch):
    def contador(batch):
        diccionario={}
        for i in batch:
            if i.item() not in diccionario:
                diccionario[i.item()]=1
            else:
                diccionario[i.item()]+=1
        return diccionario

    N_first=[]
    l_inf=0
    l_sup=0
    dictionary=contador(batch)
    for i in dictionary:
        l_sup=l_sup+dictionary[i]
        N_first.append(x[l_inf:l_sup])
        l_inf=l_sup
    return N_first
from FullENcoder import EncoderSelfAttnL 
#para utilizar al vecino
Encoder=EncoderSelfAttnL(n_heads=3, d_model=64, ff_units=10, n_features=64)
new=Encoder.batch_sequences_features(m, batch)


#contador otra vez
def contador(batch):
    diccionario={}
    for i in batch:
        if i.item() not in diccionario:
            diccionario[i.item()]=1
        else:
            diccionario[i.item()]+=1
    return diccionario
cuenta=contador(batch)
#max_key
max_key=max(cuenta, key=cuenta.get)
max_max=cuenta[max_key]

#La nueva función
def my_padd(batch, new):
    cuenta=contador(batch)
    max_key=max(cuenta, key=cuenta.get)
    max_max=cuenta(max_key)
    dim=64
    for i, j in enumerate(new):
        dif=cuenta[max_key]-len(j)
        all_zeros=torch.zeros(dim).repeat(dif,1)
        new[i]=torch.cat([new[i], all_zeros],0)
    tensor=torch.cat(new)
    mask=(tensor!=0).all(axis=2).unsqueeze(1)
    return tensor, mask
encoder2=EncoderSelfAttnL(n_heads=3, d_model=64, ff_units=10, n_features=64)
# query, mask=encoder2.my_padding(new, batch)

# encoder2.self_attn_heads.init_keys(query)
# att=encoder2.self_attn_heads(query, mask)

help1=encoder2(m, batch)


#ahora que tengo un tensor en dimension [batch, batch, dim]
#hay que regresarlo a [batch, dim] pero en grafo, eso significa
#cortar las secuencias paddiadas
"""
def get_back(encoded_states,mask):
    get_back=[]
    for i in range(help1.size()[0]):
        for j in range(help1.size()[1]):
            if mask[i][0][j]==True:
                get_back.append(torch.as_tensor(encoded_states[i][j]).float())
                
            else:
                pass

    return get_back

back=get_back(help1, mask)
"""
mask=encoder2.mask
#pad, mask=encoder2.my_padding()
got_back=encoder2.get_back(help1,mask)