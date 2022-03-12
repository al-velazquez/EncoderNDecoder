#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 11:53:33 2022

@author: leonardo
"""
import copy
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch.nn.utils.rnn import pad_sequence
import torch.nn.utils.rnn as rnn_utils

class Attention(nn.Module):
    def __init__(self, hidden_dim, input_dim=None, proj_values=False):
        super().__init__()
        self.d_k=hidden_dim
        self.input_dim=hidden_dim if input_dim is None else input_dim
        self.proj_values=proj_values
        #Affine transformations for q, k, and v
        self.linear_query=nn.Linear(self.input_dim, hidden_dim)
        self.linear_keys=nn.Linear(self.input_dim, hidden_dim)
        self.linear_value=nn.Linear(self.input_dim, hidden_dim)
        self.alphas=None
    
    def init_keys(self, keys):
        self.keys=keys
        self.proj_keys=self.linear_keys(self.keys)
        self.values=self.linear_value(self.keys) if self.proj_values else self.keys
        
    def score_function(self, query):
        proj_query=self.linear_query(query)
        #scaled dot products
        #N, 1, H x N, H, L ---> N,1,L
        dot_products=torch.bmm(proj_query, self.proj_keys.permute(0,2,1))
        scores=dot_products/np.sqrt(self.d_k)
        
        return scores
    
    def forward(self, query, mask=None):
        #query is batch-first
        scores=self.score_function(query)#N, 1, L
        if mask is not None:
            scores=scores.masked_fill(mask==0, -1e9)
        alphas = F.softmax(scores, dim=-1) # N, 1, L
        #las alphas no se van a entrenar o qué?
        self.alphas = alphas.detach()
        
        #N, 1, L x N, rnnL, H --> N, 1 , H
        context = torch.bmm(alphas, self.values)
        return context
    
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, 
                input_dim=None, proj_values=True):
        super().__init__()
        self.linear_out=nn.Linear(n_heads*d_model, d_model)
        self.attn_heads=nn.ModuleList(
            [Attention(d_model, 
                      input_dim=input_dim,
                      proj_values=proj_values) for _ in range(n_heads)])
        
    def init_keys(self, key):
        for attn in self.attn_heads:
            attn.init_keys(key)
            
    @property#not totally sure about this
    def alphas(self):
        #shape: n_heads, N, 1, L(source)
        return torch.stack(
        [attn.alphas for attn in self.attn_heads], dim=0
        )
    def output_function(self, context):
        #N, 1, n_heads*D
        concatenated=torch.cat(context, axis=-1)
        #linear transf to go back to original dimension
        out=self.linear_out(concatenated) #N,1,D
        return out
    
    def forward(self, query, mask=None):
        contexts = [attn(query, mask=mask) for attn in self.attn_heads]
        out = self.output_function(contexts)
        return out

class EncoderSelfAttnL(nn.Module):
    def __init__(self, n_heads, d_model, ff_units, n_features=None):
        super().__init__()
        self.n_heads=n_heads 
        self.d_models=d_model
        self.ff_units=ff_units
        self.n_features=n_features
        self.self_attn_heads=MultiHeadAttention(n_heads,d_model,input_dim=n_features)
        self.ffn=nn.Sequential(
        nn.Linear(d_model, ff_units),
        nn.ReLU(),
        nn.Linear(ff_units, d_model),
        )
    def contador(self, batch):
        diccionario={}
        for i in batch:
            if i.item() not in diccionario:
                diccionario[i.item()]=1
            else:
                diccionario[i.item()]+=1
        return diccionario
    
    def batch_sequences_features(self, x, batch):
        N_first=[]
        l_inf=0
        l_sup=0
        dictionary=self.contador(batch)
        for i in dictionary:
            l_sup=l_sup+dictionary[i]
            N_first.append(x[l_inf:l_sup])
            l_inf=l_sup
        return N_first
    
    def my_padding(self, array,batch):
        contador=self.contador(batch)
        max_key=max(contador, key=contador.get)
        max_max=contador[max_key]
        dim=array[0].size()[1]
        for i, j in enumerate(array):
            dif=max_max-len(j)
            all_zeros=torch.zeros(dim).repeat(dif,1)
            array[i]=torch.cat([array[i],all_zeros],0)
        tensor=torch.stack(array)
            
        mask=(tensor!=0).all(axis=2).unsqueeze(1)
        
        return tensor, mask
    def get_back(self, encoded_states, mask):
        get_back=[]
        for i in range(encoded_states.size()[0]):
            for j in range(encoded_states.size()[1]):
                if mask[i][0][j]==True:
                    get_back.append(torch.as_tensor(
                        encoded_states[i][j]).float())
                else:
                    pass
        return torch.stack(get_back)
    # def padded_and_mask(self, batched):
    #     padded=rnn_utils.pad_sequence(batched, batch_first=True)
    #     source_mask=(padded!=0).all(axis=2).unsqueeze(1)
    #     return padded, source_mask
    
    # def pad_collate(self, batch):
    #     x=[item for item in batch]
    #     x_pad=pad_sequence(x, batch_first).detach()
    #     return x_pad

    def forward(self, query, batch):
        new=self.batch_sequences_features(query, batch)
        #query=pad_collate_1(new)
        query,self.mask=self.my_padding(new, batch)
        self.self_attn_heads.init_keys(query)
        att=self.self_attn_heads(query, self.mask)
        out=self.ffn(att)
        return out
    
class DecoderSelfAttn(nn.Module):
    def __init__(self, n_heads, d_model, ff_units, n_features=None):
        super().__init__()
        self.n_heads=n_heads
        self.d_models=d_models
        self.ff_units=ff_units
        self.n_features=d_model if n_features is None else n_features
        self.self._attn_heads=MultiHeadAttention(n_heads, d_model, 
                                                 input_dim=self.n_features)
        self.cross_attn_heads=MultiHeadAttention(n_heads,d_model )
        self.ffn=nn.Sequential(nn.Linear(d_model, ff_units),
                               nn.ReLU(),
                               nn.Linear(ff_units, self.n_features))
    
    def init_keys(self, states):
        self.cross_attn_heads.init_keys(states)
    
    def forward(self, query, source_mask=None, target_mask=None):
        self.self_attn_heads.init_keys(query)
        att1=self.self_attn_heads(query, target_mask)
        att2=self.cross_attn_heads(att1, source_ask)
        out=self.fnn(att2)
        return out
    
    
        
        