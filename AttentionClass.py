#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 09:21:28 2022

@author: leonardo
"""
import copy
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

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
        
        #N, 1, L x N, L, H --> N, 1 , H
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
    
class EncoderSelfAttn(nn.Module):
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
        
    def forward(self, query, mask=None):
        self.self_attn_heads.init_keys(query)
        att=self.self_attn_heads(query, mask)
        out=self.ffn(att)
        return out