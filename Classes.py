#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 10:36:57 2022

@author: leonardo
"""

#Encoder
"""
The encoder's goal is to generate a representation of the 𝐬𝐨𝐮𝐫𝐜𝐞 𝐬𝐞𝐪𝐮𝐞𝐧𝐜𝐞, 
that is, to encode it
- capitan obvious 
"""
import torch
class Encoder(torch.nn.Module):
    def __init__(self, n_features, hidden_dim):
        #n_features are the number of features each data point have
        #while hidden_dim is the number of linear 
        #transformations that will the n_features mapped
        
        super().__init__()
        self.hidden_dim=hidden_dim
        self.n_features=n_features
        self.hidden=None #place  holder
        self.basic_rnn=torch.nn.GRU(self.n_features, 
                                    self.hidden_dim,
                                    batch_first=False) #careful here)
        
        
    def forward(self, X):
        rnn_out, self.hidden=self.basic(X)
        
        return rnn_out
    
#Decoder
"""
The decoder's goal is to generate the target sequence from an initial representation, 
that is, to decode it

In order to decode the hidden state of into a sequence,
 the decoder needs a recurent layer as well 
"""
class Decoder(torch.nn.Module):
    def __init__(self, n_features, hidden_dim):
        super().__init__()
        self.n_features=n_features #target sequence 
        self.hidden_dim=hidden_dim
        self.basic_rnn=torch.nn.GRU(self.n_features,
                                    self.hidden_dim, 
                                    batch_first=False ) #careful again
        
        #regression that need to be change
        self.regression=torch.nn.Linear(self.hidden_dim, n_features)
        
    
    def init_hidden(self, hidden_seq):
        #the final state of the encoder
        hidden_final=hidden_seq[:,-1:,:] #I dont think it came N,1, H any more
        #since rnn is batch_first=False (sequence first)
        #ero, it does not need any permutation #but I will check
        self.hidden=hidden_final
        
    def forward(self, X):
        #Now X must be 1,N, F
        sequence_first_output, self.hidden=self.basic_rnn(X, self.hidden)
        
        last_output= sequence_first_output[:,-1:,:]
        
        #regression, that need to be change 
        
        out=self.regression(last_output)
        
        return out #check output size 
    
#EncoderDecoder 
"""
This class is the main training model 
"""

class EncoderDecoder(torch.nn.Module):
    
    def __init__(self, encoder, decoder, input_len, target_len ,
                 teacher_forcing_prob=.5):
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.input_len=input_len
        self.target_len=target_len
        self.teacher_forcing_prob=teacher_forcing_prob
        self.outputs=None 
        
    def init_outputs(self, batch_size):
        device=next(self.parameters()).device
        
        #el que se usa en el modelo original es
        self.outputs=torch.zeros(batch_size, self.target_len, 
                                 self.encoder.n_features).to(device)
        